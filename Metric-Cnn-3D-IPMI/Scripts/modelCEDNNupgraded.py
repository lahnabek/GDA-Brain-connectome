"""
Convolutional Encoder-Decoder Networks with Positional Encoding and Fourier Features
Based on model3D.py with added options for:
1. Positional encoding (normalized coordinates as additional channels)
2. Fourier features (NeRF-style encoding with multiple frequencies)

Reference:
    https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
    

def module_size(module):
    assert isinstance(module, torch.nn.Module)
    n_params, n_conv_layers = 0, 0
    for name, param in module.named_parameters():
        if 'conv' in name:
            n_conv_layers += 1
        n_params += param.numel()
    return n_params, n_conv_layers


class UpsamplingNearest3d(nn.Module):
    def __init__(self, scale_factor=2.):
        super().__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
    

class UpsamplingTrilinear3d(nn.Module):
    def __init__(self, scale_factor=2.):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, 
            mode='trilinear', align_corners=True)


class _DenseLayer(nn.Sequential):
    """One dense layer within dense block, with bottleneck design.
    Args:
        in_features (int):
        growth_rate (int): # out feature maps of every dense layer
        drop_rate (float): 
        bn_size (int): Specifies maximum # features is `bn_size` * 
            `growth_rate`
        bottleneck (bool, False): If True, enable bottleneck design
    """
    def __init__(self, in_features, growth_rate, drop_rate=0., bn_size=8,
                 bottleneck=False):
        super(_DenseLayer, self).__init__()
        if bottleneck and in_features > bn_size * growth_rate:
            self.add_module('norm1', nn.BatchNorm3d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv3d(in_features, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False))
            self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
            self.add_module('relu2', nn.ReLU(inplace=True))
            self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.add_module('norm1', nn.BatchNorm3d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv3d(in_features, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False))
        if drop_rate > 0:
            self.add_module('dropout', nn.Dropout3d(p=drop_rate))
        
    def forward(self, x):
        y = super(_DenseLayer, self).forward(x)
        return torch.cat([x, y], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_features, growth_rate, drop_rate,
                 bn_size=4, bottleneck=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_features + i * growth_rate, growth_rate,
                                drop_rate=drop_rate, bn_size=bn_size,
                                bottleneck=bottleneck)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, in_features, out_features, down, bottleneck=True, 
                 drop_rate=0, upsample='nearest'):
        """Transition layer, either downsampling or upsampling, both reduce
        number of feature maps, i.e. `out_features` should be less than 
        `in_features`.
        Args:
            in_features (int):
            out_features (int):
            down (bool): If True, downsampling, else upsampling
            bottleneck (bool, True): If True, enable bottleneck design
            drop_rate (float, 0.):
        """
        super(_Transition, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(in_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        if down:
            # half feature resolution, reduce # feature maps
            if bottleneck:
                # bottleneck impl, save memory, add nonlinearity
                self.add_module('conv1', nn.Conv3d(in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout3d(p=drop_rate))
                self.add_module('norm2', nn.BatchNorm3d(out_features))
                self.add_module('relu2', nn.ReLU(inplace=True))
                # self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))
                # not using pooling, fully convolutional...
                self.add_module('conv2', nn.Conv3d(out_features, out_features,
                    kernel_size=3, stride=2, padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout3d(p=drop_rate))
            else:
                self.add_module('conv1', nn.Conv3d(in_features, out_features,
                    kernel_size=3, stride=2, padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout3d(p=drop_rate))
        else:
            # transition up, increase feature resolution, half # feature maps
            if bottleneck:
                # bottleneck impl, save memory, add nonlinearity
                self.add_module('conv1', nn.Conv3d(in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout3d(p=drop_rate))

                self.add_module('norm2', nn.BatchNorm3d(out_features))
                self.add_module('relu2', nn.ReLU(inplace=True))
                # output_padding=0, or 1 depends on the image size
                # if image size is of the power of 2, then 1 is good
                if upsample is None:
                    self.add_module('convT2', nn.ConvTranspose3d(
                        out_features, out_features, kernel_size=3, stride=2,
                        padding=1, output_padding=1, bias=False))
                elif upsample == 'trilinear':
                    self.add_module('upsample', UpsamplingTrilinear3d(scale_factor=2))
                    self.add_module('conv2', nn.Conv3d(out_features, out_features,
                        3, 1, 1, bias=False))
                elif upsample == 'nearest':
                    self.add_module('upsample', UpsamplingNearest3d(scale_factor=2))
                    self.add_module('conv2', nn.Conv3d(out_features, out_features,
                        3, 1, 1, bias=False))

                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout3d(p=drop_rate))
            else:
                # TODO modify convT
                self.add_module('convT1', nn.ConvTranspose3d(
                    out_features, out_features, kernel_size=3, stride=2,
                    padding=1, output_padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout3d(p=drop_rate))


def last_decoding(in_features, out_channels, bias=False, drop_rate=0., upsample='nearest'):
    """Last transition up layer, which outputs directly the predictions.
    """
    last_up = nn.Sequential()
    last_up.add_module('norm1', nn.BatchNorm3d(in_features))
    last_up.add_module('relu1', nn.ReLU(True))
    last_up.add_module('conv1', nn.Conv3d(in_features, in_features // 2, 
                    kernel_size=3, stride=1, padding=1, bias=False))
    if drop_rate > 0.:
        last_up.add_module('dropout1', nn.Dropout3d(p=drop_rate))
    last_up.add_module('norm2', nn.BatchNorm3d(in_features // 2))
    last_up.add_module('relu2', nn.ReLU(True))
    # last_up.add_module('convT2', nn.ConvTranspose3d(in_features // 2, 
    #                    out_channels, kernel_size=kernel_size, stride=stride, 
    #                    padding=padding, output_padding=output_padding, bias=bias))
    if upsample == 'nearest':
        last_up.add_module('upsample', UpsamplingNearest3d(scale_factor=2))
    elif upsample == 'trilinear':
        last_up.add_module('upsample', UpsamplingTrilinear3d(scale_factor=2))
    last_up.add_module('conv2', nn.Conv3d(in_features // 2, in_features // 4,
        kernel_size=3, stride=1, padding=1, bias=bias))
    last_up.add_module('norm3', nn.BatchNorm3d(in_features // 4))
    last_up.add_module('relu3', nn.ReLU(True))
    last_up.add_module('conv3', nn.Conv3d(in_features // 4, out_channels,
        kernel_size=5, stride=1, padding=2, bias=bias))
    return last_up


def activation(name):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    else:
        raise ValueError('Unknown activation function')


class PositionalEncoding(nn.Module):
    """
    Positional encoding (normalized coordinates) with optional learnable scaling and offset.
    
    Args:
        learnable_scale: If True, learnable scaling per dimension (3 parameters)
        learnable_offset: If True, learnable offset/bias per dimension (3 parameters)
    """
    def __init__(self, learnable_scale=False, learnable_offset=False):
        super(PositionalEncoding, self).__init__()
        self.learnable_scale = learnable_scale
        self.learnable_offset = learnable_offset
        
        if learnable_scale:
            # Learnable scaling per dimension (x, y, z)
            # Initialize to 1.0 to start with standard [-1, 1] range
            self.scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        else:
            self.register_buffer('scale', torch.ones(3, dtype=torch.float32))
        
        if learnable_offset:
            # Learnable offset per dimension (x, y, z)
            # Initialize to 0.0
            self.offset = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        else:
            self.register_buffer('offset', torch.zeros(3, dtype=torch.float32))
    
    def forward(self, input_tensor):
        """
        Create positional encoding (normalized coordinates) for input tensor.
        
        Args:
            input_tensor: (B, C, H, W, D) tensor
        
        Returns:
            coords: (B, 3, H, W, D) tensor with normalized coordinates
                   (optionally scaled and offset if learnable parameters are enabled)
        """
        B, C, H, W, D = input_tensor.shape
        device = input_tensor.device
        dtype = input_tensor.dtype
        
        # Create normalized coordinate grids in [-1, 1]
        coords_x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        coords_y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        coords_z = torch.linspace(-1, 1, D, device=device, dtype=dtype)
        
        # Expand to full grid
        coords_x = coords_x.view(1, 1, 1, W, 1).expand(B, 1, H, W, D)
        coords_y = coords_y.view(1, 1, H, 1, 1).expand(B, 1, H, W, D)
        coords_z = coords_z.view(1, 1, 1, 1, D).expand(B, 1, H, W, D)
        
        # Apply learnable scaling and offset
        # coords = scale * coords + offset
        coords_x = self.scale[0] * coords_x + self.offset[0]
        coords_y = self.scale[1] * coords_y + self.offset[1]
        coords_z = self.scale[2] * coords_z + self.offset[2]
        
        # Concatenate
        coords = torch.cat([coords_x, coords_y, coords_z], dim=1)  # (B, 3, H, W, D)
        
        return coords


class FourierFeatures(nn.Module):
    """
    Fourier Feature encoding (NeRF-style) with learnable or fixed frequencies.
    Better than simple positional encoding for high-frequency details.
    """
    def __init__(self, num_frequencies=10, learnable=False, scale=1.0):
        """
        Args:
            num_frequencies: Number of frequency bands (L)
            learnable: If True, frequencies are learnable parameters
            scale: Scaling factor for frequency range
        """
        super(FourierFeatures, self).__init__()
        self.num_frequencies = num_frequencies
        self.learnable = learnable
        self.scale = scale
        
        if learnable:
            # Learnable frequency matrix: (num_frequencies, 3) for x, y, z
            # Initialize with log-spaced frequencies
            frequencies_init = []
            for i in range(num_frequencies):
                freq = 2.0 ** i * np.pi
                frequencies_init.append([freq, freq, freq])
            self.frequencies = nn.Parameter(
                torch.tensor(frequencies_init, dtype=torch.float32) * scale
            )
        else:
            # Fixed frequencies: log-spaced from 2^0*π to 2^(num_frequencies-1)*π
            frequencies = []
            for i in range(num_frequencies):
                freq = 2.0 ** i * np.pi
                frequencies.append([freq, freq, freq])
            self.register_buffer('frequencies', torch.tensor(frequencies, dtype=torch.float32) * scale)
        
        # Output dimension: 3 * 2 * num_frequencies (sin + cos for each freq, for each dim)
        self.output_dim = 3 * 2 * num_frequencies
    
    def forward(self, input_tensor):
        """
        Create Fourier features (NeRF-style encoding) for input tensor.
        
        Args:
            input_tensor: (B, C, H, W, D) tensor
        
        Returns:
            fourier_features: (B, 3*2*num_frequencies, H, W, D) tensor
        """
        B, C, H, W, D = input_tensor.shape
        device = input_tensor.device
        dtype = input_tensor.dtype
        
        # Create normalized coordinate grids
        coords_x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        coords_y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        coords_z = torch.linspace(-1, 1, D, device=device, dtype=dtype)
        
        # Expand to full grid
        coords_x = coords_x.view(1, 1, 1, W, 1).expand(B, 1, H, W, D)
        coords_y = coords_y.view(1, 1, H, 1, 1).expand(B, 1, H, W, D)
        coords_z = coords_z.view(1, 1, 1, 1, D).expand(B, 1, H, W, D)
        
        # Encode with Fourier features
        fourier_list = []
        for l in range(self.num_frequencies):
            # Get frequency for this level (can be different for x, y, z if learnable)
            freq_x = self.frequencies[l, 0]
            freq_y = self.frequencies[l, 1]
            freq_z = self.frequencies[l, 2]
            
            fourier_list.append(torch.sin(freq_x * coords_x))
            fourier_list.append(torch.cos(freq_x * coords_x))
            fourier_list.append(torch.sin(freq_y * coords_y))
            fourier_list.append(torch.cos(freq_y * coords_y))
            fourier_list.append(torch.sin(freq_z * coords_z))
            fourier_list.append(torch.cos(freq_z * coords_z))
        
        fourier_features = torch.cat(fourier_list, dim=1)  # (B, 3*2*num_frequencies, H, W, D)
        
        return fourier_features


# modify the decoder network
# use upsampling instead of transconv
# it seems tranconv is much faster than neareast upsampling (or other interpo)

class DenseED(nn.Module):
    def __init__(self, in_channels, out_channels, imsize, blocks, growth_rate=16,
                 init_features=48, drop_rate=0, bn_size=8, bottleneck=False, 
                 out_activation=None, upsample='nearest',
                 use_positional_encoding=False, use_fourier_features=False, 
                 num_fourier_frequencies=10, fourier_learnable=False, fourier_scale=1.0,
                 pe_learnable_scale=False, pe_learnable_offset=False):
        """Dense Convolutional Encoder-Decoder Networks.
        Decoder: Upsampling + Conv instead of TransposeConv 
        
        Args:
            in_channels (int): number of input channels (vector field: 3)
            out_channels (int): number of output channels (metric parameters: 7)
            imsize (int): imsize size, assume squared image
            blocks (list-like): A list (of odd size) of integers
            growth_rate (int): K
            init_features (int): number of feature maps after first conv layer
            bn_size: bottleneck size for number of feature maps (not used)
            bottleneck (bool): use bottleneck for dense block or not (False)
            drop_rate (float): dropout rate
            out_activation: Output activation function, choices=[None, 'tanh',
                'sigmoid', 'softplus']
            use_positional_encoding (bool): If True, add 3 channels with normalized coordinates
            use_fourier_features (bool): If True, add Fourier-encoded coordinates (3*2*num_fourier_frequencies channels)
            num_fourier_frequencies (int): Number of frequency levels for Fourier features (default: 10)
            fourier_learnable (bool): If True, Fourier frequencies are learnable parameters (default: False)
            fourier_scale (float): Scaling factor for Fourier frequencies (default: 1.0)
            pe_learnable_scale (bool): If True, positional encoding has learnable scaling per dimension (default: False)
            pe_learnable_offset (bool): If True, positional encoding has learnable offset per dimension (default: False)
        """
        super(DenseED, self).__init__()
        
        # Store encoding options
        self.use_positional_encoding = use_positional_encoding
        self.use_fourier_features = use_fourier_features
        self.num_fourier_frequencies = num_fourier_frequencies
        
        # Create encoding modules
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                learnable_scale=pe_learnable_scale,
                learnable_offset=pe_learnable_offset
            )
        else:
            self.positional_encoding = None
        
        if use_fourier_features:
            self.fourier_features = FourierFeatures(
                num_frequencies=num_fourier_frequencies,
                learnable=fourier_learnable,
                scale=fourier_scale
            )
        else:
            self.fourier_features = None
        
        # Calculate actual input channels (vector field + encodings)
        actual_in_channels = in_channels
        if use_positional_encoding:
            actual_in_channels += 3  # Add 3 coordinate channels
        if use_fourier_features:
            actual_in_channels += 3 * 2 * num_fourier_frequencies  # Add Fourier features
        
        if len(blocks) > 1 and len(blocks) % 2 == 0:
            raise ValueError('length of blocks must be an odd number, but got {}'
                            .format(len(blocks)))
        enc_block_layers = blocks[: len(blocks) // 2]
        dec_block_layers = blocks[len(blocks) // 2:]

        self.features = nn.Sequential()
        pad = 3 if imsize % 2 == 0 else 2
        # First convolution, half image size ================
        # For even image size: k7s2p3, k5s2p2
        # For odd image size (e.g. 65): k7s2p2, k5s2p1, k13s2p5, k11s2p4, k9s2p3
        self.features.add_module('In_conv', nn.Conv3d(actual_in_channels, init_features, 
                              kernel_size=7, stride=2, padding=pad, bias=False))
        # Encoding / transition down ================
        # dense block --> encoding --> dense block --> encoding
        num_features = init_features
        for i, num_layers in enumerate(enc_block_layers):
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                bn_size=bn_size, 
                                growth_rate=growth_rate,
                                drop_rate=drop_rate, 
                                bottleneck=bottleneck)
            self.features.add_module('EncBlock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            trans_down = _Transition(in_features=num_features,
                                     out_features=num_features // 2,
                                     down=True, 
                                     drop_rate=drop_rate)
            self.features.add_module('TransDown%d' % (i + 1), trans_down)
            num_features = num_features // 2
        # Decoding / transition up ==============
        # dense block --> decoding --> dense block --> decoding --> dense block
        for i, num_layers in enumerate(dec_block_layers):
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                bn_size=bn_size, 
                                growth_rate=growth_rate,
                                drop_rate=drop_rate, 
                                bottleneck=bottleneck)
            self.features.add_module('DecBlock%d' % (i + 1), block)
            num_features += num_layers * growth_rate
            # the last decoding layer has different convT parameters
            if i < len(dec_block_layers) - 1:
                trans_up = _Transition(in_features=num_features,
                                    out_features=num_features // 2,
                                    down=False, 
                                    drop_rate=drop_rate,
                                    upsample=upsample)
                self.features.add_module('TransUp%d' % (i + 1), trans_up)
                num_features = num_features // 2
        
        # The last decoding layer =======
        last_trans_up = last_decoding(num_features, out_channels, 
            drop_rate=drop_rate, upsample=upsample)
        self.features.add_module('LastTransUp', last_trans_up)

        if out_activation is not None:
            self.features.add_module(out_activation, activation(out_activation))
        
        print('# params {}, # conv layers {}'.format(
            *self.model_size))
        if use_positional_encoding:
            pe_params = []
            if pe_learnable_scale:
                pe_params.append("learnable scale")
            if pe_learnable_offset:
                pe_params.append("learnable offset")
            pe_str = " (" + ", ".join(pe_params) + ")" if pe_params else ""
            print('  [Positional Encoding: ENABLED{}]'.format(pe_str))
        if use_fourier_features:
            learnable_str = " (learnable)" if fourier_learnable else " (fixed)"
            print('  [Fourier Features: ENABLED ({} frequencies{}, scale={})]'.format(
                num_fourier_frequencies, learnable_str, fourier_scale))

    def forward(self, x):
        """
        Forward pass with optional positional encoding and Fourier features.
        
        Args:
            x: (B, in_channels, H, W, D) tensor - input vector field
        
        Returns:
            output: (B, out_channels, H, W, D) tensor - metric parameters
        """
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            coords = self.positional_encoding(x)
            x = torch.cat([x, coords], dim=1)
        
        # Add Fourier features if enabled
        if self.use_fourier_features:
            fourier = self.fourier_features(x)
            x = torch.cat([x, fourier], dim=1)
        
        return self.features(x)

    def forward_test(self, x):
        print('input: {}'.format(x.data.size()))
        for name, module in self.features._modules.items():
            x = module(x)
            print('{}: {}'.format(name, x.data.size()))
        return x

    @property
    def model_size(self):
        return module_size(self)

    def reset_parameters(self, verbose=False):
        for module in self.modules():
            # pass self, otherwise infinite loop
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))


class Decoder(nn.Module):
    """
    Decoder to solve one PDE
    Use nearest upsampling + Conv3d to replace TransposedConv3d
    """
    def __init__(self, dim_latent, out_channels, blocks, 
        growth_rate=16, init_features=48, drop_rate=0., upsample='nearest',
        out_activation=None):
        super(Decoder, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv3d(dim_latent, init_features, 3, 1, 1, bias=False))
        num_features = init_features 
        for i, num_layers in enumerate(blocks):
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('DecBlock%d' % (i + 1), block)
            num_features += num_layers * growth_rate
            # the last decoding layer has different convT parameters
            if i < len(blocks) - 1:
                trans_up = _Transition(in_features=num_features,
                                    out_features=num_features // 2,
                                    down=False, 
                                    drop_rate=drop_rate,
                                    upsample=upsample)
                self.features.add_module('TransUp%d' % (i + 1), trans_up)
                num_features = num_features // 2
        
        # The last decoding layer =======
        last_trans_up = last_decoding(num_features, out_channels, 
            drop_rate=drop_rate, upsample=upsample)
        self.features.add_module('LastTransUp', last_trans_up)

        if out_activation is not None:
            self.features.add_module(out_activation, activation(out_activation))

    @property
    def model_size(self):
        return module_size(self)

    def forward(self, x):
        return self.features(x)

    def forward_test(self, x):
        print('input: {}'.format(x.data.size()))
        for name, module in self.features._modules.items():
            x = module(x)
            print('{}: {}'.format(name, x.data.size()))
        return x
