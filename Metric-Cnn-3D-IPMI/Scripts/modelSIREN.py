"""
Improved Physics-Informed Neural Network (PINN) with SIREN architecture
and Fourier Features for Metric Estimation.

Improvements over basic PINN:
1. Fourier Features: Random frequency encoding (NeRF-style) instead of fixed frequencies
2. SIREN architecture: Sin activations with special weight initialization
3. Better gradient flow and high-frequency detail capture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def count_parameters(model):
    """Count total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SinActivation(nn.Module):
    """
    Sinusoidal activation function for SIREN.
    """
    def __init__(self, w0=1.0):
        """
        Args:
            w0: Frequency scaling factor (default 1.0, can be 30.0 for first layer)
        """
        super(SinActivation, self).__init__()
        self.w0 = w0
    
    def forward(self, x):
        return torch.sin(self.w0 * x)


class FourierFeatures(nn.Module):
    """
    Fourier Feature encoding (NeRF-style) with learnable or fixed frequencies.
    Better than simple positional encoding for high-frequency details.
    """
    def __init__(self, input_dim=3, num_frequencies=10, learnable=False, scale=1.0):
        """
        Args:
            input_dim: Dimension of input coordinates (3 for x,y,z)
            num_frequencies: Number of frequency bands (L)
            learnable: If True, frequencies are learnable parameters
            scale: Scaling factor for frequency range
        """
        super(FourierFeatures, self).__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.learnable = learnable
        
        if learnable:
            # Learnable frequency matrix: (num_frequencies, input_dim)
            self.frequencies = nn.Parameter(
                torch.randn(num_frequencies, input_dim) * scale
            )
        else:
            # Fixed frequencies: log-spaced from 2^0 to 2^(num_frequencies-1)
            # Similar to NeRF but adapted for 3D
            frequencies = []
            for i in range(num_frequencies):
                freq = 2.0 ** i
                frequencies.append([freq] * input_dim)
            self.register_buffer('frequencies', torch.tensor(frequencies) * scale)
        
        # Output dimension: 2 * num_frequencies * input_dim (sin + cos for each freq)
        self.output_dim = 2 * num_frequencies * input_dim
    
    def forward(self, x):
        """
        Args:
            x: (..., input_dim) tensor of coordinates
        
        Returns:
            encoded: (..., output_dim) tensor of Fourier features
        """
        # x shape: (..., input_dim)
        # frequencies shape: (num_frequencies, input_dim)
        
        # Expand for each input dimension (x, y, z separately)
        # This gives us more expressive power
        encoded_list = []
        for dim_idx in range(self.input_dim):
            x_dim = x[..., dim_idx]  # (...,) - scalar for this dimension
            # Get frequencies for this dimension: (num_frequencies,)
            freq_dim = self.frequencies[:, dim_idx]  # (num_frequencies,)
            # Project: x_dim[..., None] * freq_dim[None, :] -> (..., num_frequencies)
            x_proj_dim = x_dim[..., None] * freq_dim[None, :]  # (..., num_frequencies)
            encoded_list.append(torch.sin(2 * np.pi * x_proj_dim))
            encoded_list.append(torch.cos(2 * np.pi * x_proj_dim))
        
        encoded = torch.cat(encoded_list, dim=-1)  # (..., 2 * num_frequencies * input_dim)
        
        return encoded


class MetricSIREN(nn.Module):
    """
    Improved PINN with SIREN architecture and Fourier Features.
    
    Key improvements:
    1. Fourier Features: Better high-frequency detail capture
    2. SIREN activations: Sin activations with special initialization
    3. Preserved gradients: Better for physics-informed training
    """
    def __init__(self, hidden_layers=[256, 512, 512, 256, 128], 
                 w0=30.0, w0_first=30.0,
                 num_frequencies=10, fourier_learnable=False, fourier_scale=1.0,
                 use_fourier_features=True):
        """
        Args:
            hidden_layers (list): List of hidden layer widths
            w0: Frequency scaling for hidden layers (default 30.0 for SIREN)
            w0_first: Frequency scaling for first layer (can be different)
            num_frequencies: Number of frequency bands for Fourier Features
            fourier_learnable: If True, Fourier frequencies are learnable
            fourier_scale: Scaling factor for Fourier frequencies
            use_fourier_features: Enable Fourier Features encoding
        """
        super(MetricSIREN, self).__init__()
        
        self.use_fourier_features = use_fourier_features
        self.w0 = w0
        self.w0_first = w0_first
        
        # Fourier Features encoding
        if use_fourier_features:
            self.fourier_encoder = FourierFeatures(
                input_dim=3,
                num_frequencies=num_frequencies,
                learnable=fourier_learnable,
                scale=fourier_scale
            )
            # Input: Fourier features (2*num_frequencies*3) + vector field (3)
            input_dim = self.fourier_encoder.output_dim + 3
        else:
            self.fourier_encoder = None
            # Input: coordinates (3) + vector field (3)
            input_dim = 6
        
        # Build SIREN network
        layers = []
        prev_dim = input_dim
        
        # First layer with special initialization
        first_layer = nn.Linear(prev_dim, hidden_layers[0])
        layers.append(first_layer)
        layers.append(SinActivation(w0=w0_first))
        prev_dim = hidden_layers[0]
        
        # Hidden layers with SIREN activations
        for hidden_dim in hidden_layers[1:]:
            layer = nn.Linear(prev_dim, hidden_dim)
            layers.append(layer)
            layers.append(SinActivation(w0=w0))
            prev_dim = hidden_dim
        
        # Output layer (no activation, linear output)
        output_layer = nn.Linear(prev_dim, 7)
        layers.append(output_layer)
        
        self.network = nn.Sequential(*layers)
        
        # SIREN-specific weight initialization
        self._initialize_siren_weights()
        
        # Print parameter count
        n_params = count_parameters(self)
        print(f'[MetricSIREN] Total parameters: {n_params:,}')
        print(f'[MetricSIREN] Fourier Features: {use_fourier_features}, '
              f'Frequencies: {num_frequencies}, Learnable: {fourier_learnable}')
    
    def _initialize_siren_weights(self):
        """
        SIREN weight initialization: preserves gradients through sin activations.
        
        For layer with input size n:
        - Weights: Uniform distribution in [-sqrt(6/n) / w0, sqrt(6/n) / w0]
        - First layer uses w0_first, others use w0
        """
        with torch.no_grad():
            layer_idx = 0
            for module in self.network.modules():
                if isinstance(module, nn.Linear):
                    n = module.in_features
                    
                    # Determine w0 for this layer
                    if layer_idx == 0:  # First layer
                        w0_used = self.w0_first
                    else:
                        w0_used = self.w0
                    
                    # SIREN initialization
                    bound = np.sqrt(6.0 / n) / w0_used
                    module.weight.uniform_(-bound, bound)
                    
                    # Bias initialization (small)
                    if module.bias is not None:
                        module.bias.uniform_(-1e-4, 1e-4)
                    
                    layer_idx += 1
    
    def forward(self, coords, vector_field):
        """
        Forward pass.
        
        Args:
            coords: (B, 3, H, W, D) or (3, H, W, D) tensor of spatial coordinates
            vector_field: (B, 3, H, W, D) or (3, H, W, D) tensor of vector field
        
        Returns:
            output: (B, 7, H, W, D) or (7, H, W, D) tensor of metric components
        """
        # Handle batch dimension - ensure both have same number of dimensions
        if coords.dim() == 4 and vector_field.dim() == 5:
            coords = coords.unsqueeze(0).expand(vector_field.shape[0], -1, -1, -1, -1)
            squeeze_output = False
        elif coords.dim() == 4 and vector_field.dim() == 4:
            coords = coords.unsqueeze(0)
            vector_field = vector_field.unsqueeze(0)
            squeeze_output = True
        elif coords.dim() == 5 and vector_field.dim() == 4:
            vector_field = vector_field.unsqueeze(0).expand(coords.shape[0], -1, -1, -1, -1)
            squeeze_output = False
        else:
            if coords.dim() == 4:
                coords = coords.unsqueeze(0)
                vector_field = vector_field.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
        
        B, C, H, W, D = coords.shape
        assert C == 3, "Coordinates must have 3 channels (x, y, z)"
        assert vector_field.shape[1] == 3, "Vector field must have 3 channels"
        assert vector_field.shape[0] == B, "Batch sizes must match"
        
        # Reshape to (B*H*W*D, 3) for processing
        coords_flat = coords.permute(0, 2, 3, 4, 1).reshape(-1, 3)  # (B*H*W*D, 3)
        vector_flat = vector_field.permute(0, 2, 3, 4, 1).reshape(-1, 3)  # (B*H*W*D, 3)
        
        # Apply Fourier Features encoding if enabled
        if self.use_fourier_features:
            encoded_coords = self.fourier_encoder(coords_flat)  # (B*H*W*D, fourier_dim)
            input_features = torch.cat([encoded_coords, vector_flat], dim=-1)
        else:
            # Use raw coordinates
            input_features = torch.cat([coords_flat, vector_flat], dim=-1)  # (B*H*W*D, 6)
        
        # Forward through SIREN network
        output_flat = self.network(input_features)  # (B*H*W*D, 7)
        
        # Reshape back to (B, 7, H, W, D)
        output = output_flat.reshape(B, H, W, D, 7).permute(0, 4, 1, 2, 3)  # (B, 7, H, W, D)
        
        if squeeze_output:
            output = output.squeeze(0)  # (7, H, W, D)
        
        return output


def create_siren_model(hidden_layers=[256, 512, 512, 256, 128],
                       w0=30.0, w0_first=30.0,
                       num_frequencies=10, fourier_learnable=False, fourier_scale=1.0,
                       use_fourier_features=True):
    """
    Factory function to create a SIREN model with Fourier Features.
    
    This architecture is designed to have similar parameter count to CEDNN with blocks=[40,30,40].
    Estimated parameter count: ~600K-800K parameters.
    
    Args:
        hidden_layers (list): List of hidden layer widths. Default gives ~600K-800K params.
        w0: Frequency scaling for hidden layers (SIREN default: 30.0)
        w0_first: Frequency scaling for first layer
        num_frequencies: Number of frequency bands for Fourier Features
        fourier_learnable: If True, Fourier frequencies are learnable parameters
        fourier_scale: Scaling factor for Fourier frequency range
        use_fourier_features: Enable Fourier Features encoding
    
    Returns:
        model: MetricSIREN instance
    """
    model = MetricSIREN(
        hidden_layers=hidden_layers,
        w0=w0,
        w0_first=w0_first,
        num_frequencies=num_frequencies,
        fourier_learnable=fourier_learnable,
        fourier_scale=fourier_scale,
        use_fourier_features=use_fourier_features
    )
    return model

