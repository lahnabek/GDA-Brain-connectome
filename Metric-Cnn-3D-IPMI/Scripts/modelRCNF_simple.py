"""
Simplified Riemannian Neural Network (RCNF-Simple) for Metric Estimation.

This is a simplified version of RCNF that removes the ODE integration
and uses a direct MLP transformation instead. This should be:
- Faster (no ODE integration)
- More stable (no numerical integration errors)
- Better performance (simpler architecture, easier to train)

Architecture:
1. Encodes coordinates and vector field into a latent space (with Fourier Features)
2. Transforms via deep MLP with residual connections
3. Decodes to metric components

Improvements over original RCNF:
- No ODE (direct MLP transformation)
- Residual connections for stability
- Better initialization
- Layer normalization for stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def count_parameters(model):
    """Count total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CoordinateEncoder(nn.Module):
    """
    Encodes spatial coordinates and vector field into a latent representation.
    Uses Fourier Features for better high-frequency detail capture.
    """
    def __init__(self, coord_dim=3, vector_dim=3, num_frequencies=10, hidden_dim=128):
        """
        Args:
            coord_dim: Dimension of coordinates (3 for x,y,z)
            vector_dim: Dimension of vector field (3 for vx,vy,vz)
            num_frequencies: Number of Fourier frequency bands
            hidden_dim: Dimension of encoded output
        """
        super(CoordinateEncoder, self).__init__()
        
        self.coord_dim = coord_dim
        self.vector_dim = vector_dim
        self.num_frequencies = num_frequencies
        
        # Fourier Features for coordinates
        # Fixed frequencies: log-spaced from 2^0 to 2^(num_frequencies-1)
        frequencies = []
        for i in range(num_frequencies):
            freq = 2.0 ** i
            frequencies.append([freq] * coord_dim)
        self.register_buffer('frequencies', torch.tensor(frequencies))
        
        # Fourier encoding dimension: 2 * num_frequencies * coord_dim
        fourier_dim = 2 * num_frequencies * coord_dim
        
        # Input: Fourier features (fourier_dim) + vector field (vector_dim)
        input_dim = fourier_dim + vector_dim
        
        # Encoder network with layer normalization for stability
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),  # Layer norm for stability
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.output_dim = hidden_dim
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with better initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization (better for Tanh)
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='tanh')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, coords, vector_field):
        """
        Encode coordinates and vector field.
        
        Args:
            coords: (..., coord_dim) tensor of coordinates
            vector_field: (..., vector_dim) tensor of vector field
        
        Returns:
            encoded: (..., output_dim) tensor of encoded features
        """
        # Apply Fourier Features to coordinates
        encoded_list = []
        for dim_idx in range(self.coord_dim):
            coord_dim = coords[..., dim_idx]  # (...,)
            freq_dim = self.frequencies[:, dim_idx]  # (num_frequencies,)
            coord_proj = coord_dim[..., None] * freq_dim[None, :]  # (..., num_frequencies)
            encoded_list.append(torch.sin(2 * np.pi * coord_proj))
            encoded_list.append(torch.cos(2 * np.pi * coord_proj))
        
        fourier_features = torch.cat(encoded_list, dim=-1)  # (..., fourier_dim)
        
        # Concatenate with vector field
        input_features = torch.cat([fourier_features, vector_field], dim=-1)
        
        # Encode
        encoded = self.encoder(input_features)
        
        return encoded


class ResidualMLPBlock(nn.Module):
    """
    Residual block for MLP with layer normalization.
    Helps with gradient flow and stability.
    """
    def __init__(self, dim, hidden_dim=None):
        """
        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension (default: dim * 2)
        """
        super(ResidualMLPBlock, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = dim * 2
        
        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, x):
        return x + self.block(x)  # Residual connection


class TransformationMLP(nn.Module):
    """
    Deep MLP that replaces the ODE integration.
    Uses residual connections for better gradient flow.
    """
    def __init__(self, input_dim, hidden_dims=[256, 512, 512, 256], output_dim=64, use_residual=True):
        """
        Args:
            input_dim: Dimension of input (encoded features)
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output (state for decoder)
            use_residual: Use residual connections
        """
        super(TransformationMLP, self).__init__()
        
        self.use_residual = use_residual
        
        layers = []
        prev_dim = input_dim
        
        # First layer
        layers.append(nn.Linear(prev_dim, hidden_dims[0]))
        layers.append(nn.LayerNorm(hidden_dims[0]))
        layers.append(nn.Tanh())
        prev_dim = hidden_dims[0]
        
        # Hidden layers with residual connections
        for i, hidden_dim in enumerate(hidden_dims[1:]):
            if use_residual and prev_dim == hidden_dim:
                # Use residual block if dimensions match
                layers.append(ResidualMLPBlock(prev_dim))
            else:
                # Regular layer if dimensions don't match
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.Tanh())
                prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with better initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization (better for Tanh)
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='tanh')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Transform encoded features to state representation.
        
        Args:
            x: (..., input_dim) encoded features
        
        Returns:
            output: (..., output_dim) transformed state
        """
        return self.network(x)


class MetricDecoder(nn.Module):
    """
    Decodes the transformed state into metric components.
    """
    def __init__(self, state_dim, hidden_dims=[256, 128]):
        """
        Args:
            state_dim: Dimension of state vector
            hidden_dims: List of hidden layer dimensions
        """
        super(MetricDecoder, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        # Output: 7 components for metric (same as eigen_composite format)
        layers.append(nn.Linear(prev_dim, 7))
        
        self.decoder = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='tanh')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state):
        """
        Decode state to metric components.
        
        Args:
            state: (..., state_dim) tensor of transformed state
        
        Returns:
            metric_components: (..., 7) tensor of metric components
        """
        return self.decoder(state)


class RiemannianCNF_Simple(nn.Module):
    """
    Simplified Riemannian Neural Network (without ODE).
    
    Architecture:
    1. Encode coordinates and vector field (with Fourier Features)
    2. Transform via deep MLP (replaces ODE)
    3. Decode to metric components
    
    This should be faster, more stable, and perform better than the ODE version.
    """
    def __init__(self, 
                 coord_dim=3,
                 vector_dim=3,
                 encoder_hidden_dim=128,
                 encoder_num_frequencies=10,
                 transformation_hidden_dims=[256, 512, 512, 256],
                 state_dim=64,
                 decoder_hidden_dims=[256, 128],
                 use_residual=True):
        """
        Args:
            coord_dim: Dimension of coordinates (3)
            vector_dim: Dimension of vector field (3)
            encoder_hidden_dim: Hidden dimension for coordinate encoder
            encoder_num_frequencies: Number of Fourier frequencies
            transformation_hidden_dims: Hidden dimensions for transformation MLP
            state_dim: Dimension of state vector (input to decoder)
            decoder_hidden_dims: Hidden dimensions for metric decoder
            use_residual: Use residual connections in transformation MLP
        """
        super(RiemannianCNF_Simple, self).__init__()
        
        self.coord_dim = coord_dim
        self.vector_dim = vector_dim
        self.state_dim = state_dim
        
        # Coordinate encoder
        self.coord_encoder = CoordinateEncoder(
            coord_dim=coord_dim,
            vector_dim=vector_dim,
            num_frequencies=encoder_num_frequencies,
            hidden_dim=encoder_hidden_dim
        )
        
        # Transformation MLP (replaces ODE)
        self.transformation_mlp = TransformationMLP(
            input_dim=self.coord_encoder.output_dim,
            hidden_dims=transformation_hidden_dims,
            output_dim=state_dim,
            use_residual=use_residual
        )
        
        # Metric decoder
        self.metric_decoder = MetricDecoder(
            state_dim=state_dim,
            hidden_dims=decoder_hidden_dims
        )
        
        # Print parameter count
        n_params = count_parameters(self)
        print(f'[RiemannianCNF-Simple] Total parameters: {n_params:,}')
        print(f'[RiemannianCNF-Simple] Architecture: Encoder → MLP → Decoder (no ODE)')
        print(f'[RiemannianCNF-Simple] Residual connections: {use_residual}')
    
    def forward(self, coords, vector_field):
        """
        Forward pass.
        
        Args:
            coords: (B, 3, H, W, D) or (3, H, W, D) tensor of spatial coordinates
            vector_field: (B, 3, H, W, D) or (3, H, W, D) tensor of vector field
        
        Returns:
            output: (B, 7, H, W, D) or (7, H, W, D) tensor of metric components
        """
        # Handle batch dimension
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
        
        # Encode coordinates and vector field
        context = self.coord_encoder(coords_flat, vector_flat)  # (B*H*W*D, context_dim)
        
        # Transform via MLP (replaces ODE integration)
        transformed_state = self.transformation_mlp(context)  # (B*H*W*D, state_dim)
        
        # Decode to metric components
        metric_components = self.metric_decoder(transformed_state)  # (B*H*W*D, 7)
        
        # Reshape back to (B, 7, H, W, D)
        output = metric_components.reshape(B, H, W, D, 7).permute(0, 4, 1, 2, 3)  # (B, 7, H, W, D)
        
        if squeeze_output:
            output = output.squeeze(0)  # (7, H, W, D)
        
        return output


def create_rcnf_simple_model(coord_dim=3,
                              vector_dim=3,
                              encoder_hidden_dim=128,
                              encoder_num_frequencies=10,
                              transformation_hidden_dims=[256, 512, 512, 256],
                              state_dim=64,
                              decoder_hidden_dims=[256, 128],
                              use_residual=True):
    """
    Factory function to create a simplified RCNF model (without ODE).
    
    This architecture is designed to have similar parameter count to CEDNN with blocks=[40,30,40].
    Estimated parameter count: ~600K-800K parameters.
    
    Args:
        coord_dim: Dimension of coordinates (3)
        vector_dim: Dimension of vector field (3)
        encoder_hidden_dim: Hidden dimension for coordinate encoder
        encoder_num_frequencies: Number of Fourier frequencies
        transformation_hidden_dims: Hidden dimensions for transformation MLP
        state_dim: Dimension of state vector
        decoder_hidden_dims: Hidden dimensions for metric decoder
        use_residual: Use residual connections
    
    Returns:
        model: RiemannianCNF_Simple instance
    """
    model = RiemannianCNF_Simple(
        coord_dim=coord_dim,
        vector_dim=vector_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        encoder_num_frequencies=encoder_num_frequencies,
        transformation_hidden_dims=transformation_hidden_dims,
        state_dim=state_dim,
        decoder_hidden_dims=decoder_hidden_dims,
        use_residual=use_residual
    )
    return model

