"""
Physics-Informed Neural Network (PINN) for Metric Estimation
Equivalent architecture to CEDNN in terms of parameter count.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def count_parameters(model):
    """Count total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MetricPINN(nn.Module):
    """
    Physics-Informed Neural Network for metric estimation.
    
    Takes spatial coordinates (x, y, z) and vector field (vx, vy, vz) as input,
    outputs 7 components for metric tensor representation.
    
    Architecture designed to have similar parameter count to CEDNN with blocks=[40,30,40].
    Estimated CEDNN params: ~500K-1M parameters
    """
    def __init__(self, hidden_layers=[256, 512, 512, 256, 128], activation='tanh', 
                 use_positional_encoding=True, encoding_dim=64):
        """
        Args:
            hidden_layers (list): List of hidden layer widths
            activation (str): Activation function ('tanh', 'relu', 'swish', 'gelu')
            use_positional_encoding (bool): Whether to use positional encoding for coordinates
            encoding_dim (int): Dimension of positional encoding
        """
        super(MetricPINN, self).__init__()
        
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            # Ensure encoding_dim is a multiple of 6 (2*L*3 for L frequencies)
            if encoding_dim % 6 != 0:
                encoding_dim = (encoding_dim // 6) * 6
                print(f"[MetricPINN] Adjusted encoding_dim to {encoding_dim} (must be multiple of 6)")
            self.encoding_dim = encoding_dim
        else:
            self.encoding_dim = 0
        
        # Input: coordinates (3) + vector field (3) = 6 dimensions
        # If using positional encoding: encoding_dim (already includes all x,y,z encodings) + 3 (vector field)
        # encoding_dim = 2*L*3 where L is number of frequencies
        input_dim = (encoding_dim + 3) if use_positional_encoding else 6
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        # Output layer: 7 components (for metric tensor representation)
        layers.append(nn.Linear(prev_dim, 7))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Print parameter count
        n_params = count_parameters(self)
        print(f'[MetricPINN] Total parameters: {n_params:,}')
    
    def _get_activation(self, activation):
        """Get activation function module."""
        if activation == 'tanh':
            return nn.Tanh()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'swish':
            return nn.SiLU()  # Swish is SiLU in PyTorch
        elif activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def positional_encoding(self, coords):
        """
        Positional encoding for coordinates (sinusoidal encoding).
        Helps the network learn high-frequency features.
        
        Args:
            coords: (..., 3) tensor of coordinates [x, y, z] in [0, 1] range
        
        Returns:
            encoded: (..., encoding_dim) tensor where encoding_dim = 2*L*3
        """
        # Calculate L from encoding_dim: encoding_dim = 2*L*3, so L = encoding_dim / 6
        L = self.encoding_dim // 6
        
        encodings = []
        for i in range(L):
            for coord_idx in range(3):
                encodings.append(torch.sin(2 ** i * np.pi * coords[..., coord_idx]))
                encodings.append(torch.cos(2 ** i * np.pi * coords[..., coord_idx]))
        
        return torch.stack(encodings, dim=-1)  # (..., encoding_dim)
    
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
            # coords is (3, H, W, D), vector_field is (B, 3, H, W, D)
            coords = coords.unsqueeze(0).expand(vector_field.shape[0], -1, -1, -1, -1)
            squeeze_output = False
        elif coords.dim() == 4 and vector_field.dim() == 4:
            # Both are (3, H, W, D) or (C, H, W, D)
            coords = coords.unsqueeze(0)
            vector_field = vector_field.unsqueeze(0)
            squeeze_output = True
        elif coords.dim() == 5 and vector_field.dim() == 4:
            # coords is (B, 3, H, W, D), vector_field is (3, H, W, D)
            vector_field = vector_field.unsqueeze(0).expand(coords.shape[0], -1, -1, -1, -1)
            squeeze_output = False
        else:
            # Both have same dimensions
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
        
        # Coordinates should already be in [0, 1] range from create_coordinate_grid
        # But ensure they are normalized
        coords_normalized = coords_flat  # Already normalized in create_coordinate_grid
        
        # Apply positional encoding if enabled
        if self.use_positional_encoding:
            encoded_coords = self.positional_encoding(coords_normalized)
            # encoded_coords shape: (B*H*W*D, encoding_dim)
            input_features = torch.cat([encoded_coords, vector_flat], dim=-1)  # (B*H*W*D, encoding_dim + 3)
        else:
            input_features = torch.cat([coords_normalized, vector_flat], dim=-1)  # (B*H*W*D, 6)
        
        # Forward through network
        output_flat = self.network(input_features)  # (B*H*W*D, 7)
        
        # Reshape back to (B, 7, H, W, D)
        output = output_flat.reshape(B, H, W, D, 7).permute(0, 4, 1, 2, 3)  # (B, 7, H, W, D)
        
        if squeeze_output:
            output = output.squeeze(0)  # (7, H, W, D)
        
        return output


def create_pinn_model(hidden_layers=[256, 512, 512, 256, 128], 
                      activation='tanh',
                      use_positional_encoding=True,
                      encoding_dim=64):
    """
    Factory function to create a PINN model with specified architecture.
    
    This architecture is designed to have similar parameter count to CEDNN with blocks=[40,30,40].
    Estimated parameter count: ~500K-1M parameters.
    
    Args:
        hidden_layers (list): List of hidden layer widths. Default gives ~600K-800K params.
        activation (str): Activation function
        use_positional_encoding (bool): Use positional encoding for coordinates
        encoding_dim (int): Dimension of positional encoding (must be multiple of 6)
    
    Returns:
        model: MetricPINN instance
    """
    model = MetricPINN(
        hidden_layers=hidden_layers,
        activation=activation,
        use_positional_encoding=use_positional_encoding,
        encoding_dim=encoding_dim
    )
    return model

