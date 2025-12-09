import torch, sys
sys.path.append('../Packages')
from Packages.util import riemann
import numpy as np

# Import standard SPDManifoldLayer
try:
    from spdnet_layer import SPDManifoldLayer
    USE_STANDARD_SPD = True
except ImportError:
    USE_STANDARD_SPD = False
    print("[WARNING] spdnet_layer not found, using manual implementation")

def eigen_composite(u):
    theta = u[3].unsqueeze(-1).unsqueeze(-1)
    
    K = torch.zeros((*u.shape[1:], 3, 3))
    kx = u[0]
    ky = u[1]
    kz = u[2]
    K[...,0,1] = -kx
    K[...,0,2] = ky
    K[...,1,0] = kx
    K[...,1,2] = -kz
    K[...,2,0] = -ky
    K[...,2,1] = kz
    
    I = torch.zeros((*u.shape[1:], 3, 3))
    I[...,0,0] = 1
    I[...,1,1] = 1
    I[...,2,2] = 1
    
    Lamda = torch.zeros((*u.shape[1:], 3, 3))
    Lamda[...,0,0] = torch.exp(u[4])
    Lamda[...,1,1] = torch.exp(u[5])
    Lamda[...,2,2] = torch.exp(u[6])
    
    R = I + torch.mul(torch.sin(theta), K) + \
                torch.mul((1-torch.cos(theta)), torch.einsum('...ij,...jk->...ik', K, K))
    psd = torch.einsum('...ij,...jk,...lk->...il', R, Lamda, R)
    
    return psd
    
def pde(u, vector_lin, mask, differential_accuracy=2):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      u:                Output of the network - six distinct entries of a symmetric matrix.     (1, (n+1)n/2, h, w, d)
      vector_lin:     Input of the network                                                    (1, n, h, w, d)

    Returns:
      Output tensor nabla vv with shape `(n, h, w, d)`.
    """
    # [h, w, d, 3, 3]
    metric_mat = eigen_composite(u)
    tensor_mat = torch.inverse(metric_mat)
    
    nabla_vv = riemann.covariant_derivative_3d(vector_lin, metric_mat, mask, differential_accuracy=differential_accuracy)
    denominator = vector_lin[0]*vector_lin[0]+vector_lin[1]*vector_lin[1]+vector_lin[2]*vector_lin[2]
    denominator += 1-mask
    sigma = (vector_lin[0]*nabla_vv[0]+vector_lin[1]*nabla_vv[1]+vector_lin[2]*nabla_vv[2])/(denominator)*mask
    
    return torch.stack((nabla_vv[0]-sigma*vector_lin[0],nabla_vv[1]-sigma*vector_lin[1],nabla_vv[2]-sigma*vector_lin[2]),0)


# Import standard SPDManifoldLayer
try:
    from spdnet_layer import SPDManifoldLayer
    USE_STANDARD_SPD = True
except ImportError:
    USE_STANDARD_SPD = False
    print("[WARNING] spdnet_layer not found, using manual implementation")


def spd_manifold_layer_manual(u, epsilon=1e-4, max_value=100.0):
    """
    SPDManifoldNet Layer: Projects network output onto SPD manifold.
    
    This function replaces eigen_composite() by using a manifold-based approach.
    Instead of spectral decomposition (rotation + eigenvalues), it:
    1. Constructs a symmetric matrix from 6 parameters (upper triangular)
    2. Projects it onto SPD manifold using matrix exponential or eigendecomposition
    
    Args:
        u: (6, H, W, D) tensor - 6 parameters for symmetric 3x3 matrix
           Order: [a, b, c, d, e, f] where matrix is:
                  [[a, b, c],
                   [b, d, e],
                   [c, e, f]]
        epsilon: Small value to ensure positive definiteness (default: 1e-4, larger for stability)
        max_value: Maximum value for clipping to prevent overflow (default: 100.0)
    
    Returns:
        metric_spd: (H, W, D, 3, 3) tensor of SPD matrices
    """
    # Extract shape
    H, W, D = u.shape[1:]
    
    # Clip input values to prevent overflow in matrix operations
    # This is crucial to avoid infinite values in eigendecomposition
    u_clipped = torch.clamp(u, min=-max_value, max=max_value)
    
    # Construct symmetric matrix from 6 parameters
    # u[0] = a, u[1] = b, u[2] = c, u[3] = d, u[4] = e, u[5] = f
    metric = torch.zeros(H, W, D, 3, 3, device=u.device, dtype=u.dtype)
    
    # Upper triangular part
    metric[..., 0, 0] = u_clipped[0]  # a
    metric[..., 0, 1] = u_clipped[1]  # b
    metric[..., 0, 2] = u_clipped[2]  # c
    metric[..., 1, 1] = u_clipped[3]  # d
    metric[..., 1, 2] = u_clipped[4]  # e
    metric[..., 2, 2] = u_clipped[5]  # f
    
    # Make symmetric (copy upper to lower)
    metric[..., 1, 0] = u_clipped[1]  # b
    metric[..., 2, 0] = u_clipped[2]  # c
    metric[..., 2, 1] = u_clipped[4]  # e
    
    # Project onto SPD manifold using eigendecomposition
    # This ensures the output is SPD by:
    # 1. Computing eigendecomposition: metric = U @ diag(λ) @ U^T
    # 2. Clipping negative eigenvalues: λ' = max(ε, λ)
    # 3. Clipping large eigenvalues to prevent overflow
    # 4. Reconstructing: metric_spd = U @ diag(λ') @ U^T
    
    # Eigendecomposition (eigh for symmetric matrices, faster and more stable)
    try:
        eigenvals, eigenvecs = torch.linalg.eigh(metric)
    except RuntimeError as e:
        # If eigendecomposition fails, return identity matrix scaled by epsilon
        print(f"[WARNING] Eigendecomposition failed in spd_manifold_layer: {e}")
        metric_spd = torch.eye(3, device=u.device, dtype=u.dtype).expand(H, W, D, 3, 3).clone()
        metric_spd = metric_spd * epsilon
        return metric_spd
    
    # Ensure all eigenvalues are positive and bounded
    # Clip to [epsilon, max_value] to prevent overflow and underflow
    eigenvals_clipped = torch.clamp(eigenvals, min=epsilon, max=max_value)
    
    # Reconstruct SPD matrix
    # metric_spd = eigenvecs @ diag(eigenvals_clipped) @ eigenvecs^T
    # Using einsum for batched matrix multiplication
    metric_spd = torch.einsum('...ij,...j,...kj->...ik', 
                              eigenvecs, 
                              eigenvals_clipped, 
                              eigenvecs)
    
    # Final clipping to ensure numerical stability
    metric_spd = torch.clamp(metric_spd, min=-max_value*2, max=max_value*2)
    
    # Ensure symmetry (numerical errors might break it slightly)
    metric_spd = (metric_spd + metric_spd.transpose(-2, -1)) / 2.0
    
    return metric_spd


def pde_spdnet(u, vector_lin, mask, differential_accuracy=2, epsilon=1e-4):
    """
    PDE loss computation using SPDManifoldNet layer instead of eigen_composite.
    
    This function is identical to pde() except it uses spd_manifold_layer() 
    instead of eigen_composite() to construct the metric tensor.
    
    Args:
        u: (6, H, W, D) tensor - Network output (6 parameters for symmetric matrix)
        vector_lin: (3, H, W, D) tensor - Vector field input
        mask: (H, W, D) tensor - Brain mask
        differential_accuracy: Accuracy for numerical differentiation
        epsilon: Small value for SPD projection (default: 1e-4, larger for stability)
    
    Returns:
        Output tensor nabla vv with shape (3, H, W, D)
    """
    # [h, w, d, 3, 3] - Construct SPD metric using standard SPDManifoldLayer
    if USE_STANDARD_SPD:
        # Use standard implementation
        spd_layer = SPDManifoldLayer(epsilon=epsilon)
        metric_mat = spd_layer(u)
    else:
        # Fallback to manual implementation
        metric_mat = spd_manifold_layer_manual(u, epsilon=epsilon)
    
    # Compute inverse with numerical stability check
    # Use pinv (pseudo-inverse) if matrix is singular, or add small regularization
    try:
        tensor_mat = torch.inverse(metric_mat)
        # Check for infinite values in inverse
        if torch.any(torch.isinf(tensor_mat)) or torch.any(torch.isnan(tensor_mat)):
            # Fallback to pseudo-inverse with regularization
            identity = torch.eye(3, device=metric_mat.device, dtype=metric_mat.dtype)
            identity = identity.expand_as(metric_mat)
            tensor_mat = torch.inverse(metric_mat + epsilon * identity)
    except RuntimeError:
        # If inverse fails, use pseudo-inverse with regularization
        identity = torch.eye(3, device=metric_mat.device, dtype=metric_mat.dtype)
        identity = identity.expand_as(metric_mat)
        tensor_mat = torch.inverse(metric_mat + epsilon * identity)
    
    # Compute covariant derivative (same as original)
    nabla_vv = riemann.covariant_derivative_3d(vector_lin, metric_mat, mask, 
                                                differential_accuracy=differential_accuracy)
    
    # Project out component parallel to vector field (same as original)
    denominator = vector_lin[0]*vector_lin[0] + vector_lin[1]*vector_lin[1] + vector_lin[2]*vector_lin[2]
    denominator += 1 - mask
    sigma = (vector_lin[0]*nabla_vv[0] + vector_lin[1]*nabla_vv[1] + vector_lin[2]*nabla_vv[2]) / (denominator) * mask
    
    return torch.stack((nabla_vv[0]-sigma*vector_lin[0],
                        nabla_vv[1]-sigma*vector_lin[1],
                        nabla_vv[2]-sigma*vector_lin[2]), 0)


# Export spd_manifold_layer function for use in other modules
def spd_manifold_layer(u, epsilon=1e-4):
    """
    Wrapper function for spd_manifold_layer that uses the standard SPDManifoldLayer.
    
    Args:
        u: (6, H, W, D) tensor - 6 parameters for symmetric 3x3 matrix
        epsilon: Minimum eigenvalue value
    
    Returns:
        metric_spd: (H, W, D, 3, 3) tensor of SPD matrices
    """
    if USE_STANDARD_SPD:
        spd_layer = SPDManifoldLayer(epsilon=epsilon)
        return spd_layer(u)
    else:
        return spd_manifold_layer_manual(u, epsilon=epsilon)
