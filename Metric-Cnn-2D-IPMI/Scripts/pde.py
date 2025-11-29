import torch, sys

sys.path.append('../Packages')
from util import tensors, riemann

def matrix_exp_2d(A):
    """
    Construct positive definite matrix from symmetric matrix field A
    Args:
        A, torch.Tensor
    Returns: 
        psd, torch.Tensor
    """
    I = torch.zeros_like(A, device='cuda')
    I[...,0,0] = 1
    I[...,1,1] = 1
    
    s = ((A[...,0,0]+A[...,1,1])/2.).unsqueeze(-1).unsqueeze(-1)
    q = torch.sqrt(-torch.det(A-torch.mul(s, I))).unsqueeze(-1).unsqueeze(-1)
    
    psd = torch.exp(s)*(torch.mul((torch.cosh(q)-s*torch.sinh(q)/q),I)+torch.sinh(q)/q*A)
    return psd

def pde(u, vector_lin, mask, differential_accuracy=2):
    s = tensors.lin2mat(u)
    metric_mat = matrix_exp_2d(s)
    nabla_vv = riemann.covariant_derivative_2d(vector_lin, metric_mat, mask, differential_accuracy=differential_accuracy)
    sigma = ((vector_lin[0]*nabla_vv[0]+vector_lin[1]*nabla_vv[1])/(vector_lin[0]*vector_lin[0]+vector_lin[1]*vector_lin[1]))
    
    return torch.stack((nabla_vv[0]-sigma*vector_lin[0],nabla_vv[1]-sigma*vector_lin[1]),0)
