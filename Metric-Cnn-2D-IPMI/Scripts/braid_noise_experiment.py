"""
Synthetic Braid Noise Robustness Experiment

This script tests the Metric CNN's robustness to noise on synthetic braid data.
The braids are generated using sin/cos curves, which form crossing patterns.

Usage:
    python braid_noise_experiment.py --noise_level 0.0 0.05 0.1 0.2 --plot
"""

import os, sys
sys.path.append('../Packages')

import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import filters

# Local imports
from model import DenseED
from util import tensors
from plot import show_2d_tensors

# ============================================================================
# MATRIX EXPONENTIAL FOR METRIC
# ============================================================================

def matrix_exp_2d(A):
    """
    Construct positive definite matrix from symmetric matrix field A
    """
    device = A.device
    I = torch.zeros_like(A)
    I[...,0,0] = 1
    I[...,1,1] = 1
    
    s = ((A[...,0,0]+A[...,1,1])/2.).unsqueeze(-1).unsqueeze(-1)
    q = torch.sqrt(-torch.det(A-torch.mul(s, I))).unsqueeze(-1).unsqueeze(-1)
    
    psd = torch.exp(s)*(torch.mul((torch.cosh(q)-s*torch.sinh(q)/q),I)+torch.sinh(q)/q*A)
    return psd

# ============================================================================
# LOAD BRAID DATA
# ============================================================================

def load_braid_data(input_dir, add_noise=0.0):
    """
    Load sin/cos braid vector fields and optionally add noise.
    
    Args:
        input_dir: Path to Brains folder
        add_noise: Noise level (0.0 = no noise)
    
    Returns:
        vector_field: Combined sin+cos vector field (4 channels)
        mask1, mask2: Masks for each braid strand
    """
    device = torch.device('cpu')
    
    # Load sin braid
    vector_field1 = torch.from_numpy(
        sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/sin/sin_vector_field.nhdr'))
    ).permute(2,0,1).float()
    
    mask1 = torch.from_numpy(
        sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/sin/sin_filt_mask.nhdr'))
    ).permute(1,0).float()
    boundary_mask1 = torch.where(torch.from_numpy(filters.laplace(mask1.numpy()))>0,1,0).float()
    mask1 = mask1 - boundary_mask1
    
    # Load cos braid
    vector_field2 = torch.from_numpy(
        sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/cos/cos_vector_field.nhdr'))
    ).permute(2,0,1).float()
    
    mask2 = torch.from_numpy(
        sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/cos/cos_filt_mask.nhdr'))
    ).permute(1,0).float()
    boundary_mask2 = torch.where(torch.from_numpy(filters.laplace(mask2.numpy()))>0,1,0).float()
    mask2 = mask2 - boundary_mask2
    
    # Scale vector fields (as in original notebook)
    vector_field1 = vector_field1 * 1000.0
    vector_field2 = vector_field2 * 1000.0
    
    # Add noise if requested
    if add_noise > 0:
        noise1 = torch.randn_like(vector_field1) * add_noise * vector_field1.std()
        noise2 = torch.randn_like(vector_field2) * add_noise * vector_field2.std()
        vector_field1 = vector_field1 + noise1
        vector_field2 = vector_field2 + noise2
    
    # Combine into 4-channel input
    vector_field = torch.cat((vector_field1, vector_field2), dim=0)
    
    return vector_field, mask1, mask2

# ============================================================================
# INFERENCE
# ============================================================================

def run_braid_inference(vector_field, mask1, mask2, checkpoint_path):
    """
    Run inference on braid data and return predicted metric.
    """
    device = torch.device('cpu')
    
    # Load model (note: braid model has 4 input channels)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = DenseED(in_channels=4, out_channels=3, 
                    imsize=100,
                    blocks=[6, 8, 6],
                    growth_rate=16, 
                    init_features=48,
                    drop_rate=0,
                    bottleneck=True,
                    out_activation=None,
                    upsample='nearest')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Run inference
    with torch.no_grad():
        u_pred = model(vector_field.unsqueeze(0))
        u_pred = u_pred.squeeze()
        s_pred = tensors.lin2mat(u_pred)
        metric_pred_mat = matrix_exp_2d(s_pred)
        metric_pred_lin = tensors.mat2lin(metric_pred_mat)
    
    # Create combined mask
    mask = torch.where(mask1 + mask2 > 0, 1.0, 0.0)
    
    return metric_pred_lin, metric_pred_mat, mask

# ============================================================================
# EXPERIMENT
# ============================================================================

def braid_noise_experiment(input_dir, checkpoint_path, noise_levels, plot=True):
    """
    Run noise robustness experiment on braid data.
    """
    print("="*60)
    print("BRAID NOISE ROBUSTNESS EXPERIMENT")
    print("="*60)
    
    # First, get clean reference
    print("\nComputing clean reference...")
    vf_clean, mask1, mask2 = load_braid_data(input_dir, add_noise=0.0)
    metric_clean, metric_mat_clean, mask = run_braid_inference(
        vf_clean, mask1, mask2, checkpoint_path
    )
    metric_clean_np = metric_clean.numpy()
    
    results = []
    metrics_list = [metric_clean_np]
    vf_list = [vf_clean]
    
    # Test each noise level
    for noise in noise_levels:
        print(f"\nTesting noise level σ={noise}...")
        
        vf_noisy, _, _ = load_braid_data(input_dir, add_noise=noise)
        metric_noisy, _, _ = run_braid_inference(vf_noisy, mask1, mask2, checkpoint_path)
        metric_noisy_np = metric_noisy.numpy()
        
        # Compute error
        mae = np.abs(metric_noisy_np - metric_clean_np).mean()
        rmse = np.sqrt(((metric_noisy_np - metric_clean_np) ** 2).mean())
        max_err = np.abs(metric_noisy_np - metric_clean_np).max()
        
        results.append({
            'noise': noise,
            'MAE': mae,
            'RMSE': rmse,
            'Max Error': max_err
        })
        
        metrics_list.append(metric_noisy_np)
        vf_list.append(vf_noisy)
    
    # Print results table
    print("\n" + "="*60)
    print("NUMERICAL RESULTS (vs Clean)")
    print("="*60)
    print(f"{'Noise σ':<10} {'MAE':<12} {'RMSE':<12} {'Max Error':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['noise']:<10.2f} {r['MAE']:<12.6f} {r['RMSE']:<12.6f} {r['Max Error']:<12.6f}")
    print("="*60)
    
    # Create visualization
    if plot:
        create_braid_plot(vf_list, metrics_list, mask, noise_levels, results)
    
    return results

def create_braid_plot(vf_list, metrics_list, mask, noise_levels, results):
    """
    Create visualization comparing braid metrics at different noise levels.
    """
    n_cols = len(noise_levels) + 1
    fig, axes = plt.subplots(3, n_cols, figsize=(4*n_cols, 10))
    
    labels = ['Clean'] + [f'σ={n}' for n in noise_levels]
    
    for i, (vf, metric, label) in enumerate(zip(vf_list, metrics_list, labels)):
        # Row 1: Sin vector field (first 2 channels)
        axes[0, i].imshow(vf[0].numpy(), cmap='viridis')
        axes[0, i].set_title(f'Sin VF ({label})')
        axes[0, i].axis('off')
        
        # Row 2: Cos vector field (channels 2-4)
        axes[1, i].imshow(vf[2].numpy(), cmap='viridis')
        axes[1, i].set_title(f'Cos VF ({label})')
        axes[1, i].axis('off')
        
        # Row 3: Metric g₁₁
        masked_metric = metric[0] * mask.numpy()
        axes[2, i].imshow(masked_metric, cmap='hot')
        
        if i == 0:
            axes[2, i].set_title(f'Metric g₁₁ ({label})\nReference')
        else:
            mae = results[i-1]['MAE']
            axes[2, i].set_title(f'Metric g₁₁ ({label})\nMAE={mae:.4f}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    # Save
    plots_dir = '../plots'
    os.makedirs(plots_dir, exist_ok=True)
    fig_path = f'{plots_dir}/braid_noise_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_path}")
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Braid Noise Robustness Experiment')
    parser.add_argument('--input_dir', type=str, default='../Brains')
    parser.add_argument('--checkpoint', type=str, default='../Checkpoints/braid/model.pth.tar')
    parser.add_argument('--noise_level', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument('--plot', action='store_true')
    
    args = parser.parse_args()
    
    # Check if braid checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Braid checkpoint not found at {args.checkpoint}")
        print("You need to first train the braid model using MetricCnnTrainingInferenceBraid.ipynb")
        print("Or specify the correct checkpoint path with --checkpoint")
        sys.exit(1)
    
    braid_noise_experiment(
        args.input_dir, 
        args.checkpoint, 
        args.noise_level,
        plot=args.plot
    )

