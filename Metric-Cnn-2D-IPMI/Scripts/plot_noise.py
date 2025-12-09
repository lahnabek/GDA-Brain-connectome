"""
Inference script to test trained Metric CNN with noisy data.

Usage:
    python plot_noise.py --noise_level 0.1
    python plot_noise.py --noise_level 0.0  # Clean (no noise)
    python plot_noise.py --noise_level 0.05 0.1 0.2  # Multiple noise levels
"""

import os, argparse, sys
sys.path.append('../Packages')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

import Packages.data.convert as convert
import Packages.util.tensors as tensors
from model import DenseED
from pde import matrix_exp_2d


def inference_with_noise(brain_id, input_dir, checkpoint_path, noise_level=0.1, save_output=True):
    """
    Load trained model and run inference on noisy data.
    
    Args:
        brain_id: Subject ID (e.g., '100610')
        input_dir: Path to brain data (e.g., '../Brains')
        checkpoint_path: Path to saved model (e.g., '../Checkpoints/100610/model.pth.tar')
        noise_level: Standard deviation of Gaussian noise (0.0 = no noise, 0.1 = 10% noise)
        save_output: Whether to save the output metric to file
    
    Returns:
        metric_pred_lin: The predicted metric tensor
    """
    device = torch.device("cpu")
    
    # Load the trained model
    print(f"\n{'='*60}")
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = DenseED(in_channels=2, out_channels=3, 
                    imsize=100,
                    blocks=[6, 8, 6],
                    growth_rate=16, 
                    init_features=48,
                    drop_rate=0,
                    bottleneck=True, 
                    out_activation=None,
                    upsample='nearest')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode (no training)
    model.to(device)
    
    # Load clean input data
    vector_path = f'{input_dir}/{brain_id}/{brain_id}_vector_field.nhdr'
    vector_lin = convert.read_nhdr(vector_path).to(device).permute(2,0,1).float()
    
    print(f"Original vector field shape: {vector_lin.shape}")
    print(f"Original vector field stats: min={vector_lin.min():.4f}, max={vector_lin.max():.4f}, std={vector_lin.std():.4f}")
    
    # Add Gaussian noise
    if noise_level > 0:
        noise = torch.randn_like(vector_lin) * noise_level * vector_lin.std()
        vector_lin_noisy = vector_lin + noise
        print(f"Added Gaussian noise with level {noise_level} (noise std={noise.std():.4f})")
    else:
        vector_lin_noisy = vector_lin
        print("No noise added (clean inference)")
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():  # No gradients needed for inference
        u_pred = model(vector_lin_noisy.unsqueeze(0))
        u_pred = u_pred[...,1:146,1:175].squeeze()
        s_pred = tensors.lin2mat(u_pred)
        metric_pred_mat = matrix_exp_2d(s_pred)
        metric_pred_lin = tensors.mat2lin(metric_pred_mat)
    
    # Save the result to plots folder
    if save_output:
        plots_dir = '../plots'
        os.makedirs(plots_dir, exist_ok=True)
        output_name = f'{plots_dir}/{brain_id}_metric_noise_{noise_level:.2f}.nhdr'
        sitk.WriteImage(
            sitk.GetImageFromArray(np.transpose(metric_pred_lin.cpu().detach().numpy(), (2,1,0))), 
            output_name
        )
        print(f"Saved: {output_name}")
    
    print(f"{'='*60}\n")
    return metric_pred_lin, vector_lin_noisy


def compare_metrics(brain_id, input_dir, noise_levels):
    """
    Compare metrics at different noise levels.
    
    Args:
        brain_id: Subject ID
        input_dir: Path to brain data
        noise_levels: List of noise levels to compare
    """
    # Load clean metric as reference
    clean_path = f'{input_dir}/{brain_id}/{brain_id}_learned_metric_final.nhdr'
    if os.path.exists(clean_path):
        metric_clean = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(clean_path)))
        print(f"Loaded clean reference metric from {clean_path}")
    else:
        print(f"Warning: Clean metric not found at {clean_path}")
        metric_clean = None
    
    results = {}
    plots_dir = '../plots'
    for noise in noise_levels:
        noisy_path = f'{plots_dir}/{brain_id}_metric_noise_{noise:.2f}.nhdr'
        if os.path.exists(noisy_path):
            metric_noisy = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(noisy_path)))
            
            if metric_clean is not None:
                diff = (metric_clean - metric_noisy).abs()
                results[noise] = {
                    'mean_diff': diff.mean().item(),
                    'max_diff': diff.max().item(),
                    'std_diff': diff.std().item()
                }
                print(f"Noise {noise:.2f}: Mean diff={results[noise]['mean_diff']:.6f}, Max diff={results[noise]['max_diff']:.6f}")
    
    return results


def plot_comparison(brain_id, input_dir, noise_levels, checkpoint_path):
    """
    Create a visual comparison of metrics at different noise levels.
    """
    fig, axes = plt.subplots(2, len(noise_levels) + 1, figsize=(4*(len(noise_levels)+1), 8))
    
    # Load and plot clean
    vector_path = f'{input_dir}/{brain_id}/{brain_id}_vector_field.nhdr'
    vector_clean = convert.read_nhdr(vector_path).permute(2,0,1).float()
    
    # Plot clean vector field
    axes[0, 0].imshow(vector_clean[0].numpy(), cmap='viridis')
    axes[0, 0].set_title('Clean Input (V1)')
    axes[0, 0].axis('off')
    
    # Plot clean metric
    clean_metric_path = f'{input_dir}/{brain_id}/{brain_id}_learned_metric_final.nhdr'
    if os.path.exists(clean_metric_path):
        metric_clean = sitk.GetArrayFromImage(sitk.ReadImage(clean_metric_path))
        axes[1, 0].imshow(metric_clean[0], cmap='hot')
        axes[1, 0].set_title('Clean Metric')
        axes[1, 0].axis('off')
    
    # Plot noisy versions
    for i, noise in enumerate(noise_levels):
        # Run inference with noise
        metric, vector_noisy = inference_with_noise(
            brain_id, input_dir, checkpoint_path, noise, save_output=True
        )
        
        # Plot noisy input
        axes[0, i+1].imshow(vector_noisy[0].numpy(), cmap='viridis')
        axes[0, i+1].set_title(f'Noisy Input (σ={noise})')
        axes[0, i+1].axis('off')
        
        # Plot noisy metric
        axes[1, i+1].imshow(metric[0].detach().numpy(), cmap='hot')
        axes[1, i+1].set_title(f'Metric (noise={noise})')
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    
    # Save figure to plots folder
    plots_dir = '../plots'
    os.makedirs(plots_dir, exist_ok=True)
    fig_path = f'{plots_dir}/noise_comparison_{brain_id}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison figure to: {fig_path}")
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained Metric CNN with noisy data')
    parser.add_argument('--brain_id', type=str, default='100610', help='Subject ID')
    parser.add_argument('--input_dir', type=str, default='../Brains', help='Path to brain data')
    parser.add_argument('--checkpoint', type=str, default='../Checkpoints/100610/model.pth.tar', 
                        help='Path to saved model checkpoint')
    parser.add_argument('--noise_level', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.2],
                        help='Noise level(s) to test (can specify multiple)')
    parser.add_argument('--plot', action='store_true', help='Create comparison plot')
    parser.add_argument('--compare', action='store_true', help='Compare metrics numerically')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Metric CNN Noise Testing")
    print("="*60)
    print(f"Brain ID: {args.brain_id}")
    print(f"Input dir: {args.input_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Noise levels: {args.noise_level}")
    print("="*60)
    
    if args.plot:
        # Create visual comparison
        plot_comparison(args.brain_id, args.input_dir, args.noise_level, args.checkpoint)
    else:
        # Run inference for each noise level
        for noise in args.noise_level:
            inference_with_noise(args.brain_id, args.input_dir, args.checkpoint, noise)
    
    if args.compare:
        # Compare metrics numerically
        print("\n" + "="*60)
        print("Metric Comparison (vs clean)")
        print("="*60)
        compare_metrics(args.brain_id, args.input_dir, args.noise_level)

