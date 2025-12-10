"""
Train Braid Model (CPU Compatible)

Trains the Metric CNN on synthetic braid data (sin/cos crossing curves).
This version works on CPU/MPS (Apple Silicon).

Usage:
    python train_braid_cpu.py --epochs 100
"""

import os, sys
sys.path.append('../Packages')

import torch
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from skimage import filters
from tqdm import tqdm

from model import DenseED
from util import riemann, tensors

# ============================================================================
# DATASET
# ============================================================================

class BraidDataset(Dataset):
    def __init__(self, data_dir, device):
        self.data_dir = data_dir
        self.device = device
        
    def __len__(self):
        return 1
        
    def __getitem__(self, idx):
        # Load sin braid
        vector_field1_path = f'{self.data_dir}/sin/sin_vector_field.nhdr'
        mask1_path = f'{self.data_dir}/sin/sin_filt_mask.nhdr'
        vector_field1 = torch.from_numpy(
            sitk.GetArrayFromImage(sitk.ReadImage(vector_field1_path))
        ).permute(2,0,1).to(self.device).float() * 1000.0
        
        mask1 = torch.from_numpy(
            sitk.GetArrayFromImage(sitk.ReadImage(mask1_path))
        ).permute(1,0)
        boundary_mask1 = torch.where(torch.from_numpy(filters.laplace(mask1.numpy()))>0,1,0)
        mask1 = (mask1 - boundary_mask1).float().to(self.device)
        
        # Load cos braid  
        vector_field2_path = f'{self.data_dir}/cos/cos_vector_field.nhdr'
        mask2_path = f'{self.data_dir}/cos/cos_filt_mask.nhdr'
        vector_field2 = torch.from_numpy(
            sitk.GetArrayFromImage(sitk.ReadImage(vector_field2_path))
        ).permute(2,0,1).to(self.device).float() * 1000.0
        
        mask2 = torch.from_numpy(
            sitk.GetArrayFromImage(sitk.ReadImage(mask2_path))
        ).permute(1,0)
        boundary_mask2 = torch.where(torch.from_numpy(filters.laplace(mask2.numpy()))>0,1,0)
        mask2 = (mask2 - boundary_mask2).float().to(self.device)

        sample = {
            'vector_field': torch.cat((vector_field1, vector_field2), 0),
            'mask1': mask1.unsqueeze(0),
            'mask2': mask2.unsqueeze(0)
        }
        return sample

# ============================================================================
# MATRIX EXPONENTIAL
# ============================================================================

def matrix_exp_2d(A):
    """
    Construct positive definite matrix from symmetric matrix field A
    """
    I = torch.zeros_like(A)
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
    return nabla_vv

# ============================================================================
# TRAINING
# ============================================================================

def train_braid(input_dir, output_dir, epochs=100, lr=1e-2, resume=False):
    """
    Train the Metric CNN on braid data.
    """
    device = torch.device('cpu')
    torch.set_default_dtype(torch.float32)
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Model
    model = DenseED(
        in_channels=4,  # 2 channels per braid strand
        out_channels=3, 
        imsize=100,
        blocks=[6, 8, 6],
        growth_rate=16, 
        init_features=48,
        drop_rate=0,
        bottleneck=True,
        out_activation=None,
        upsample='nearest'
    )
    model.to(device)
    model.train()
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    
    dataset = BraidDataset(input_dir, device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    start_epoch = 0
    epoch_loss_list = []
    
    if resume and os.path.exists(f'{output_dir}/model.pth.tar'):
        checkpoint = torch.load(f'{output_dir}/model.pth.tar', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 50)
    
    for epoch in tqdm(range(start_epoch, start_epoch + epochs)):
        epoch_loss = 0
        
        for i, batch in enumerate(dataloader):
            input_vf = batch['vector_field'].to(device)
            input_vf.requires_grad = True
            
            mask1 = batch['mask1'].squeeze()
            mask2 = batch['mask2'].squeeze()
            
            # Forward pass
            u_pred = model(input_vf)
            
            # PDE loss for each strand
            pde_loss1 = pde(u_pred.squeeze(), input_vf[0,:2].squeeze(), mask1, differential_accuracy=2)
            pde_loss2 = pde(u_pred.squeeze(), input_vf[0,2:].squeeze(), mask2, differential_accuracy=2)
            
            # Mask the PDE residuals
            f_pred_1 = pde_loss1 * mask1.unsqueeze(0).expand(2,-1,-1)
            f_pred_2 = pde_loss2 * mask2.unsqueeze(0).expand(2,-1,-1)
            f_pred = torch.cat((f_pred_1, f_pred_2), 0)
            
            # Target is zero (geodesic equation satisfied)
            f_true = torch.zeros_like(f_pred)
            
            # Backward pass
            optimizer.zero_grad()
            loss = criterion(f_pred, f_true)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss_list.append(epoch_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss:.6f}")
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
            }, f'{output_dir}/model.pth.tar')
    
    # Final save
    torch.save({
        'epoch': start_epoch + epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss_list[-1]
    }, f'{output_dir}/model.pth.tar')
    
    print(f"\nTraining complete! Model saved to {output_dir}/model.pth.tar")
    
    # Save loss plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Braid Model Training Loss')
    plt.savefig(f'{output_dir}/training_loss.png', dpi=100)
    print(f"Loss plot saved to {output_dir}/training_loss.png")
    
    return model

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Braid Model')
    parser.add_argument('--input_dir', type=str, default='../Brains')
    parser.add_argument('--output_dir', type=str, default='../Checkpoints/braid')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--resume', action='store_true')
    
    args = parser.parse_args()
    
    # Check if data exists
    if not os.path.exists(f'{args.input_dir}/sin/sin_vector_field.nhdr'):
        print(f"ERROR: Braid data not found at {args.input_dir}/sin/")
        print("Make sure the synthetic braid data (sin/cos) is in the Brains folder")
        sys.exit(1)
    
    train_braid(
        args.input_dir,
        args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        resume=args.resume
    )

