import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)   # important : remet le working directory au dossier Scripts

sys.path.insert(0, PROJECT_ROOT)

import torch, argparse, json, glob
import Packages.data.convert as convert
import Packages.util.tensors as tensors
import SimpleITK as sitk
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import DatasetHCP
from pde import *
from modelRCNF_simple import create_rcnf_simple_model
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd

# ----------- CONFIG -------------
brain_id = "111312"
input_dir = "../Brains"
output_dir = "../Checkpoints_RCNF_sans_ODE"  # Include RCNF-Simple in name
gpu_device = -1
epoch_num = 200  # 10000
learning_rate = 3e-3  # Reduced from 1e-2 for better stability
terminating_loss = 1e6
checkpoint_save_frequency = 50
RESUME = False
RESUME_FILE = None  # None pour le dernier model

# RCNF-Simple architecture hyperparameters (NO ODE)
RCNF_COORD_DIM = 3
RCNF_VECTOR_DIM = 3
RCNF_ENCODER_HIDDEN_DIM = 128  # Hidden dimension for coordinate encoder
RCNF_ENCODER_NUM_FREQUENCIES = 10  # Number of Fourier frequency bands
RCNF_TRANSFORMATION_HIDDEN_DIMS = [256, 512, 512, 256]  # Hidden dimensions for transformation MLP (replaces ODE)
RCNF_STATE_DIM = 64  # Dimension of state vector (input to decoder)
RCNF_DECODER_HIDDEN_DIMS = [256, 128]  # Hidden dimensions for metric decoder
RCNF_USE_RESIDUAL = True  # Use residual connections in transformation MLP
# --------------------------------


def create_coordinate_grid(shape, device='cpu'):
    """
    Create coordinate grid for RCNF input.
    
    Args:
        shape: (C, H, W, D) or (H, W, D) shape of the vector field
        device: torch device
    
    Returns:
        coords: (3, H, W, D) tensor of coordinates [x, y, z] normalized to [0, 1]
    """
    if len(shape) == 4:
        C, H, W, D = shape
    else:
        H, W, D = shape
    
    # Create coordinate grids (normalized to [0, 1])
    x = torch.linspace(0, 1, H, device=device)
    y = torch.linspace(0, 1, W, device=device)
    z = torch.linspace(0, 1, D, device=device)
    
    # Create meshgrid with indexing='ij' for (H, W, D) ordering
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Stack to (3, H, W, D) where first dim is [x, y, z]
    coords = torch.stack([X, Y, Z], dim=0)
    
    return coords


def plot_loss():
    """Plot and save loss curve."""
    figures_dir = f"{output_dir}/figures"
    os.makedirs(figures_dir, exist_ok=True)

    # Charger le CSV des pertes
    loss_df = pd.read_csv(f"{output_dir}/{brain_id}/loss.csv")

    # Style publication
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8,5), dpi=150)

    ax.plot(loss_df["epoch"], loss_df["loss"], linewidth=2, color="royalblue")

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title(
        f"Training Loss — brain {brain_id}\n"
        f"epochs={loss_df['epoch'].iloc[-1]} | lr={learning_rate} | RCNF-Simple (no ODE, residual: {RCNF_USE_RESIDUAL})",
        fontsize=14,
        pad=10
    )

    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(f"{figures_dir}/loss_curve.png", dpi=300)
    plt.close(fig)


def train(brain_id, input_dir, output_dir, gpu_device=0, epoch_num=10000, learning_rate=1e-4,
          terminating_loss=1e6, checkpoint_save_frequency=1000, resume=RESUME, resume_from_checkpoint=RESUME_FILE):

    device = torch.device("cpu")

    output_dir = f'{output_dir}/{brain_id}'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    metrics_dir = f"{output_dir}/metrics"
    os.makedirs(metrics_dir, exist_ok=True)   # dossier où on enregistrera toutes les métriques (intermédiaires)

    # --- save config ---
    config_file = os.path.join(output_dir, "training_config.json")
    # Charger le config existant si on reprend, sinon créer un nouveau
    config_data = {}
    if resume and os.path.isfile(config_file):
        with open(config_file, "r") as f:
            config_data = json.load(f)
    # Mettre à jour avec les valeurs actuelles
    config_data.update({
        "brain_id": brain_id,
        "epochs": epoch_num,
        "learning_rate": learning_rate,
        "checkpoint_frequency": checkpoint_save_frequency,
        "model_type": "RCNF-Simple",
        "coord_dim": RCNF_COORD_DIM,
        "vector_dim": RCNF_VECTOR_DIM,
        "encoder_hidden_dim": RCNF_ENCODER_HIDDEN_DIM,
        "encoder_num_frequencies": RCNF_ENCODER_NUM_FREQUENCIES,
        "transformation_hidden_dims": RCNF_TRANSFORMATION_HIDDEN_DIMS,
        "state_dim": RCNF_STATE_DIM,
        "decoder_hidden_dims": RCNF_DECODER_HIDDEN_DIMS,
        "use_residual": RCNF_USE_RESIDUAL,
        "note": "Simplified version without ODE - uses direct MLP transformation",
        "resume": resume,
    })
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)

    # Create RCNF-Simple model (without ODE)
    print("="*80)
    print("CRÉATION DU MODÈLE RCNF-SIMPLE (SANS ODE)")
    print("="*80)
    print("Améliorations:")
    print("  - Pas d'intégration ODE (plus rapide et stable)")
    print("  - MLP direct avec residual connections")
    print("  - Layer normalization pour stabilité")
    print("  - Meilleure initialisation (He/Kaiming)")
    print("="*80)
    
    model = create_rcnf_simple_model(
        coord_dim=RCNF_COORD_DIM,
        vector_dim=RCNF_VECTOR_DIM,
        encoder_hidden_dim=RCNF_ENCODER_HIDDEN_DIM,
        encoder_num_frequencies=RCNF_ENCODER_NUM_FREQUENCIES,
        transformation_hidden_dims=RCNF_TRANSFORMATION_HIDDEN_DIMS,
        state_dim=RCNF_STATE_DIM,
        decoder_hidden_dims=RCNF_DECODER_HIDDEN_DIMS,
        use_residual=RCNF_USE_RESIDUAL
    )

    model.train()
    model = model.to(device)
    
    criterion = torch.nn.MSELoss()
    # Use Adam optimizer (better for MLPs) with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # More aggressive scheduler for faster convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15)
    dataset_id = DatasetHCP(input_dir, sample_name_list=[str(brain_id)])
    dataloader_id = DataLoader(dataset_id, batch_size=1, shuffle=False, num_workers=0)

    # --- loss CSV ---
    loss_csv_path = os.path.join(output_dir, "loss.csv")
    if not (resume and os.path.isfile(loss_csv_path)):
        # (ré)initialise le fichier de pertes uniquement si on ne reprend pas
        with open(loss_csv_path, 'w') as f:
            f.write("epoch,loss\n")

    # --- load inputs once ---
    vec_path = f'{input_dir}/{brain_id}/{brain_id}_shrinktensor_principal_vector_field.nhdr'
    mask_path = f'{input_dir}/{brain_id}/{brain_id}_shrinktensor_filt_mask.nhdr'
    base_vector = convert.read_nhdr(vec_path).to(device).float()
    base_mask   = convert.read_nhdr(mask_path).to(device).float()
    
    # Create coordinate grid for base_vector shape
    base_coords = create_coordinate_grid(base_vector.shape, device=device)

    # --- reprise depuis un checkpoint spécifique ou model_final.pth.tar ---
    start_epoch = 0
    last_loss = None
    if resume:
        # Si un fichier checkpoint spécifique est fourni, l'utiliser
        if resume_from_checkpoint is not None:
            ckpt_path = resume_from_checkpoint
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"[resume] Fichier checkpoint spécifié non trouvé : {ckpt_path}")
        else:
            # Sinon, utiliser model_final.pth.tar par défaut
            ckpt_path = os.path.join(output_dir, "model_final.pth.tar")
        
        if os.path.isfile(ckpt_path):
            print(f"[resume] Chargement du checkpoint : {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Lire l'epoch depuis le checkpoint
            start_epoch = int(checkpoint.get('epoch', 0)) + 1
            last_loss = checkpoint.get('loss', None)
            # Vérifier aussi dans training_config.json si l'epoch y est sauvegardée (seulement si on utilise model_final)
            if resume_from_checkpoint is None and os.path.isfile(config_file):
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    if 'final_epoch' in config_data:
                        start_epoch = int(config_data['final_epoch']) + 1
                        print(f"[resume] Epoch lu depuis training_config.json : {start_epoch}")
            print(f"[resume] Reprise à l'epoch {start_epoch} (loss précédente={last_loss})")
        else:
            print(f"[resume] Checkpoint non trouvé : {ckpt_path}, entraînement à partir de zéro.")

    last_epoch = start_epoch - 1

    for epoch in tqdm(range(start_epoch, epoch_num)):
        epoch_loss_id = 0

        for i, batched_id_sample in enumerate(dataloader_id):
            input_vector = batched_id_sample['vector_field']
            input_vector = input_vector.to(device)
            input_vector.requires_grad = True
            
            # Create coordinate grid for this batch
            # input_vector shape is (B, 3, H, W, D) from DataLoader
            _, _, H, W, D = input_vector.shape
            coords = create_coordinate_grid((H, W, D), device=device)
            # coords is (3, H, W, D), model will handle batch dimension
            
            optimizer.zero_grad()

            # RCNF forward: takes coordinates and vector field
            # model handles batch dimension automatically
            u_pred_id = model(coords, input_vector)
            mask = batched_id_sample['mask'].squeeze().to(device)
            pde_loss = pde(u_pred_id.squeeze(), input_vector.squeeze(), mask)
            f_pred_id = torch.einsum('...ij,...ij->...ij', pde_loss, mask.expand(3,-1,-1,-1))
            f_true_id = torch.zeros_like(f_pred_id)

            loss_id = criterion(f_pred_id, f_true_id)
            loss_id.backward()
            
            # Gradient clipping for stability (reduced clipping for simpler model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            epoch_loss_id += loss_id.item()

        scheduler.step(epoch_loss_id)

        last_epoch = epoch

        # write loss
        with open(loss_csv_path, 'a') as f:
            f.write(f"{epoch},{epoch_loss_id}\n")

        # --- save intermediate checkpoints ---
        if epoch % checkpoint_save_frequency == 0:
            ckpt_path = f'{output_dir}/checkpoint_epoch_{epoch:04d}.pth.tar'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss_id,
                'config': {
                    "learning_rate": learning_rate,
                    "coord_dim": RCNF_COORD_DIM,
                    "vector_dim": RCNF_VECTOR_DIM,
                    "encoder_hidden_dim": RCNF_ENCODER_HIDDEN_DIM,
                    "encoder_num_frequencies": RCNF_ENCODER_NUM_FREQUENCIES,
                    "transformation_hidden_dims": RCNF_TRANSFORMATION_HIDDEN_DIMS,
                    "state_dim": RCNF_STATE_DIM,
                    "decoder_hidden_dims": RCNF_DECODER_HIDDEN_DIMS,
                    "use_residual": RCNF_USE_RESIDUAL,
                    "epoch": epoch,
                    "model_type": "RCNF-Simple"
                }
            }, ckpt_path)

            # --- save the metric tensor at this epoch ---
            with torch.no_grad():
                u_pred = model(base_coords, base_vector)
                metric_mat = eigen_composite(u_pred)
                metric_lin = tensors.mat2lin(metric_mat)
                np_metric = np.transpose(metric_lin.detach().cpu().numpy(), (3,2,1,0))

                sitk.WriteImage(sitk.GetImageFromArray(np_metric),
                                f"{metrics_dir}/metric_epoch_{epoch:04d}.nhdr")

        # stop
        if epoch_loss_id < terminating_loss:
            break
    
    # Si aucune epoch n'a été faite (start_epoch >= epoch_num), on essaie de récupérer la dernière loss connue
    if start_epoch >= epoch_num:
        epoch_loss_id = last_loss if last_loss is not None else 0.0

    # --- final model ---
    final_ckpt = f'{output_dir}/model_final.pth.tar'
    torch.save({
        'epoch': last_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss_id,
        'config': {
            "learning_rate": learning_rate,
            "coord_dim": RCNF_COORD_DIM,
            "vector_dim": RCNF_VECTOR_DIM,
            "encoder_hidden_dim": RCNF_ENCODER_HIDDEN_DIM,
            "encoder_num_frequencies": RCNF_ENCODER_NUM_FREQUENCIES,
            "transformation_hidden_dims": RCNF_TRANSFORMATION_HIDDEN_DIMS,
            "state_dim": RCNF_STATE_DIM,
            "decoder_hidden_dims": RCNF_DECODER_HIDDEN_DIMS,
            "use_residual": RCNF_USE_RESIDUAL,
            "epoch": last_epoch,
            "model_type": "RCNF-Simple"
        }
    }, final_ckpt)

    # --- mettre à jour training_config.json avec l'epoch finale ---
    if os.path.isfile(config_file):
        with open(config_file, "r") as f:
            config_data = json.load(f)
        config_data['final_epoch'] = last_epoch
        config_data['final_loss'] = epoch_loss_id
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

    # save final metric
    with torch.no_grad():
        u_pred = model(base_coords, base_vector)
        metric_mat = eigen_composite(u_pred)
        metric_lin = tensors.mat2lin(metric_mat)
        np_metric = np.transpose(metric_lin.detach().cpu().numpy(), (3,2,1,0))

        sitk.WriteImage(sitk.GetImageFromArray(np_metric),
                        f"{metrics_dir}/metric_final.nhdr")
    plot_loss()


# run
if __name__ == '__main__':

    train(brain_id, input_dir, output_dir, gpu_device, epoch_num, learning_rate,
          terminating_loss, checkpoint_save_frequency, resume=RESUME)

