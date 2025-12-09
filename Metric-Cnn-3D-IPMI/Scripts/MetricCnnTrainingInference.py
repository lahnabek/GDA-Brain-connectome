import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)   # important : remet le working directory au dossier Scripts

sys.path.insert(0, PROJECT_ROOT)

import torch, argparse, json, glob
import torch.nn.functional as F
import Packages.data.convert as convert
import Packages.util.tensors as tensors
import SimpleITK as sitk
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import DatasetHCP
from pde import *
from model3D import *
from modelCEDNNupgraded import DenseED
import matplotlib.pyplot as plt
import pandas as pd
# ----------- CONFIG -------------
brain_id = "111312"
input_dir = "../Brains"
output_dir = "../Checkpoints_CEDNN_10_10_10"
gpu_device = -1
epoch_num = 2000 
learning_rate = 1e-2
terminating_loss = 1e6
checkpoint_save_frequency = 50
BOTTLENECK = True
RESUME = True
RESUME_FILE = None#"../Checkpoints_CEDNN_10_10_10_Upgraded_AF3/111312/checkpoint_epoch_1300.pth.tar" # 
BLOCKS = [10,10,10] # [40,30,40]xcc
# --------------------------------

def apply_anisotropic_filter(metric_mat, kernel_size=3, epsilon=1e-6):
    """
    Applique un filtre anisotrope guidé par la métrique riemannienne.
    
    Pour chaque voxel x, définit un noyau gaussien riemannien:
    K_x(u) = exp(-0.5 * u^T * g(x)^(-1) * u)
    
    où u parcourt les décalages d'un voisinage discret (kernel_size x kernel_size x kernel_size).
    Le filtre est appliqué voxel par voxel en pondérant chaque voisinage par K_x.
    
    Args:
        metric_mat: (H, W, D, 3, 3) tensor - Tenseur métrique SPD
        kernel_size: int - Taille du noyau (default: 3 pour un voisinage 3x3x3)
        epsilon: float - Petite valeur pour la stabilité numérique lors de l'inversion
    
    Returns:
        filtered_metric: (H, W, D, 3, 3) tensor - Tenseur métrique filtré
    """
    device = metric_mat.device
    dtype = metric_mat.dtype
    H, W, D = metric_mat.shape[:3]
    
    # Calculer l'inverse de la métrique pour chaque voxel: g(x)^(-1)
    # Ajouter une petite régularisation pour la stabilité numérique
    identity = torch.eye(3, device=device, dtype=dtype).expand(H, W, D, 3, 3)
    metric_reg = metric_mat + epsilon * identity
    
    try:
        metric_inv = torch.linalg.inv(metric_reg)  # (H, W, D, 3, 3)
    except RuntimeError:
        # Fallback si l'inversion échoue
        print("[WARNING] Matrix inversion failed in anisotropic filter, using identity")
        metric_inv = identity / epsilon
    
    # Créer les décalages pour le voisinage (kernel_size x kernel_size x kernel_size)
    # Par exemple, pour kernel_size=3: [-1, 0, 1] dans chaque direction
    half_kernel = kernel_size // 2
    offsets_list = []
    for dx in range(-half_kernel, half_kernel + 1):
        for dy in range(-half_kernel, half_kernel + 1):
            for dz in range(-half_kernel, half_kernel + 1):
                offsets_list.append([dx, dy, dz])
    offsets = torch.tensor(offsets_list, device=device, dtype=dtype)  # (kernel_size^3, 3)
    num_offsets = offsets.shape[0]
    
    # VERSION COMPLÈTEMENT VECTORISÉE
    # Calculer tous les poids pour tous les voxels en une fois
  
    # Calculer quad_forms pour tous les voxels et offsets de manière vectorisée
    # quad_forms[i,j,k,o] = offsets[o]^T @ metric_inv[i,j,k] @ offsets[o]
    # metric_inv: (H, W, D, 3, 3) = (h, w, d, m, n)
    # offsets: (num_offsets, 3) = (o, m) et (o, n)
    # Résultat: (H, W, D, num_offsets) = (h, w, d, o)
    quad_forms_all = torch.einsum('om,hwdmn,on->hwdo', offsets, metric_inv, offsets)
    # quad_forms_all: (H, W, D, num_offsets)
    
    weights_all = torch.exp(-0.5 * quad_forms_all)  # (H, W, D, num_offsets)
    
    
    
    # Appliquer la convolution pondérée de manière complètement vectorisée
    # Pour chaque offset, on va décaler metric_mat et multiplier par les poids
    
    # Padding simple pour gérer les bords (mode 'replicate' pour les bords)
    # On pad de half_kernel dans chaque direction
    pad_size = half_kernel
    metric_padded = F.pad(metric_mat, (0, 0, 0, 0, pad_size, pad_size, pad_size, pad_size, pad_size, pad_size), mode='constant')
    # metric_padded: (H+2*pad, W+2*pad, D+2*pad, 3, 3)
    
    # Initialiser les sommes pondérées
    weighted_sums = torch.zeros_like(metric_mat)  # (H, W, D, 3, 3)
    weight_sums = torch.zeros(H, W, D, device=device, dtype=dtype)  # (H, W, D)
    
    # Pour chaque offset, calculer la contribution de manière vectorisée
    for idx, offset in enumerate(offsets):
        dx, dy, dz = int(offset[0].item()), int(offset[1].item()), int(offset[2].item())
        
        # Calculer les indices dans le tenseur paddé
        # Les voxels (i,j,k) dans l'original correspondent à (i+pad, j+pad, k+pad) dans le paddé
        # Le voisin (i+dx, j+dy, k+dz) correspond à (i+pad+dx, j+pad+dy, k+pad+dz)
        i_start = pad_size + max(0, -dx)
        i_end = pad_size + H + min(0, -dx)
        j_start = pad_size + max(0, -dy)
        j_end = pad_size + W + min(0, -dy)
        k_start = pad_size + max(0, -dz)
        k_end = pad_size + D + min(0, -dz)
        
        # Extraire la région valide du tenseur décalé
        neighbor_region = metric_padded[
            pad_size + dx : pad_size + H + dx,
            pad_size + dy : pad_size + W + dy,
            pad_size + dz : pad_size + D + dz
        ]  # (H, W, D, 3, 3)
        
        # Extraire les poids pour cet offset
        weights_offset = weights_all[:, :, :, idx]  # (H, W, D)
        
        # Ajouter la contribution pondérée
        # weighted_sums += weights_offset[..., None, None] * neighbor_region
        weighted_sums += weights_offset.unsqueeze(-1).unsqueeze(-1) * neighbor_region
        weight_sums += weights_offset
    
    # Normaliser
    
    valid_mask = weight_sums > 1e-10  # (H, W, D)
    filtered_metric = torch.zeros_like(metric_mat)
    filtered_metric[valid_mask] = (weighted_sums[valid_mask] / 
                                  weight_sums[valid_mask].unsqueeze(-1).unsqueeze(-1))
    filtered_metric[~valid_mask] = metric_mat[~valid_mask]
    
    return filtered_metric


def plot_loss():
        # ---------------------------------------------------------------------
    #  ENREGISTREMENT DE LA COURBE DE LOSS (FIGURE PUBLIABLE)
    # ---------------------------------------------------------------------
    

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
        f"epochs={loss_df['epoch'].iloc[-1]} | lr={learning_rate} | blocks={BLOCKS}",
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
        "blocks": BLOCKS,
        "resume": resume,
    })
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)

    blocks = BLOCKS
    model = DenseED(in_channels=3, out_channels=7, 
                    imsize=100,
                    blocks=blocks,
                    growth_rate=16, 
                    init_features=48,
                    drop_rate=0,
                    out_activation=None,
                    bottleneck=BOTTLENECK,
                    upsample='nearest')

    model.train()
    model = model.to(device)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
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
            input_id = batched_id_sample['vector_field']
            input_id = input_id.to(device)
            input_id.requires_grad = True
            optimizer.zero_grad()

            u_pred_id = model(input_id)
            mask = batched_id_sample['mask'].squeeze().to(device)
            pde_loss = pde(u_pred_id.squeeze(), input_id.squeeze(), mask)
            f_pred_id = torch.einsum('...ij,...ij->...ij', pde_loss, mask.expand(3,-1,-1,-1))
            f_true_id = torch.zeros_like(f_pred_id)

            loss_id = criterion(f_pred_id, f_true_id)
            loss_id.backward()
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
                    "blocks": BLOCKS,
                    "epoch": epoch
                }
            }, ckpt_path)

            # --- save the metric tensor at this epoch ---
            with torch.no_grad():
                u_pred = model(base_vector.unsqueeze(0)).squeeze()
                metric_mat = eigen_composite(u_pred)
                metric_lin = tensors.mat2lin(metric_mat)
                np_metric = metric_lin.cpu().numpy()
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
            "blocks": BLOCKS,
            "epoch": last_epoch
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
    u_pred = model(base_vector.unsqueeze(0)).squeeze()
    metric_mat = eigen_composite(u_pred)
    metric_lin = tensors.mat2lin(metric_mat)
    np_metric = np.transpose(metric_lin.detach().cpu().numpy(), (3,2,1,0))

    sitk.WriteImage(sitk.GetImageFromArray(np_metric),
                    f"{metrics_dir}/metric_final.nhdr") # tensor_lin pour geodesic ?
    plot_loss()



def train_upgraded(brain_id, input_dir, output_dir, gpu_device=0, epoch_num=10000, learning_rate=1e-4,
                   terminating_loss=1e6, checkpoint_save_frequency=1000, resume=False, resume_from_checkpoint=None,
                   blocks=[10, 10, 10], growth_rate=16, init_features=48, drop_rate=0, bn_size=8,
                   bottleneck=False, upsample='nearest',
                   use_positional_encoding=False, use_fourier_features=False, num_fourier_frequencies=10,
                   fourier_learnable=False, fourier_scale=1.0,
                   pe_learnable_scale=False, pe_learnable_offset=False,
                   use_anisotropic_filter=False, filter_kernel_size=3):
    """
    Training function using DenseED model (same structure as model3D) with optional:
    - Positional encoding (normalized coordinates as additional channels)
    - Fourier features (NeRF-style encoding with multiple frequencies)
    - Anisotropic filter (Riemannian Gaussian kernel guided by the metric)
    
    Parameters:
    -----------
    brain_id : str
        Subject ID
    input_dir : str
        Input directory path
    output_dir : str
        Output directory path (will append "_Upgraded" suffix)
    gpu_device : int
        GPU device ID (-1 for CPU)
    epoch_num : int
        Number of training epochs
    learning_rate : float
        Learning rate
    terminating_loss : float
        Early stopping loss threshold
    checkpoint_save_frequency : int
        Frequency of checkpoint saving
    resume : bool
        Whether to resume training
    resume_from_checkpoint : str, optional
        Path to specific checkpoint file
    blocks : list
        List of block sizes (must be odd length)
    growth_rate : int
        Growth rate K for dense blocks
    init_features : int
        Initial feature maps
    drop_rate : float
        Dropout rate
    bn_size : int
        Bottleneck size multiplier
    bottleneck : bool
        Enable bottleneck in dense blocks (default: False, same as model3D)
    upsample : str
        Upsampling method ('nearest' or 'trilinear')
    use_positional_encoding : bool
        Add normalized coordinates as 3 additional channels (default: False)
    use_fourier_features : bool
        Add Fourier-encoded coordinates (default: False)
    num_fourier_frequencies : int
        Number of frequency levels for Fourier features (default: 10)
    fourier_learnable : bool
        If True, Fourier frequencies are learnable parameters (default: False)
    fourier_scale : float
        Scaling factor for Fourier frequencies (default: 1.0)
    pe_learnable_scale : bool
        If True, positional encoding has learnable scaling per dimension (3 parameters, default: False)
    pe_learnable_offset : bool
        If True, positional encoding has learnable offset per dimension (3 parameters, default: False)
    use_anisotropic_filter : bool
        Apply anisotropic Riemannian Gaussian filter before PDE loss (default: False)
    filter_kernel_size : int
        Size of the filter kernel (default: 3 for 3x3x3 neighborhood)
    """
    device = torch.device("cpu")
    
    # Output directory with appropriate suffix
    suffix_parts = []
    if use_positional_encoding:
        pe_suffix = "PE"
        if pe_learnable_scale or pe_learnable_offset:
            pe_params = []
            if pe_learnable_scale:
                pe_params.append("S")  # S for Scale
            if pe_learnable_offset:
                pe_params.append("O")  # O for Offset
            pe_suffix += "".join(pe_params)
        suffix_parts.append(pe_suffix)
    if use_fourier_features:
        ff_suffix = f"FF{num_fourier_frequencies}"
        if fourier_learnable:
            ff_suffix += "L"  # L for Learnable
        suffix_parts.append(ff_suffix)
    if use_anisotropic_filter:
        suffix_parts.append(f"AF{filter_kernel_size}")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    output_dir_upgraded = f'{output_dir}_Upgraded{suffix}/{brain_id}'
    if not os.path.isdir(output_dir_upgraded):
        os.makedirs(output_dir_upgraded)

    metrics_dir = f"{output_dir_upgraded}/metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    # --- save config ---
    config_file = os.path.join(output_dir_upgraded, "training_config.json")
    config_data = {}
    if resume and os.path.isfile(config_file):
        with open(config_file, "r") as f:
            config_data = json.load(f)
    
    config_data.update({
        "brain_id": brain_id,
        "epochs": epoch_num,
        "learning_rate": learning_rate,
        "checkpoint_frequency": checkpoint_save_frequency,
        "blocks": blocks,
        "growth_rate": growth_rate,
        "init_features": init_features,
        "drop_rate": drop_rate,
        "bn_size": bn_size,
        "bottleneck": bottleneck,
        "upsample": upsample,
        "use_positional_encoding": use_positional_encoding,
        "pe_learnable_scale": pe_learnable_scale,
        "pe_learnable_offset": pe_learnable_offset,
        "use_fourier_features": use_fourier_features,
        "num_fourier_frequencies": num_fourier_frequencies,
        "fourier_learnable": fourier_learnable,
        "fourier_scale": fourier_scale,
        "use_anisotropic_filter": use_anisotropic_filter,
        "filter_kernel_size": filter_kernel_size,
        "optimizer": "Adadelta",
        "scheduler": "ReduceLROnPlateau",
        "model_type": "DenseED" ,
        "resume": resume,
    })
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)

    # Create model (same structure as model3D)
    print("="*80)
    print("CRÉATION DU MODÈLE DenseED (structure identique à model3D)")
    print("="*80)
    print(f"  - Blocks: {blocks}")
    print(f"  - Growth rate: {growth_rate}")
    print(f"  - Init features: {init_features}")
    print(f"  - Bottleneck: {bottleneck}")
    print(f"  - Positional Encoding: {use_positional_encoding}", end="")
    if use_positional_encoding:
        pe_params = []
        if pe_learnable_scale:
            pe_params.append("learnable scale")
        if pe_learnable_offset:
            pe_params.append("learnable offset")
        if pe_params:
            print(f" ({', '.join(pe_params)})")
        else:
            print()
    else:
        print()
    print(f"  - Fourier Features: {use_fourier_features}", end="")
    if use_fourier_features:
        learnable_str = " (learnable)" if fourier_learnable else " (fixed)"
        print(f" ({num_fourier_frequencies} frequencies{learnable_str}, scale={fourier_scale})")
    else:
        print()
    print(f"  - Anisotropic Filter: {use_anisotropic_filter}", end="")
    if use_anisotropic_filter:
        print(f" (kernel size: {filter_kernel_size}x{filter_kernel_size}x{filter_kernel_size})")
    else:
        print()
    print("="*80)
    
    # Output channels: 7 for eigen_composite
    out_channels = 7
    
    model = DenseED(
        in_channels=3, 
        out_channels=out_channels, 
        imsize=100,
        blocks=blocks,
        growth_rate=growth_rate, 
        init_features=init_features,
        drop_rate=drop_rate,
        bn_size=bn_size,
        bottleneck=bottleneck,
        out_activation=None,
        upsample=upsample,
        use_positional_encoding=use_positional_encoding,
        use_fourier_features=use_fourier_features,
        num_fourier_frequencies=num_fourier_frequencies,
        fourier_learnable=fourier_learnable,
        fourier_scale=fourier_scale,
        pe_learnable_scale=pe_learnable_scale,
        pe_learnable_offset=pe_learnable_offset
    )

    model.train()
    model = model.to(device)
    
    # Count parameters and save to config
    n_params, n_conv = model.model_size
    config_data['num_params'] = n_params
    config_data['num_conv_layers'] = n_conv
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)
    
    criterion = torch.nn.MSELoss()
    
    # Use same optimizer and scheduler as original train function
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, threshold=1e-3)
    dataset_id = DatasetHCP(input_dir, sample_name_list=[str(brain_id)])
    dataloader_id = DataLoader(dataset_id, batch_size=1, shuffle=False, num_workers=0)

    # --- loss CSV ---
    loss_csv_path = os.path.join(output_dir_upgraded, "loss.csv")
    if not (resume and os.path.isfile(loss_csv_path)):
        with open(loss_csv_path, 'w') as f:
            f.write("epoch,loss\n")

    # --- load inputs once ---
    vec_path = f'{input_dir}/{brain_id}/{brain_id}_shrinktensor_principal_vector_field.nhdr'
    mask_path = f'{input_dir}/{brain_id}/{brain_id}_shrinktensor_filt_mask.nhdr'
    base_vector = convert.read_nhdr(vec_path).to(device).float()
    base_mask   = convert.read_nhdr(mask_path).to(device).float()

    # --- resume from checkpoint ---
    start_epoch = 0
    last_loss = None
    if resume:
        if resume_from_checkpoint is not None:
            ckpt_path = resume_from_checkpoint
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"[resume] Fichier checkpoint spécifié non trouvé : {ckpt_path}")
        else:
            ckpt_path = os.path.join(output_dir_upgraded, "model_final.pth.tar")
        
        if os.path.isfile(ckpt_path):
            print(f"[resume] Chargement du checkpoint : {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"[resume] État du scheduler restauré")
            start_epoch = int(checkpoint.get('epoch', 0)) + 1
            last_loss = checkpoint.get('loss', None)
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
            
            optimizer.zero_grad()

            u_pred_id = model(input_vector).squeeze()
            mask = batched_id_sample['mask'].squeeze().to(device)
            
            # Apply anisotropic filter if enabled
            if use_anisotropic_filter:
                # Reconstruct SPD metric from network output
                metric_mat = eigen_composite(u_pred_id)  # (H, W, D, 3, 3)
                
                # Apply anisotropic filter
                metric_mat = apply_anisotropic_filter(metric_mat, kernel_size=filter_kernel_size)
                
                # Recompute u_pred_id from filtered metric for PDE loss
                # Note: We need to convert back from metric to u parameters
                # For simplicity, we'll use the filtered metric directly in the PDE
                # by modifying the PDE computation to accept the metric directly
                # However, since pde() expects u_pred_id, we'll need to work around this
                # For now, we'll apply the filter and then use the filtered metric in a modified way
                # Actually, we need to pass the filtered metric to the PDE loss
                # Let's modify the approach: we'll compute the PDE loss with the filtered metric
                # by temporarily storing it and using it in a custom PDE computation
                
                # Store filtered metric and use it in PDE computation
                # We'll need to modify pde() to optionally accept a pre-computed metric
                # For now, let's use a workaround: compute covariant derivative with filtered metric
                from Packages.util import riemann
                tensor_mat = torch.inverse(metric_mat)
                nabla_vv = riemann.covariant_derivative_3d(input_vector.squeeze(), metric_mat, mask, differential_accuracy=2)
                denominator = input_vector.squeeze()[0]*input_vector.squeeze()[0]+input_vector.squeeze()[1]*input_vector.squeeze()[1]+input_vector.squeeze()[2]*input_vector.squeeze()[2]
                denominator += 1-mask
                sigma = (input_vector.squeeze()[0]*nabla_vv[0]+input_vector.squeeze()[1]*nabla_vv[1]+input_vector.squeeze()[2]*nabla_vv[2])/(denominator)*mask
                pde_loss = torch.stack((nabla_vv[0]-sigma*input_vector.squeeze()[0],nabla_vv[1]-sigma*input_vector.squeeze()[1],nabla_vv[2]-sigma*input_vector.squeeze()[2]),0)
            else:
                # Use standard PDE loss function
                pde_loss = pde(u_pred_id, input_vector.squeeze(), mask)
            
            f_pred_id = torch.einsum('...ij,...ij->...ij', pde_loss, mask.expand(3,-1,-1,-1))
            f_true_id = torch.zeros_like(f_pred_id)

            loss_id = criterion(f_pred_id, f_true_id)
            
            # Check for NaN/Inf before backward - STOP TRAINING if detected
            if torch.isnan(loss_id) or torch.isinf(loss_id):
                print("="*80)
                print(f"[ERREUR CRITIQUE] Loss NaN/Inf détectée à l'epoch {epoch}, batch {i}.")
                print(f"Valeur de la loss: {loss_id.item()}")
                print("L'entraînement est arrêté pour éviter la corruption du modèle.")
                print("="*80)
                optimizer.zero_grad()
                # Save current state before stopping
                emergency_ckpt = f'{output_dir_upgraded}/emergency_stop_epoch_{epoch}_batch_{i}.pth.tar'
                torch.save({
                    'epoch': epoch,
                    'batch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': float('nan'),
                    'reason': 'NaN/Inf loss detected'
                }, emergency_ckpt)
                print(f"État d'urgence sauvegardé dans: {emergency_ckpt}")
                # Update config to indicate emergency stop
                if os.path.isfile(config_file):
                    with open(config_file, "r") as f:
                        config_data = json.load(f)
                    config_data['emergency_stop'] = True
                    config_data['stop_epoch'] = epoch
                    config_data['stop_batch'] = i
                    config_data['stop_reason'] = 'NaN/Inf loss detected'
                    with open(config_file, "w") as f:
                        json.dump(config_data, f, indent=2)
                return  # Stop training completely
            
            loss_id.backward()
            optimizer.step()
            epoch_loss_id += loss_id.item()

        # Update scheduler (ReduceLROnPlateau needs loss value)
        scheduler.step(epoch_loss_id)

        last_epoch = epoch

        # write loss
        with open(loss_csv_path, 'a') as f:
            f.write(f"{epoch},{epoch_loss_id}\n")

        # --- save intermediate checkpoints ---
        if epoch % checkpoint_save_frequency == 0:
            ckpt_path = f'{output_dir_upgraded}/checkpoint_epoch_{epoch:04d}.pth.tar'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss_id,
                'config': {
                    "learning_rate": learning_rate,
                    "blocks": blocks,
                    "growth_rate": growth_rate,
                    "init_features": init_features,
                    "bottleneck": bottleneck,
                    "upsample": upsample,
                    "use_positional_encoding": use_positional_encoding,
                    "pe_learnable_scale": pe_learnable_scale,
                    "pe_learnable_offset": pe_learnable_offset,
                    "use_fourier_features": use_fourier_features,
                    "num_fourier_frequencies": num_fourier_frequencies,
                    "fourier_learnable": fourier_learnable,
                    "fourier_scale": fourier_scale,
                    "use_anisotropic_filter": use_anisotropic_filter,
                    "filter_kernel_size": filter_kernel_size,
                    "optimizer": "Adadelta",
                    "scheduler": "ReduceLROnPlateau",
                    "epoch": epoch,
                    "model_type": "DenseED"
                }
            }, ckpt_path)

            # --- save the metric tensor at this epoch ---
            with torch.no_grad():
                u_pred = model(base_vector.unsqueeze(0)).squeeze()
                # Use appropriate method to construct metric
                metric_mat = eigen_composite(u_pred)
                metric_lin = tensors.mat2lin(metric_mat)
                np_metric = np.transpose(metric_lin.detach().cpu().numpy(), (3,2,1,0))

                sitk.WriteImage(sitk.GetImageFromArray(np_metric),
                                f"{metrics_dir}/metric_epoch_{epoch:04d}.nhdr")

        # stop
        if epoch_loss_id < terminating_loss:
            break
    
    if start_epoch >= epoch_num:
        epoch_loss_id = last_loss if last_loss is not None else 0.0

    # --- final model ---
    final_ckpt = f'{output_dir_upgraded}/model_final.pth.tar'
    torch.save({
        'epoch': last_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': epoch_loss_id,
        'config': {
            "learning_rate": learning_rate,
            "blocks": blocks,
            "growth_rate": growth_rate,
            "init_features": init_features,
            "bottleneck": bottleneck,
            "upsample": upsample,
            "use_positional_encoding": use_positional_encoding,
            "pe_learnable_scale": pe_learnable_scale,
            "pe_learnable_offset": pe_learnable_offset,
            "use_fourier_features": use_fourier_features,
            "num_fourier_frequencies": num_fourier_frequencies,
            "fourier_learnable": fourier_learnable,
            "fourier_scale": fourier_scale,
            "use_anisotropic_filter": use_anisotropic_filter,
            "filter_kernel_size": filter_kernel_size,
            "epoch": last_epoch,
            "model_type": "DenseED"
        }
    }, final_ckpt)

    # --- update training_config.json with final epoch ---
    if os.path.isfile(config_file):
        with open(config_file, "r") as f:
            config_data = json.load(f)
        config_data['final_epoch'] = last_epoch
        config_data['final_loss'] = epoch_loss_id
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

    # save final metric
    with torch.no_grad():
        u_pred = model(base_vector.unsqueeze(0)).squeeze()
        # Use appropriate method to construct metric
        metric_mat = eigen_composite(u_pred)
        metric_lin = tensors.mat2lin(metric_mat)
        np_metric = np.transpose(metric_lin.detach().cpu().numpy(), (3,2,1,0))

        sitk.WriteImage(sitk.GetImageFromArray(np_metric),
                        f"{metrics_dir}/metric_final.nhdr")
    
    # Plot loss
    plot_loss()

# run
if __name__ == '__main__':

    # Original training with eigen_composite
    # train(brain_id, input_dir, output_dir, gpu_device, epoch_num, learning_rate,
    #        terminating_loss, checkpoint_save_frequency, resume=RESUME)
    
    # Upgraded training with positional encoding and/or Fourier features
    train_upgraded(brain_id, input_dir, output_dir, gpu_device, epoch_num, learning_rate,
                   terminating_loss, checkpoint_save_frequency, resume=RESUME, resume_from_checkpoint=RESUME_FILE,
                   blocks=[10, 10, 10],
                   bottleneck=True,  # Same as model3D default
                   upsample='nearest',
                   use_positional_encoding=False,  # Set to True to enable
                   use_fourier_features=False,  # Set to True to enable
                   num_fourier_frequencies=10,
                   fourier_learnable=False,  # Set to True to make frequencies learnable
                   fourier_scale=1.0,  # Scaling factor for Fourier frequencies
                   pe_learnable_scale=False,  # Set to True to make PE scaling learnable
                   pe_learnable_offset=False,  # Set to True to make PE offset learnable
                   use_anisotropic_filter=True,  # Set to True to enable
                   filter_kernel_size=3)
    
  



