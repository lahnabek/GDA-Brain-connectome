import os, argparse, sys
sys.path.append('../Packages')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import torch, skimage

import Packages.data.convert as convert
import Packages.util.tensors as tensors
import SimpleITK as sitk

from torch.utils.data import DataLoader
from tqdm import tqdm
from pde import *
from dataset import *
from model import *
from plot import *

def train(brain_id, input_dir, output_dir, gpu_device, epoch_num, learning_rate, terminating_loss, checkpoint_save_frequency):
    # device = torch.device('cuda')
    # torch.cuda.set_device(gpu_device)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # # --- device selection block (safe across CUDA, MPS, CPU) ---
    # if torch.cuda.is_available():
    #     device = torch.device(f"cuda:{gpu_device}")
    #     torch.cuda.set_device(gpu_device)
    # elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    #     # Apple Metal backend
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
    device = torch.device("cpu")
    # Use set_default_dtype instead of deprecated set_default_tensor_type
    torch.set_default_dtype(torch.float32)

    print("Using device:", device)
    # -----------------------------------------------------------

    output_dir = f'{output_dir}/{brain_id}'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    model = DenseED(in_channels=2, out_channels=3, 
                    imsize=100,
                    blocks=[6, 8, 6],
                    growth_rate=16, 
                    init_features=48,
                    drop_rate=0,
                    bottleneck=True, 
                    out_activation=None,
                    upsample='nearest')
    model.train()
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    dataset_id = ImageDataset(input_dir, sample_name_list=[brain_id])
    dataloader_id = DataLoader(dataset_id, batch_size=1, shuffle=False, num_workers=0)

    with open(f'{output_dir}/loss.txt', 'w+') as f:
        f.write(f'From 0 to {epoch_num}\n')
        f.write(f'MSE; Adadelta: lr={learning_rate}\n')

    for epoch in tqdm(range(epoch_num)):
        epoch_loss_id = 0

        for i, batched_id_sample in enumerate(dataloader_id):
            input_id = batched_id_sample['vector_field'].float().to(device)
            input_id.requires_grad = True
            mask = batched_id_sample['mask'].float()
            u_pred_id = model(input_id)[...,1:146,1:175]
            pde_loss = pde(u_pred_id.squeeze(), input_id.squeeze(), mask.squeeze(), differential_accuracy=2)
            f_pred_id = torch.einsum('...ij,...ij->...ij', pde_loss, mask.squeeze().unsqueeze(0).expand(2,-1,-1))
            f_true_id = torch.zeros_like(f_pred_id)

            optimizer.zero_grad()
            loss_id = criterion(f_pred_id, f_true_id)
            loss_id.backward()
            epoch_loss_id += loss_id.item()
            optimizer.step()
        scheduler.step(epoch_loss_id)

        with open(f'{output_dir}/loss.txt', 'a') as f:
            f.write(f'{epoch_loss_id}\n')

        if epoch%checkpoint_save_frequency==0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_id_state_dict': optimizer.state_dict(),
            'loss_id': epoch_loss_id,
            }, f'{output_dir}/model.pth.tar')

        if epoch_loss_id<terminating_loss:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_id_state_dict': optimizer.state_dict(),
            'loss_id': epoch_loss_id,
            }, f'{output_dir}/model.pth.tar')
            break
            
    checkpoint = torch.load(f'{output_dir}/model.pth.tar')
    model = DenseED(in_channels=2, out_channels=3, 
                    imsize=100,
                    blocks=[6, 8, 6],
                    growth_rate=16, 
                    init_features=48,
                    drop_rate=0,
                    out_activation=None,
                    upsample='nearest')
    model.load_state_dict(checkpoint['model_state_dict'])

    vector_lin = convert.read_nhdr(f'{input_dir}/{brain_id}/{brain_id}_vector_field.nhdr').to(device).permute(2,0,1).float()
    mask = convert.read_nhdr(f'{input_dir}/{brain_id}/{brain_id}_filt_mask.nhdr').permute(1,0).float()
    eroded_mask = skimage.morphology.erosion(mask.cpu().numpy(), skimage.morphology.square(4))

    u_pred = model(vector_lin.unsqueeze(0))
    u_pred = u_pred[...,1:146,1:175].squeeze()
    s_pred = tensors.lin2mat(u_pred)

    metric_pred_mat = matrix_exp_2d(s_pred)
    metric_pred_lin = tensors.mat2lin(metric_pred_mat)

    file_name = f'{input_dir}/{brain_id}/{brain_id}_learned_metric_final.nhdr'
    sitk.WriteImage(sitk.GetImageFromArray(np.transpose(metric_pred_lin.cpu().detach().numpy(),(2,1,0))), file_name)
    print(f'{file_name} saved')

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

parser = argparse.ArgumentParser()
parser.add_argument('--brain_id', type=str, required=True, help='the HCP subject ID')
parser.add_argument('--input_dir', type=str, required=True, help='path to the brain data')
parser.add_argument('--output_dir', type=str, required=True, help='path to model checkpoint')
parser.add_argument('--gpu_device', type=int, required=True, help='an integer for the accumulator')
parser.add_argument('--epoch_num', type=int, required=False, help='total epochs for training')
parser.add_argument('--learning_rate', type=float, required=False, help='initial learning rate of model')
parser.add_argument('--terminating_loss', type=float, required=False, help='loss threshold for termination')
parser.add_argument('--checkpoint_save_frequency', type=int, required=False, help='frequency of checkpoint save')
args = parser.parse_args()

train(brain_id=args.brain_id, input_dir=args.input_dir, output_dir=args.output_dir, gpu_device=args.gpu_device, epoch_num=args.epoch_num, learning_rate=args.learning_rate, terminating_loss=args.terminating_loss, checkpoint_save_frequency=args.checkpoint_save_frequency)