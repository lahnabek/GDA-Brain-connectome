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
from model3D import *
import matplotlib.pyplot as plt
import pandas as pd
# ----------- CONFIG -------------
brain_id = "111312"
input_dir = "../Brains"
output_dir = "../Checkpoints_40_30_40"
gpu_device = -1
epoch_num = 50 #10000
learning_rate = 1e-3
terminating_loss = 1e6
checkpoint_save_frequency = 5
RESUME = False
BLOCKS = [40,30,40] # [40,30,40]
# --------------------------------

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
          terminating_loss=1e6, checkpoint_save_frequency=1000, resume=RESUME):

    device = torch.device("cpu")

    output_dir = f'{output_dir}/{brain_id}'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    metrics_dir = f"{output_dir}/metrics"
    os.makedirs(metrics_dir, exist_ok=True)   # dossier où on enregistrera toutes les métriques (intermédiaires)

    # --- save config ---
    config_file = os.path.join(output_dir, "training_config.json")
    with open(config_file, "w") as f:
        json.dump({
            "brain_id": brain_id,
            "epochs": epoch_num,
            "learning_rate": learning_rate,
            "checkpoint_frequency": checkpoint_save_frequency,
            "blocks": BLOCKS,
            "resume": resume,
        }, f, indent=2)

    blocks = BLOCKS
    model = DenseED(in_channels=3, out_channels=7, 
                    imsize=100,
                    blocks=blocks,
                    growth_rate=16, 
                    init_features=48,
                    drop_rate=0,
                    out_activation=None,
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

    # --- reprise depuis le dernier checkpoint éventuel ---
    start_epoch = 0
    last_loss = None
    if resume:
        ckpt_pattern = os.path.join(output_dir, "checkpoint_epoch_*.pth.tar")
        ckpt_files = sorted(glob.glob(ckpt_pattern))
        if ckpt_files:
            # on prend le dernier checkpoint en se basant sur le nom
            last_ckpt = ckpt_files[-1]
            print(f"[resume] Chargement du checkpoint : {last_ckpt}")
            checkpoint = torch.load(last_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = int(checkpoint.get('epoch', 0)) + 1
            last_loss = checkpoint.get('loss', None)
            print(f"[resume] Reprise à l'epoch {start_epoch} (loss précédente={last_loss})")
        else:
            print("[resume] Aucun checkpoint trouvé, entraînement à partir de zéro.")

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

    # save final metric
    u_pred = model(base_vector.unsqueeze(0)).squeeze()
    metric_mat = eigen_composite(u_pred)
    metric_lin = tensors.mat2lin(metric_mat)
    np_metric = np.transpose(metric_lin.detach().cpu().numpy(), (3,2,1,0))

    sitk.WriteImage(sitk.GetImageFromArray(np_metric),
                    f"{metrics_dir}/metric_final.nhdr") # tensor_lin pour geodesic ?
    plot_loss()


# run
if __name__ == '__main__':

    train(brain_id, input_dir, output_dir, gpu_device, epoch_num, learning_rate,
          terminating_loss, checkpoint_save_frequency, resume=RESUME)


