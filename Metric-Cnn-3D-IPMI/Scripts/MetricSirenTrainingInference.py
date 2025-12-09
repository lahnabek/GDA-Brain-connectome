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
from modelSIREN import create_siren_model
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd

# ----------- CONFIG -------------
brain_id = "111312"
input_dir = "../Brains"
output_dir = "../Checkpoints_SIREN_256_512_512_256_128"  # Include SIREN and architecture in name
gpu_device = -1
epoch_num = 1700  # 10000
learning_rate = 5*1e-3
terminating_loss = 1e6
checkpoint_save_frequency = 100
RESUME = True
RESUME_FILE = None  # None pour le dernier model
# SIREN architecture: hidden layers
SIREN_HIDDEN_LAYERS = [256, 512, 512, 256, 128]  # Equivalent to ~600K-800K parameters
SIREN_W0 = 30.0  # Frequency scaling for hidden layers
SIREN_W0_FIRST = 30.0  # Frequency scaling for first layer
SIREN_NUM_FREQUENCIES = 10  # Number of Fourier frequency bands
SIREN_FOURIER_LEARNABLE = True  # Set to True to learn frequencies
SIREN_FOURIER_SCALE = 1.0  # Scale for frequency range
SIREN_USE_FOURIER_FEATURES = True  # Enable Fourier Features encoding
# --------------------------------


def create_coordinate_grid(shape, device='cpu'):
    """
    Create coordinate grid for SIREN input.
    
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
        f"epochs={loss_df['epoch'].iloc[-1]} | lr={learning_rate} | SIREN layers={SIREN_HIDDEN_LAYERS}",
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
        "model_type": "SIREN",
        "hidden_layers": SIREN_HIDDEN_LAYERS,
        "w0": SIREN_W0,
        "w0_first": SIREN_W0_FIRST,
        "num_frequencies": SIREN_NUM_FREQUENCIES,
        "fourier_learnable": SIREN_FOURIER_LEARNABLE,
        "fourier_scale": SIREN_FOURIER_SCALE,
        "use_fourier_features": SIREN_USE_FOURIER_FEATURES,
        "resume": resume,
    })
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)

    # Create SIREN model
    model = create_siren_model(
        hidden_layers=SIREN_HIDDEN_LAYERS,
        w0=SIREN_W0,
        w0_first=SIREN_W0_FIRST,
        num_frequencies=SIREN_NUM_FREQUENCIES,
        fourier_learnable=SIREN_FOURIER_LEARNABLE,
        fourier_scale=SIREN_FOURIER_SCALE,
        use_fourier_features=SIREN_USE_FOURIER_FEATURES
    )

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

            # SIREN forward: takes coordinates and vector field
            # model handles batch dimension automatically
            u_pred_id = model(coords, input_vector)
            mask = batched_id_sample['mask'].squeeze().to(device)
            pde_loss = pde(u_pred_id.squeeze(), input_vector.squeeze(), mask)
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
                    "hidden_layers": SIREN_HIDDEN_LAYERS,
                    "w0": SIREN_W0,
                    "w0_first": SIREN_W0_FIRST,
                    "num_frequencies": SIREN_NUM_FREQUENCIES,
                    "fourier_learnable": SIREN_FOURIER_LEARNABLE,
                    "fourier_scale": SIREN_FOURIER_SCALE,
                    "use_fourier_features": SIREN_USE_FOURIER_FEATURES,
                    "epoch": epoch,
                    "model_type": "SIREN"
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
            "hidden_layers": SIREN_HIDDEN_LAYERS,
            "w0": SIREN_W0,
            "w0_first": SIREN_W0_FIRST,
            "num_frequencies": SIREN_NUM_FREQUENCIES,
            "fourier_learnable": SIREN_FOURIER_LEARNABLE,
            "fourier_scale": SIREN_FOURIER_SCALE,
            "use_fourier_features": SIREN_USE_FOURIER_FEATURES,
            "epoch": last_epoch,
            "model_type": "SIREN"
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


def analyze_siren_hyperparameters(checkpoint_dir, brain_id="111312"):
    """
    Analyse les hyperparamètres SIREN à partir d'un modèle entraîné.
    
    Cette fonction :
    1. Charge le modèle final et analyse ses paramètres
    2. Analyse la courbe de loss pour suggérer un meilleur learning rate
    3. Analyse les fréquences Fourier apprises (si learnable=True)
    4. Suggère des hyperparamètres optimaux basés sur l'analyse
    
    Args:
        checkpoint_dir: Chemin vers le dossier checkpoint (ex: "../Checkpoints_SIREN_...")
        brain_id: ID du cerveau
    
    Returns:
        dict: Dictionnaire avec recommandations d'hyperparamètres
    """
    import pandas as pd
    import torch
    
    print("="*80)
    print("ANALYSE DES HYPERPARAMÈTRES SIREN")
    print("="*80)
    
    checkpoint_path = f"{checkpoint_dir}/{brain_id}/model_final.pth.tar"
    loss_csv_path = f"{checkpoint_dir}/{brain_id}/loss.csv"
    config_path = f"{checkpoint_dir}/{brain_id}/training_config.json"
    
    recommendations = {}
    
    # 1. Analyser la courbe de loss
    if os.path.isfile(loss_csv_path):
        print("\n[1] ANALYSE DE LA COURBE DE LOSS")
        print("-" * 80)
        df = pd.read_csv(loss_csv_path)
        
        initial_loss = df.iloc[0]['loss']
        final_loss = df.iloc[-1]['loss']
        total_reduction = (1 - final_loss/initial_loss) * 100
        
        print(f"Loss initiale: {initial_loss:.2e}")
        print(f"Loss finale: {final_loss:.2e}")
        print(f"Réduction totale: {total_reduction:.1f}%")
        
        # Analyser les dernières epochs
        last_100 = df.tail(100)
        last_50 = df.tail(50)
        
        # Calculer la pente (vitesse de descente)
        if len(last_50) > 1:
            slope = np.polyfit(last_50['epoch'], last_50['loss'], 1)[0]
            print(f"Pente (dernières 50 epochs): {slope:.2e} per epoch")
            
            # Calculer la variabilité
            std_last_100 = last_100['loss'].std()
            mean_last_100 = last_100['loss'].mean()
            cv = (std_last_100 / mean_last_100) * 100  # Coefficient of variation
            print(f"Variabilité (CV sur 100 dernières epochs): {cv:.2f}%")
            
            # Suggérer un learning rate
            current_lr = 3e-3  # Valeur par défaut, sera lue du config si disponible
            
            if os.path.isfile(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    current_lr = config.get('learning_rate', 3e-3)
            
            print(f"\nLearning rate actuel: {current_lr:.2e}")
            
            # Recommandations basées sur l'analyse
            if abs(slope) < 1e4:  # Très faible descente
                if cv < 1.0:  # Stable mais ne descend plus
                    suggested_lr = current_lr * 0.5  # Réduire pour fine-tuning
                    print(f"⚠️  La loss stagne. Suggéré: {suggested_lr:.2e} (réduction de 50%)")
                else:  # Instable
                    suggested_lr = current_lr * 0.3  # Réduire plus agressivement
                    print(f"⚠️  La loss oscille. Suggéré: {suggested_lr:.2e} (réduction de 70%)")
            elif abs(slope) > 1e5:  # Descente rapide
                suggested_lr = current_lr * 1.2  # Peut augmenter légèrement
                print(f"✓ La loss descend bien. Suggéré: {suggested_lr:.2e} (augmentation de 20%)")
            else:  # Descente modérée
                suggested_lr = current_lr  # Garder similaire
                print(f"✓ La loss descend modérément. Suggéré: {suggested_lr:.2e} (identique)")
            
            recommendations['learning_rate'] = suggested_lr
            recommendations['loss_analysis'] = {
                'initial': float(initial_loss),
                'final': float(final_loss),
                'reduction_pct': float(total_reduction),
                'slope': float(slope),
                'variability_cv': float(cv)
            }
    
    # 2. Analyser les paramètres du modèle (fréquences Fourier si learnable)
    if os.path.isfile(checkpoint_path):
        print("\n[2] ANALYSE DES PARAMÈTRES DU MODÈLE")
        print("-" * 80)
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Chercher les fréquences Fourier apprises
            fourier_freqs = None
            for key in state_dict.keys():
                if 'fourier_encoder.frequencies' in key:
                    fourier_freqs = state_dict[key].cpu().numpy()
                    break
            
            if fourier_freqs is not None:
                print(f"Fréquences Fourier apprises trouvées: shape {fourier_freqs.shape}")
                print(f"Plage des fréquences: [{fourier_freqs.min():.3f}, {fourier_freqs.max():.3f}]")
                print(f"Moyenne: {fourier_freqs.mean():.3f}, Std: {fourier_freqs.std():.3f}")
                
                # Analyser la distribution
                if fourier_freqs.max() > 10:
                    print(f"⚠️  Fréquences très élevées détectées. FOURIER_SCALE pourrait être réduit.")
                    suggested_scale = max(0.5, SIREN_FOURIER_SCALE * 0.7)
                    recommendations['fourier_scale'] = suggested_scale
                elif fourier_freqs.max() < 1:
                    print(f"⚠️  Fréquences très faibles. FOURIER_SCALE pourrait être augmenté.")
                    suggested_scale = min(2.0, SIREN_FOURIER_SCALE * 1.5)
                    recommendations['fourier_scale'] = suggested_scale
                else:
                    print(f"✓ Plage de fréquences raisonnable.")
                    recommendations['fourier_scale'] = SIREN_FOURIER_SCALE
            else:
                print("Fréquences Fourier fixes (non apprenables) - pas d'analyse possible")
                recommendations['fourier_scale'] = SIREN_FOURIER_SCALE
            
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
    
    # 3. Recommandations générales basées sur la littérature
    print("\n[3] RECOMMANDATIONS GÉNÉRALES")
    print("-" * 80)
    
    # W0 et W0_FIRST
    print(f"W0 actuel: {SIREN_W0}, W0_FIRST: {SIREN_W0_FIRST}")
    if SIREN_W0 == SIREN_W0_FIRST:
        print("✓ W0 et W0_FIRST identiques - standard pour SIREN")
        recommendations['w0'] = SIREN_W0
        recommendations['w0_first'] = SIREN_W0_FIRST
    else:
        print("⚠️  W0 et W0_FIRST différents - peut être optimisé")
        recommendations['w0'] = 30.0  # Valeur standard
        recommendations['w0_first'] = 30.0
    
    # Num frequencies
    print(f"NUM_FREQUENCIES: {SIREN_NUM_FREQUENCIES}")
    if SIREN_NUM_FREQUENCIES >= 8 and SIREN_NUM_FREQUENCIES <= 12:
        print("✓ Nombre de fréquences dans la plage recommandée (8-12)")
        recommendations['num_frequencies'] = SIREN_NUM_FREQUENCIES
    else:
        print("⚠️  Nombre de fréquences hors plage recommandée")
        recommendations['num_frequencies'] = 10  # Valeur standard
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DES RECOMMANDATIONS")
    print("="*80)
    for key, value in recommendations.items():
        if key != 'loss_analysis':
            print(f"{key}: {value}")
    
    return recommendations


# run
if __name__ == '__main__':

    train(brain_id, input_dir, output_dir, gpu_device, epoch_num, learning_rate,
          terminating_loss, checkpoint_save_frequency, resume=RESUME)
    
    # Analyser les hyperparamètres après entraînement
    #analyze_siren_hyperparameters(output_dir, brain_id)

