import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

sys.path.insert(0, PROJECT_ROOT)
import torch
import glob
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import Packages.data.convert as convert
from Packages.util import tensors
from Packages.algo.geodesic import geodesicpath_3d
from Packages.algo.metricModSolver import solve_3d  
from Packages.util.riemann import covariant_derivative_3d


def select_seeded_mask_point(mask, seed=0, margin=2):
    """
    Sélectionne un point reproductible dans le mask à partir d’une seed.
    
    mask : array 3D binaire
    seed : int (reproductibilité)
    margin : nombre de voxels à exclure des bords pour éviter les problèmes
    
    Retour :
        (x, y, z) tuple d'indices valides
    """

    # Trouver les indices où mask == 1
    coords = np.argwhere(mask > 0)

    if coords.size == 0:
        raise ValueError("Mask is empty !")

    # On enlève les points trop proches du bord
    X, Y, Z = mask.shape
    valid = coords[
        (coords[:,0] > margin) & (coords[:,0] < X - margin) &
        (coords[:,1] > margin) & (coords[:,1] < Y - margin) &
        (coords[:,2] > margin) & (coords[:,2] < Z - margin)
    ]

    if valid.size == 0:
        raise ValueError("Mask too small after margin filtering")

    # Générateur déterministe
    rng = np.random.default_rng(seed)

    # Sélection déterministe parmi les points valides
    idx = rng.integers(0, len(valid))

    return tuple(valid[idx])



def plot_geodesics_from_checkpoints(
        metrics_dir,
        mask_path,
        seed_start_point,
        init_velocity,
        num_intermediate=3,
        delta_t=0.15,
        iter_num=8000, #1800 dans la fonction 
        stop_angle=30,
        DIR = False,
        vector_path=None,
        max_length=None):  # Distance maximale du point initial (en voxels)

    # -------------------------------
    # 1) Récupérer tous les fichiers métriques triés
    # -------------------------------
    metric_files = sorted(glob.glob(f"{metrics_dir}/metric_epoch_*.nhdr"))
    if len(metric_files) == 0:
        raise RuntimeError("Aucun metric_epoch_XXXX.nhdr trouvé.")

    final_file = f"{metrics_dir}/metric_final.nhdr"
    if os.path.isfile(final_file):
        metric_files.append(final_file)

    # -------------------------------
    # 2) Sélectionner checkpoints équidistants
    # -------------------------------
    num_checkpoints = num_intermediate + 2
    idx_selected = np.round(np.linspace(0, len(metric_files)-1, num_checkpoints)).astype(int)
    selected_files = [metric_files[i] for i in idx_selected]

    print("Fichiers sélectionnés :")
    for f in selected_files:
        print("  ", f)

    # -------------------------------
    # 3) Charger mask
    # -------------------------------
    mask = convert.read_nhdr(mask_path).cpu().numpy()

    # -------------------------------
    # 3b) Champ de vecteurs indépendant de la métrique
    # -------------------------------
    
    print(f"Champ de vecteurs (indépendant de g) : {vector_path}")
    vec_torch = convert.read_nhdr(vector_path).cpu()
    # on suppose (3,X,Y,Z)
    vector_field = vec_torch.numpy().astype(float)

    # -------------------------------
    # 4) Convertir vector_field en format (X,Y,Z,3) pour la sélection et l'intégration
    # -------------------------------
    X, Y, Z = mask.shape
    if vector_field.shape[0] == 3:
        vector_field_xyz = np.transpose(vector_field, (1, 2, 3, 0))  # (X,Y,Z,3)
    else:
        vector_field_xyz = vector_field  # Déjà (X,Y,Z,3)

    # -------------------------------
    # 5) Sélectionner le point de départ (centre de la région haute, comme dans plot_geodesics_on_slice_from_checkpoints)
    # -------------------------------
    # Utiliser une slice au milieu (z) pour sélectionner le point dans la région haute
    start_point, _ = select_oriented_point_on_slice(
        mask,
        vector_field_xyz,
        axis="z",
        slice_index=Z // 2,  # Slice au milieu
        seed=seed_start_point,
        margin=2,
        min_inplane=0.0,
        use_center_high_region=True  # Centre de la région haute (genu du corpus callosum)
    )
    sx, sy, sz = start_point
    print(f"Start point = {start_point}")

    # -------------------------------
    # 6) Couleurs + Figure
    # -------------------------------
    cmap = plt.cm.viridis(np.linspace(0, 1, len(selected_files)))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # -------------------------------
    # 7) Pour chaque métrique : calculer la géodésique
    # -------------------------------
    for k, metric_file in enumerate(selected_files):

        print(f"\n=== Processing {metric_file} ===")

        # Lire metric (format NHDR) – ce fichier contient la métrique g sous forme linéarisée (6 composantes)
        tens_lin = convert.read_nhdr(metric_file).cpu().numpy()  # (X,Y,Z,6) ou (6,X,Y,Z)
        if tens_lin.shape[-1] == 6:
            metric_6 = tens_lin                     # (X,Y,Z,6)
            tensor_lin_g = np.transpose(tens_lin, (3, 0, 1, 2))  # (6,X,Y,Z)
        else:
            tensor_lin_g = tens_lin                             # (6,X,Y,Z)
            metric_6 = np.transpose(tens_lin, (1, 2, 3, 0))     # (X,Y,Z,6)

        # Construire g(x) en 3x3
        metric_3x3 = tensors.tens_6_to_tens_3x3(metric_6)       # (X,Y,Z,3,3)

        # Champ de directions : champ de vecteurs d'entrée (indépendant de g)
        vector_lin = vector_field  # (3,X,Y,Z)

        # Inverse de la métrique pour geodesicpath_3d
        g_inv_3x3 = np.linalg.inv(metric_3x3)                  # (X,Y,Z,3,3)
        tensor_lin_for_geo_6 = tensors.tens_3x3_to_tens_6(g_inv_3x3)  # (X,Y,Z,6)
        tensor_lin_for_geo = np.transpose(tensor_lin_for_geo_6, (3, 0, 1, 2))  # (6,X,Y,Z)

        # Géodésique
        gx, gy, gz = geodesicpath_3d(
            tensor_lin=tensor_lin_for_geo,
            vector_lin=vector_lin,
            mask_image=mask,
            start_coordinate=start_point,
            initial_velocity=init_velocity,
            delta_t=delta_t,
            iter_num=iter_num,
            stop_angle=stop_angle,
            both_directions=DIR
        )
        
        # Tronquer la géodésique si max_length est spécifié
        if max_length is not None:
            gx, gy, gz = truncate_geodesic_by_length(gx, gy, gz, start_point, max_length)

        # Créer un label plus clair
        metric_name = os.path.basename(metric_file)
        if "metric_final" in metric_name:
            label = "CNN-learned metric (final)"
        elif "metric_epoch" in metric_name:
            try:
                epoch_num = metric_name.split("_")[-1].split(".")[0]
                label = f"CNN-learned metric (epoch {epoch_num})"
            except:
                label = f"CNN-learned metric ({metric_name})"
        else:
            label = f"CNN-learned metric ({metric_name})"

        ax.plot(gx, gy, gz, color=cmap[k], linewidth=2, label=label, alpha=0.8)

    # -------------------------------
    # 8) Géodésique ground truth (intégration directe du vector field)
    # -------------------------------
    print("\n=== Computing ground truth path (vector field integration) ===")
    gx_gt, gy_gt, gz_gt = integrate_vector_field(
        vector_field_xyz,
        start_point=start_point,
        mask=mask,
        delta_t=delta_t,
        iter_num=iter_num,
        both_directions=DIR,
        max_length=max_length
    )
    
    ax.plot(gx_gt, gy_gt, gz_gt, color="red", linewidth=2.5,
            label="Ground truth (vector field)", zorder=10)

    # -------------------------------
    # 9) Afficher le point de départ
    # -------------------------------
    ax.scatter(sx, sy, sz, color="black", s=100, marker="*", zorder=15,
               label="Seed point", edgecolors="white", linewidths=1)

    # -------------------------------
    # 10) Amélioration de l'affichage
    # -------------------------------
    ax.set_title("Geodesic Tractography Evolution During Training", 
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("X (mm)", fontsize=11)
    ax.set_ylabel("Y (mm)", fontsize=11)
    ax.set_zlabel("Z (mm)", fontsize=11)
    
    # Légende en dessous de la figure
    ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3, 
              fontsize=9, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.show()


def select_oriented_point_on_slice(mask, vec_field, axis="z",
                                   slice_index=None, seed=0,
                                   margin=2, min_inplane=0.,
                                   use_center_high_region=True):
    """
    Sélectionne un point (x,y,z) :
      - dans le mask,
      - dans la slice (axis, slice_index),
      - où la direction principale TDir est bien dans le plan
        (grande composante dans le plan),
      - à au moins `margin` voxels des bords,
      - de façon reproductible via `seed`.
    
    Si use_center_high_region=True, sélectionne le centre de la région haute
    du masque (genu du corpus callosum) au lieu d'un point aléatoire.
    
    Retourne (point, slice_index_effective).
    """
    X, Y, Z = mask.shape
    # Choix automatique de la slice si None
    if slice_index is None:
        if axis == "z":
            slice_index = Z // 2
        elif axis == "y":
            slice_index = Y // 2
        elif axis == "x":
            slice_index = X // 2
        else:
            raise ValueError("axis doit être 'x','y' ou 'z'")

    rng = np.random.default_rng(seed)

    if axis == "z":
        sl_mask = mask[:, :, slice_index]              # (X,Y)
        sl_vec = vec_field[:, :, slice_index, :]       # (X,Y,3)
        # composante dans le plan (x,y)
        inplane = np.sqrt(sl_vec[...,0]**2 + sl_vec[...,1]**2)
        coords = np.argwhere((sl_mask > 0) & (inplane > min_inplane))
        if coords.size == 0:
            raise ValueError("Pas de voxels avec vecteur dans le plan (z).")
        valid = coords[
            (coords[:,0] > margin) & (coords[:,0] < X - margin) &
            (coords[:,1] > margin) & (coords[:,1] < Y - margin)
        ]
        if valid.size == 0:
            raise ValueError("Pas de voxels valides après margin (z).")
        
        if use_center_high_region:
            # Sélectionner la région haute (Y élevé = partie supérieure/anterior)
            # Prendre le tiers supérieur des voxels valides selon Y
            y_coords = valid[:, 1]
            y_threshold = np.percentile(y_coords, 66.67)  # Top 33% selon Y
            high_region = valid[y_coords >= y_threshold]
            if high_region.size == 0:
                high_region = valid  # Fallback si aucun point dans la région haute
            # Centre de la région haute (moyenne des coordonnées)
            center = high_region.mean(axis=0).astype(int)
            # Trouver le point valide le plus proche du centre
            distances = np.sum((high_region - center)**2, axis=1)
            idx_closest = np.argmin(distances)
            x, y = high_region[idx_closest]
        else:
            x, y = valid[rng.integers(0, len(valid))]
        z = slice_index

    elif axis == "y":
        sl_mask = mask[:, slice_index, :]              # (X,Z)
        sl_vec = vec_field[:, slice_index, :, :]       # (X,Z,3)
        # composante dans le plan (x,z)
        inplane = np.sqrt(sl_vec[...,0]**2 + sl_vec[...,2]**2)
        coords = np.argwhere((sl_mask > 0) & (inplane > min_inplane))
        if coords.size == 0:
            raise ValueError("Pas de voxels avec vecteur dans le plan (y).")
        valid = coords[
            (coords[:,0] > margin) & (coords[:,0] < X - margin) &
            (coords[:,1] > margin) & (coords[:,1] < Z - margin)
        ]
        if valid.size == 0:
            raise ValueError("Pas de voxels valides après margin (y).")
        
        if use_center_high_region:
            # Sélectionner la région haute (Z élevé = partie supérieure)
            z_coords = valid[:, 1]  # Z est l'axe 1 dans (X,Z)
            z_threshold = np.percentile(z_coords, 66.67)  # Top 33% selon Z
            high_region = valid[z_coords >= z_threshold]
            if high_region.size == 0:
                high_region = valid
            center = high_region.mean(axis=0).astype(int)
            distances = np.sum((high_region - center)**2, axis=1)
            idx_closest = np.argmin(distances)
            x, z = high_region[idx_closest]
        else:
            x, z = valid[rng.integers(0, len(valid))]
        y = slice_index

    else:  # axis == "x"
        sl_mask = mask[slice_index, :, :]              # (Y,Z)
        sl_vec = vec_field[slice_index, :, :, :]       # (Y,Z,3)
        # composante dans le plan (y,z)
        inplane = np.sqrt(sl_vec[...,1]**2 + sl_vec[...,2]**2)
        coords = np.argwhere((sl_mask > 0) & (inplane > min_inplane))
        if coords.size == 0:
            raise ValueError("Pas de voxels avec vecteur dans le plan (x).")
        valid = coords[
            (coords[:,0] > margin) & (coords[:,0] < Y - margin) &
            (coords[:,1] > margin) & (coords[:,1] < Z - margin)
        ]
        if valid.size == 0:
            raise ValueError("Pas de voxels valides après margin (x).")
        
        if use_center_high_region:
            # Sélectionner la région haute (Z élevé = partie supérieure)
            z_coords = valid[:, 1]  # Z est l'axe 1 dans (Y,Z)
            z_threshold = np.percentile(z_coords, 66.67)  # Top 33% selon Z
            high_region = valid[z_coords >= z_threshold]
            if high_region.size == 0:
                high_region = valid
            center = high_region.mean(axis=0).astype(int)
            distances = np.sum((high_region - center)**2, axis=1)
            idx_closest = np.argmin(distances)
            y, z = high_region[idx_closest]
        else:
            y, z = valid[rng.integers(0, len(valid))]
        x = slice_index

    return (int(x), int(y), int(z)), int(slice_index)


def integrate_vector_field(vector_field_xyz, start_point, mask, delta_t=0.15, iter_num=8000, both_directions=True, max_length=None):
    """
    Intègre directement le vector field pour créer une courbe qui le suit.
    Résout dγ/dt = v(γ(t)) où v est le vector field.
    
    Parameters:
    -----------
    vector_field_xyz : array (X, Y, Z, 3)
        Champ de vecteurs dans l'espace
    start_point : tuple (x, y, z)
        Point de départ
    mask : array (X, Y, Z)
        Masque binaire
    delta_t : float
        Pas de temps pour l'intégration
    iter_num : int
        Nombre d'itérations
    both_directions : bool
        Si True, intègre dans les deux directions
    max_length : float, optional
        Distance maximale (euclidienne) du point initial. Si None, pas de limite.
    
    Returns:
    --------
    gx, gy, gz : arrays
        Coordonnées de la courbe intégrée
    """
    X, Y, Z = mask.shape
    sx, sy, sz = start_point
    
    # Normaliser le vector field pour avoir une vitesse constante
    v_norm = np.linalg.norm(vector_field_xyz, axis=3, keepdims=True)
    v_norm = np.where(v_norm > 1e-6, v_norm, 1.0)  # Éviter division par zéro
    vector_field_normalized = vector_field_xyz / v_norm
    
    def interpolate_vector_at_point(x, y, z):
        """Interpole le vector field au point (x, y, z) en utilisant une interpolation trilinéaire simple."""
        # Coordonnées entières
        x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        
        # Vérifier les bornes
        x0 = max(0, min(X-1, x0))
        x1 = max(0, min(X-1, x1))
        y0 = max(0, min(Y-1, y0))
        y1 = max(0, min(Y-1, y1))
        z0 = max(0, min(Z-1, z0))
        z1 = max(0, min(Z-1, z1))
        
        # Coefficients d'interpolation
        dx = x - x0
        dy = y - y0
        dz = z - z0
        
        # Interpolation trilinéaire
        v000 = vector_field_normalized[x0, y0, z0, :]
        v001 = vector_field_normalized[x0, y0, z1, :]
        v010 = vector_field_normalized[x0, y1, z0, :]
        v011 = vector_field_normalized[x0, y1, z1, :]
        v100 = vector_field_normalized[x1, y0, z0, :]
        v101 = vector_field_normalized[x1, y0, z1, :]
        v110 = vector_field_normalized[x1, y1, z0, :]
        v111 = vector_field_normalized[x1, y1, z1, :]
        
        v = (v000 * (1-dx) * (1-dy) * (1-dz) +
             v001 * (1-dx) * (1-dy) * dz +
             v010 * (1-dx) * dy * (1-dz) +
             v011 * (1-dx) * dy * dz +
             v100 * dx * (1-dy) * (1-dz) +
             v101 * dx * (1-dy) * dz +
             v110 * dx * dy * (1-dz) +
             v111 * dx * dy * dz)
        
        return v
    
    # Intégration forward
    gamma = np.zeros((iter_num, 3))
    gamma[0] = np.array([sx, sy, sz])
    
    for i in range(1, iter_num):
        # Point courant
        point = gamma[i-1]
        
        # Vérifier si on est toujours dans le masque
        x_idx, y_idx, z_idx = int(np.round(point[0])), int(np.round(point[1])), int(np.round(point[2]))
        if (x_idx < 0 or x_idx >= X or y_idx < 0 or y_idx >= Y or z_idx < 0 or z_idx >= Z):
            break
        if mask[x_idx, y_idx, z_idx] == 0:
            break
        
        # Interpoler le vector field au point courant
        try:
            v = interpolate_vector_at_point(point[0], point[1], point[2])
        except:
            v = np.array([0.0, 0.0, 0.0])
        
        # Normaliser la vitesse
        v_norm_val = np.linalg.norm(v)
        if v_norm_val > 1e-6:
            v = v / v_norm_val
        else:
            v = np.array([0.0, 0.0, 0.0])
        
        # Vérifier l'arrêt si la vitesse devient trop faible
        if v_norm_val < 1e-6:
            break
        
        # Euler step
        gamma[i] = gamma[i-1] + delta_t * v
        
        # Vérifier la distance maximale du point initial
        if max_length is not None:
            dist_from_start = np.linalg.norm(gamma[i] - gamma[0])
            if dist_from_start > max_length:
                break
    
    # Tronquer aux points valides
    valid_len = i
    gx_forward = gamma[:valid_len, 0]
    gy_forward = gamma[:valid_len, 1]
    gz_forward = gamma[:valid_len, 2]
    
    if both_directions:
        # Intégration backward
        gamma_back = np.zeros((iter_num, 3))
        gamma_back[0] = np.array([sx, sy, sz])
        
        for i in range(1, iter_num):
            # Point courant
            point = gamma_back[i-1]
            
            # Vérifier si on est toujours dans le masque
            x_idx, y_idx, z_idx = int(np.round(point[0])), int(np.round(point[1])), int(np.round(point[2]))
            if (x_idx < 0 or x_idx >= X or y_idx < 0 or y_idx >= Y or z_idx < 0 or z_idx >= Z):
                break
            if mask[x_idx, y_idx, z_idx] == 0:
                break
            
            # Interpoler le vector field au point courant
            try:
                v = interpolate_vector_at_point(point[0], point[1], point[2])
            except:
                v = np.array([0.0, 0.0, 0.0])
            
            # Normaliser la vitesse
            v_norm_val = np.linalg.norm(v)
            if v_norm_val > 1e-6:
                v = v / v_norm_val
            else:
                v = np.array([0.0, 0.0, 0.0])
            
            # Vérifier l'arrêt si la vitesse devient trop faible
            if v_norm_val < 1e-6:
                break
            
            # Euler step backward (direction opposée)
            gamma_back[i] = gamma_back[i-1] - delta_t * v
            
            # Vérifier la distance maximale du point initial
            if max_length is not None:
                dist_from_start = np.linalg.norm(gamma_back[i] - gamma_back[0])
                if dist_from_start > max_length:
                    break
        
        valid_len_back = i
        gx_back = gamma_back[:valid_len_back, 0]
        gy_back = gamma_back[:valid_len_back, 1]
        gz_back = gamma_back[:valid_len_back, 2]
        
        # Concaténer (backward inversé + forward)
        gx = np.concatenate([gx_back[::-1], gx_forward])
        gy = np.concatenate([gy_back[::-1], gy_forward])
        gz = np.concatenate([gz_back[::-1], gz_forward])
    else:
        gx = gx_forward
        gy = gy_forward
        gz = gz_forward
    
    return gx, gy, gz


def truncate_geodesic_by_length(gx, gy, gz, start_point, max_length):
    """
    Tronque une géodésique pour qu'elle ne dépasse pas une distance maximale du point initial.
    
    Parameters:
    -----------
    gx, gy, gz : arrays
        Coordonnées de la géodésique
    start_point : tuple (x, y, z)
        Point de départ
    max_length : float
        Distance maximale (euclidienne) du point initial
    
    Returns:
    --------
    gx_trunc, gy_trunc, gz_trunc : arrays
        Géodésique tronquée
    """
    if max_length is None:
        return gx, gy, gz
    
    sx, sy, sz = start_point
    start_array = np.array([sx, sy, sz])
    
    # Calculer les distances depuis le point initial
    points = np.column_stack([gx, gy, gz])
    distances = np.linalg.norm(points - start_array, axis=1)
    
    # Trouver le dernier point qui respecte la limite
    valid_mask = distances <= max_length
    if not np.any(valid_mask):
        # Si aucun point n'est valide, retourner juste le point initial
        return np.array([sx]), np.array([sy]), np.array([sz])
    
    # Trouver l'index du dernier point valide
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return np.array([sx]), np.array([sy]), np.array([sz])
    
    last_valid_idx = valid_indices[-1]
    
    return gx[:last_valid_idx+1], gy[:last_valid_idx+1], gz[:last_valid_idx+1]


def plot_geodesics_on_slice_from_checkpoints(
    metrics_dir,
    mask_path,
    seed_start_point,
    axis="z",             # 'x', 'y' ou 'z'
    slice_index=None,
    num_intermediate=3,
    delta_t=0.15,
    iter_num=8000,
    stop_angle=120,
    quiver_step=3,        # décimation du champ
    arrow_len=2.0,        # longueur visuelle des directions
    zoom=False,
    zoom_size=20,
    vector_path=None,
    max_length=None,      # Distance maximale du point initial (en voxels)
):
    """
    Visualise les géodésiques projetées sur une coupe 2D
    + champ directionnel (TDir, vecteur propre principal).
    """

    # --- 1) Fichiers métriques ---
    metric_files = sorted(glob.glob(f"{metrics_dir}/metric_epoch_*.nhdr"))
    final_file = os.path.join(metrics_dir, "metric_final.nhdr")
    if os.path.isfile(final_file):
        metric_files.append(final_file)
    if len(metric_files) == 0:
        raise RuntimeError("Aucun fichier metric_epoch_*.nhdr ou metric_final.nhdr trouvé.")

    num_checkpoints = num_intermediate + 2
    idx_selected = np.round(np.linspace(0, len(metric_files)-1, num_checkpoints)).astype(int)
    selected_files = [metric_files[i] for i in idx_selected]

    print("Fichiers sélectionnés :")
    for f in selected_files:
        print("  ", f)

    # --- 2) Mask ---
    mask = convert.read_nhdr(mask_path).cpu().numpy()
    X, Y, Z = mask.shape

    # --- 3) Figure & fond (mask sur la slice) ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # On n'affiche le fond qu'après avoir fixé slice_index (ci-dessous)

    start_point = None   # sera choisi à l'itération k==0
    slice_index_eff = slice_index

    cmap = plt.cm.viridis(np.linspace(0, 1, len(selected_files)))

    # --- 3b) Champ de vecteurs indépendant de la métrique ---

    print(f"Champ de vecteurs (indépendant de g) : {vector_path}")
    vec_torch = convert.read_nhdr(vector_path).cpu()
    vector_field = vec_torch.numpy().astype(float)            # (3,X,Y,Z)
    vector_field_xyz = np.transpose(vector_field, (1, 2, 3, 0))  # (X,Y,Z,3)

    # --- 4) Boucle sur les checkpoints ---
    for k, metric_file in enumerate(selected_files):

        print(f"\n=== Processing {metric_file} ===")

        # Charger métrique (linéarisée) g
        tens_lin = convert.read_nhdr(metric_file).cpu().numpy()  # (X,Y,Z,6) ou (6,X,Y,Z)
        if tens_lin.shape[-1] == 6:
            metric_6 = tens_lin                     # (X,Y,Z,6)
            tensor_lin_g = np.transpose(tens_lin, (3,0,1,2))  # (6,X,Y,Z)
        else:
            tensor_lin_g = tens_lin                  # (6,X,Y,Z)
            metric_6 = np.transpose(tens_lin, (1,2,3,0))      # (X,Y,Z,6)

        # Convertir g -> matrices 3x3
        metric_3x3 = tensors.tens_6_to_tens_3x3(metric_6)     # (X,Y,Z,3,3)

        # Champ de directions pour geodesicpath_3d :
        # on utilise le champ de vecteurs d'entrée (indépendant de g)
        vector_lin = vector_field                               # (3,X,Y,Z)

        # --- choisir start_point & slice à la première itération ---
        if start_point is None:
            start_point, slice_index_eff = select_oriented_point_on_slice(
                mask,
                vector_field_xyz,
                axis=axis,
                slice_index=slice_index,
                seed=seed_start_point,
                margin=2,
                )
            sx, sy, sz = start_point
            print(f"Start point = {start_point}, axis={axis}, slice_index={slice_index_eff}")

            # Fond (mask) une fois que la slice est fixée
            if axis == "z":
                m = mask[:, :, slice_index_eff]      # (X,Y)
            elif axis == "y":
                m = mask[:, slice_index_eff, :]      # (X,Z)
            else:
                m = mask[slice_index_eff, :, :]      # (Y,Z)
            ax.imshow(m.T, origin="lower", cmap="gray", alpha=0.2)

        # --- Géodésique 3D depuis ce start_point (identique pour tous les checkpoints) ---
        # tensor_lin pour les géodésiques doit représenter g^{-1} sous forme (6,X,Y,Z)
        g_inv_3x3 = np.linalg.inv(metric_3x3)                      # (X,Y,Z,3,3)
        tensor_lin_for_geo_6 = tensors.tens_3x3_to_tens_6(g_inv_3x3)   # (X,Y,Z,6)
        tensor_lin_for_geo = np.transpose(tensor_lin_for_geo_6, (3,0,1,2))  # (6,X,Y,Z)

        sx, sy, sz = start_point
        gx, gy, gz = geodesicpath_3d(
            tensor_lin=tensor_lin_for_geo,
            vector_lin=vector_lin,   # champ de vecteurs d'entrée
            mask_image=mask,
            start_coordinate=[sx, sy, sz],
            initial_velocity=None,        # direction auto via direction_3d
            delta_t=delta_t,
            iter_num=iter_num,
            stop_angle=stop_angle,
            both_directions=True
        )
        
        # Tronquer la géodésique si max_length est spécifié
        if max_length is not None:
            gx, gy, gz = truncate_geodesic_by_length(gx, gy, gz, start_point, max_length)

        # --- Projection simple dans le plan ---
        if axis == "z":
            proj_x = gx
            proj_y = gy
        elif axis == "y":
            proj_x = gx
            proj_y = gz
        else:  # 'x'
            proj_x = gy
            proj_y = gz

        # Créer un label plus clair pour la légende
        metric_name = os.path.basename(metric_file)
        if "metric_final" in metric_name:
            label = "CNN-learned metric (final)"
        elif "metric_epoch" in metric_name:
            # Extraire le numéro d'epoch
            try:
                epoch_num = metric_name.split("_")[-1].split(".")[0]
                label = f"CNN-learned metric (epoch {epoch_num})"
            except:
                label = f"CNN-learned metric ({metric_name})"
        else:
            label = f"CNN-learned metric ({metric_name})"
        
        ax.plot(
            proj_x, proj_y,
            color=cmap[k],
            linewidth=2,
            label=label,
            alpha=0.8
        )

        # --- Champ directionnel (vecteur indépendant de g) sur la slice (une seule fois) ---
        if k == 0:
            if axis == "z":
                vf = vector_field_xyz[:, :, slice_index_eff, :]     # (X,Y,3)
                U = vf[::quiver_step, ::quiver_step, 0] * arrow_len
                V = vf[::quiver_step, ::quiver_step, 1] * arrow_len
                Xq, Yq = np.meshgrid(
                    np.arange(0, X, quiver_step),
                    np.arange(0, Y, quiver_step),
                    indexing='ij'
                )

            elif axis == "y":
                vf = vector_field_xyz[:, slice_index_eff, :, :]     # (X,Z,3)
                U = vf[::quiver_step, ::quiver_step, 0] * arrow_len
                V = vf[::quiver_step, ::quiver_step, 2] * arrow_len
                Xq, Yq = np.meshgrid(
                    np.arange(0, X, quiver_step),
                    np.arange(0, Z, quiver_step),
                    indexing='ij'
                )

            else:  # 'x'
                vf = vector_field_xyz[slice_index_eff, :, :, :]     # (Y,Z,3)
                U = vf[::quiver_step, ::quiver_step, 1] * arrow_len
                V = vf[::quiver_step, ::quiver_step, 2] * arrow_len
                Xq, Yq = np.meshgrid(
                    np.arange(0, vf.shape[0], quiver_step),
                    np.arange(0, vf.shape[1], quiver_step),
                    indexing='ij'
                )

            ax.quiver(
                Xq, Yq, U, V,
                color="gray", alpha=0.6,
                angles="xy", scale_units="xy", scale=1.0,
                linewidths=0.5, label="Vector field"
            )

    # --- 5) Géodésique ground truth (intégration directe du vector field) ---
    print("\n=== Computing ground truth path (vector field integration) ===")
    sx, sy, sz = start_point
    gx_gt, gy_gt, gz_gt = integrate_vector_field(
        vector_field_xyz,
        start_point=(sx, sy, sz),
        mask=mask,
        delta_t=delta_t,
        iter_num=iter_num,
        both_directions=True,
        max_length=max_length
    )
    
    # Projection de la géodésique ground truth
    if axis == "z":
        proj_x_gt = gx_gt
        proj_y_gt = gy_gt
    elif axis == "y":
        proj_x_gt = gx_gt
        proj_y_gt = gz_gt
    else:  # 'x'
        proj_x_gt = gy_gt
        proj_y_gt = gz_gt
    
    # Afficher la géodésique ground truth en rouge
    ax.plot(
        proj_x_gt, proj_y_gt,
        color="red",
        linewidth=2.5,
        linestyle="-",
        label="Ground truth (vector field)",
        zorder=10
    )

    # --- 6) Marquer le point de départ ---
    sx, sy, sz = start_point
    if axis == "z":
        sx2, sy2 = sx, sy
    elif axis == "y":
        sx2, sy2 = sx, sz
    else:
        sx2, sy2 = sy, sz

    ax.scatter(sx2, sy2, color="black", s=100, marker="*", zorder=15, label="Seed point", edgecolors="white", linewidths=1)

    # --- 7) Amélioration de l'affichage ---
    ax.set_aspect("equal")
    
    # Titre scientifique en anglais
    axis_labels = {"x": "Sagittal", "y": "Coronal", "z": "Axial"}
    ax.set_title(f"Geodesic Tractography Comparison\n{axis_labels.get(axis, axis)} slice (index {slice_index_eff})", 
                 fontsize=14, fontweight="bold", pad=15)
    
    # Labels des axes
    if axis == "z":
        ax.set_xlabel("X (mm)", fontsize=11)
        ax.set_ylabel("Y (mm)", fontsize=11)
    elif axis == "y":
        ax.set_xlabel("X (mm)", fontsize=11)
        ax.set_ylabel("Z (mm)", fontsize=11)
    else:
        ax.set_xlabel("Y (mm)", fontsize=11)
        ax.set_ylabel("Z (mm)", fontsize=11)
    
    # Légende en dessous de la figure (horizontalement)
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, fontsize=9, framealpha=0.9)

    # --- Option de zoom autour du point de départ ---
    if zoom:
        cx, cy = sx2, sy2
        ax.set_xlim(cx - zoom_size, cx + zoom_size)
        ax.set_ylim(cy - zoom_size, cy + zoom_size)

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # Réserver de l'espace en bas pour la légende
    plt.show()


def analyze_metric_field(metrics_dir, mask_path=None):
    """
    Analyse simple de la métrique finale dans un dossier de checkpoints.

    - Cherche d'abord 'metric_final.nhdr', sinon prend le dernier 'metric_epoch_*.nhdr'
    - Calcule les valeurs propres de g (3x3) et imprime quelques stats globales
      (moyenne / écart-type / min / max) et un indicateur d'anisotropie (lambda_max / lambda_min).
    - Optionnellement, ne garde que les voxels dans le masque si mask_path est fourni.
    """
    # 1) Choisir le fichier de métrique
    metric_final = os.path.join(metrics_dir, "metric_final.nhdr")
    if os.path.isfile(metric_final):
        metric_file = metric_final
    else:
        epoch_files = sorted(glob.glob(os.path.join(metrics_dir, "metric_epoch_*.nhdr")))
        if not epoch_files:
            raise RuntimeError(f"Aucun fichier metric_*.nhdr trouvé dans {metrics_dir}")
        metric_file = epoch_files[-1]

    print(f"[analyse métrique] Fichier utilisé : {metric_file}")

    # 2) Charger la métrique linéarisée
    metric_lin = convert.read_nhdr(metric_file).cpu().numpy()
    # metric_lin peut être (X,Y,Z,6) ou (6,X,Y,Z)
    if metric_lin.shape[-1] == 6:
        metric_6 = metric_lin  # (X,Y,Z,6)
    elif metric_lin.shape[0] == 6:
        metric_6 = np.transpose(metric_lin, (1, 2, 3, 0))  # (X,Y,Z,6)
    else:
        raise ValueError(f"Forme inattendue pour metric_lin: {metric_lin.shape}")

    # 3) Conversion en matrices 3x3
    # metric_3x3 : (X,Y,Z,3,3)
    metric_3x3 = tensors.tens_6_to_tens_3x3(metric_6)

    # 4) Masque éventuel
    if mask_path is not None:
        mask = convert.read_nhdr(mask_path).cpu().numpy().astype(bool)
        if mask.shape != metric_3x3.shape[:3]:
            raise ValueError(f"Dimensions masque {mask.shape} != métrique {metric_3x3.shape[:3]}")
        metric_flat = metric_3x3[mask]
    else:
        metric_flat = metric_3x3.reshape(-1, 3, 3)

    # 5) Valeurs propres
    eigvals, _ = np.linalg.eigh(metric_flat)  # (N,3)
    # On trie pour avoir lambda_min <= ... <= lambda_max
    eigvals_sorted = np.sort(eigvals, axis=1)
    lam_min = eigvals_sorted[:, 0]
    lam_mid = eigvals_sorted[:, 1]
    lam_max = eigvals_sorted[:, 2]

    # 6) Statistiques globales
    def stats(arr, name):
        print(f"{name:10s} : mean={arr.mean():.4e}, std={arr.std():.4e}, "
              f"min={arr.min():.4e}, max={arr.max():.4e}")

    print("\n=== Statistiques des valeurs propres de g ===")
    stats(lam_min, "lambda_min")
    stats(lam_mid, "lambda_mid")
    stats(lam_max, "lambda_max")

    # 7) Indicateur d'anisotropie : condition = lambda_max / lambda_min
    eps = 1e-8
    cond = lam_max / (lam_min + eps)
    print("\n=== Indicateur d'anisotropie (lambda_max / lambda_min) ===")
    stats(cond, "cond")

    # Fraction de voxels "presque isotropes"
    thr_isotropic = 1.2
    frac_iso = np.mean(cond < thr_isotropic)
    print(f"\nFraction de voxels avec cond < {thr_isotropic} (≈ quasi isotrope) : {frac_iso*100:.2f}%")

    return {
        "file": metric_file,
        "lambda_min": lam_min,
        "lambda_mid": lam_mid,
        "lambda_max": lam_max,
        "cond": cond,
        "frac_isotropic": frac_iso,
    }


def plot_metric_eigs_on_slice(metrics_dir,
                              mask_path,
                              axis="z",
                              slice_index=None,
                              vmin=None,
                              vmax=None):
    """
    Figure (idée 1) : cartes 2D des valeurs propres de g (lambda_min, lambda_max)
    et du conditionnement cond = lambda_max / lambda_min sur une coupe.
    """
    # choisir fichier métrique
    metric_final = os.path.join(metrics_dir, "metric_final.nhdr")
    if os.path.isfile(metric_final):
        metric_file = metric_final
    else:
        files = sorted(glob.glob(os.path.join(metrics_dir, "metric_epoch_*.nhdr")))
        if not files:
            raise RuntimeError(f"Aucun metric_*.nhdr trouvé dans {metrics_dir}")
        metric_file = files[-1]

    print(f"[plot_metric_eigs_on_slice] Fichier utilisé : {metric_file}")

    metric_lin = convert.read_nhdr(metric_file).cpu().numpy()
    if metric_lin.shape[-1] == 6:
        metric_6 = metric_lin
    elif metric_lin.shape[0] == 6:
        metric_6 = np.transpose(metric_lin, (1, 2, 3, 0))
    else:
        raise ValueError(f"Forme inattendue pour metric_lin: {metric_lin.shape}")

    metric_3x3 = tensors.tens_6_to_tens_3x3(metric_6)  # (X,Y,Z,3,3)
    X, Y, Z = metric_3x3.shape[:3]

    mask = convert.read_nhdr(mask_path).cpu().numpy().astype(bool)
    if mask.shape != (X, Y, Z):
        raise ValueError(f"Dimensions masque {mask.shape} != métrique {metric_3x3.shape[:3]}")

    # choisir slice
    if slice_index is None:
        slice_index = {"x": X // 2, "y": Y // 2, "z": Z // 2}[axis]

    if axis == "z":
        sl_g = metric_3x3[:, :, slice_index, :, :]
        sl_mask = mask[:, :, slice_index]
    elif axis == "y":
        sl_g = metric_3x3[:, slice_index, :, :, :]
        sl_mask = mask[:, slice_index, :]
    else:  # 'x'
        sl_g = metric_3x3[slice_index, :, :, :, :]
        sl_mask = mask[slice_index, :, :]

    # valeurs propres pour la slice
    flat = sl_g.reshape(-1, 3, 3)
    eigvals, _ = np.linalg.eigh(flat)
    eigvals = np.sort(eigvals, axis=1)
    lam_min = eigvals[:, 0].reshape(sl_mask.shape)
    lam_max = eigvals[:, 2].reshape(sl_mask.shape)
    cond = lam_max / (lam_min + 1e-8)

    # masquage pour l'affichage
    lam_min_disp = np.where(sl_mask, lam_min, np.nan)
    lam_max_disp = np.where(sl_mask, lam_max, np.nan)
    cond_disp = np.where(sl_mask, cond, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    ims = []
    ims.append(axes[0].imshow(lam_min_disp.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax))
    axes[0].set_title(r"$\lambda_{\min}$")
    ims.append(axes[1].imshow(lam_max_disp.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax))
    axes[1].set_title(r"$\lambda_{\max}$")
    ims.append(axes[2].imshow(cond_disp.T, origin="lower", cmap="magma"))
    axes[2].set_title(r"$\lambda_{\max} / \lambda_{\min}$")

    for ax in axes:
        ax.set_aspect("equal")

    fig.colorbar(ims[0], ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(ims[1], ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(ims[2], ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_loss_vs_anisotropy(checkpoint_dir,
                            subject_id="111312",
                            mask_path=None):
    """
    Figure (idée 3) : courbe de loss vs un indicateur global d'anisotropie
    (moyenne de cond = lambda_max / lambda_min sur tout le cerveau) au cours du training.

    On suppose :
      - loss.csv dans checkpoint_dir/subject_id/
      - metrics dans checkpoint_dir/subject_id/metrics/metric_epoch_XXXX.nhdr
    """
    subj_dir = os.path.join(checkpoint_dir, subject_id)
    loss_csv = os.path.join(subj_dir, "loss.csv")
    metrics_dir = os.path.join(subj_dir, "metrics")

    if not os.path.isfile(loss_csv):
        raise FileNotFoundError(loss_csv)
    if not os.path.isdir(metrics_dir):
        raise FileNotFoundError(metrics_dir)

    # loss
    import pandas as pd
    df = pd.read_csv(loss_csv)
    epochs_loss = df["epoch"].to_numpy()
    losses = df["loss"].to_numpy()

    # metrics per epoch
    metric_files = sorted(glob.glob(os.path.join(metrics_dir, "metric_epoch_*.nhdr")))
    if not metric_files:
        raise RuntimeError(f"Aucun metric_epoch_*.nhdr dans {metrics_dir}")

    # optionnel : masque
    if mask_path is not None:
        mask = convert.read_nhdr(mask_path).cpu().numpy().astype(bool)
    else:
        mask = None

    epochs_cond = []
    mean_cond = []

    for f in metric_files:
        name = os.path.basename(f)
        # extraire epoch dans metric_epoch_XXXX.nhdr
        try:
            e = int(name.split("_")[-1].split(".")[0])
        except Exception:
            continue

        metric_lin = convert.read_nhdr(f).cpu().numpy()
        if metric_lin.shape[-1] == 6:
            metric_6 = metric_lin
        elif metric_lin.shape[0] == 6:
            metric_6 = np.transpose(metric_lin, (1, 2, 3, 0))
        else:
            continue
        metric_3x3 = tensors.tens_6_to_tens_3x3(metric_6)
        flat = metric_3x3.reshape(-1, 3, 3)
        eigvals, _ = np.linalg.eigh(flat)
        eigvals = np.sort(eigvals, axis=1)
        lam_min = eigvals[:, 0]
        lam_max = eigvals[:, 2]
        cond = lam_max / (lam_min + 1e-8)

        if mask is not None:
            if mask.shape != metric_3x3.shape[:3]:
                raise ValueError("mask et métrique incompatibles")
            cond = cond[mask.reshape(-1)]

        epochs_cond.append(e)
        mean_cond.append(cond.mean())

    epochs_cond = np.array(epochs_cond)
    mean_cond = np.array(mean_cond)

    # tri par epoch
    order = np.argsort(epochs_cond)
    epochs_cond = epochs_cond[order]
    mean_cond = mean_cond[order]

    fig, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(epochs_loss, losses, "-b", label="loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(epochs_cond, mean_cond, "-r", label="mean cond")
    ax2.set_ylabel("mean cond(lambda_max/lambda_min)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    fig.tight_layout()
    plt.title("Loss vs anisotropie moyenne de la métrique")
    plt.show()


def plot_covariant_error_on_slice(metric_file,
                                  vector_path,
                                  mask_path,
                                  axis="z",
                                  slice_index=None):
    """
    Figure (idée 5) : carte 2D de l'erreur géométrique locale
      e(x) = || nabla_v v || (norme euclidienne)
    pour un champ de vecteurs v indépendant de la métrique.
    """
    # charger métrique
    metric_lin = convert.read_nhdr(metric_file)
    metric_np = metric_lin.cpu().numpy()
    if metric_np.shape[-1] == 6:
        metric_6 = metric_np
    elif metric_np.shape[0] == 6:
        metric_6 = np.transpose(metric_np, (1, 2, 3, 0))
    else:
        raise ValueError(f"Forme inattendue métrique: {metric_np.shape}")
    metric_3x3 = tensors.tens_6_to_tens_3x3(metric_6)  # (X,Y,Z,3,3)

    
    # charger champ de vecteurs (on suppose (3,X,Y,Z))
    vec = convert.read_nhdr(vector_path)
    vector_lin = vec.cpu().to(dtype=torch.float32)

    # charger masque
    mask_t = convert.read_nhdr(mask_path).cpu().to(dtype=torch.float32)

    # mettre métrique en torch pour riemann
    g_torch = torch.from_numpy(metric_3x3).to(dtype=torch.float32)
    # riemann.covariant_derivative_3d attend metric_mat shape [h,w,d,3,3] et vector_lin [3,h,w,d]
    nabla_vv = covariant_derivative_3d(vector_lin, g_torch, mask_t, differential_accuracy=2)
    # norme euclidienne
    e = torch.sqrt((nabla_vv ** 2).sum(dim=0)).cpu().numpy()  # (X,Y,Z)

    X, Y, Z = e.shape
    if slice_index is None:
        slice_index = {"x": X // 2, "y": Y // 2, "z": Z // 2}[axis]

    if axis == "z":
        sl = e[:, :, slice_index]
    elif axis == "y":
        sl = e[:, slice_index, :]
    else:
        sl = e[slice_index, :, :]

    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(sl.T, origin="lower", cmap="magma")
    ax.set_title(r"$\| \nabla_v v \|$ sur une coupe")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_metric_eigs_and_covariant_error(metrics_dir,
                                         vector_path,
                                         mask_path,
                                         axis="z",
                                         slice_index=None):
    """
    Combined figure displaying metric condition number (λ_max / λ_min) and covariant derivative error (||∇v v||)
    side by side for scientific publication.
    
    Parameters:
    -----------
    metrics_dir : str
        Directory containing metric files (metric_final.nhdr or metric_epoch_*.nhdr)
    vector_path : str
        Path to vector field file (independent of metric)
    mask_path : str
        Path to mask file
    axis : str
        Slice axis ('x', 'y', or 'z')
    slice_index : int, optional
        Slice index. If None, uses middle slice.
    """
    # --- 1) Load metric ---
    metric_final = os.path.join(metrics_dir, "metric_final.nhdr")
    if os.path.isfile(metric_final):
        metric_file = metric_final
    else:
        files = sorted(glob.glob(os.path.join(metrics_dir, "metric_epoch_*.nhdr")))
        if not files:
            raise RuntimeError(f"No metric_*.nhdr found in {metrics_dir}")
        metric_file = files[-1]
    
    print(f"[plot_metric_eigs_and_covariant_error] Using metric file: {metric_file}")
    
    metric_lin = convert.read_nhdr(metric_file).cpu().numpy()
    if metric_lin.shape[-1] == 6:
        metric_6 = metric_lin
    elif metric_lin.shape[0] == 6:
        metric_6 = np.transpose(metric_lin, (1, 2, 3, 0))
    else:
        raise ValueError(f"Unexpected metric_lin shape: {metric_lin.shape}")
    
    metric_3x3 = tensors.tens_6_to_tens_3x3(metric_6)  # (X,Y,Z,3,3)
    X, Y, Z = metric_3x3.shape[:3]
    
    mask = convert.read_nhdr(mask_path).cpu().numpy().astype(bool)
    if mask.shape != (X, Y, Z):
        raise ValueError(f"Mask dimensions {mask.shape} != metric {metric_3x3.shape[:3]}")
    
    # --- 2) Select slice ---
    if slice_index is None:
        slice_index = {"x": X // 2, "y": Y // 2, "z": Z // 2}[axis]
    
    if axis == "z":
        sl_g = metric_3x3[:, :, slice_index, :, :]
        sl_mask = mask[:, :, slice_index]
    elif axis == "y":
        sl_g = metric_3x3[:, slice_index, :, :, :]
        sl_mask = mask[:, slice_index, :]
    else:  # 'x'
        sl_g = metric_3x3[slice_index, :, :, :, :]
        sl_mask = mask[slice_index, :, :]
    
    # --- 3) Compute condition number (λ_max / λ_min) ---
    flat = sl_g.reshape(-1, 3, 3)
    eigvals, _ = np.linalg.eigh(flat)
    eigvals = np.sort(eigvals, axis=1)
    lam_min = eigvals[:, 0].reshape(sl_mask.shape)
    lam_max = eigvals[:, 2].reshape(sl_mask.shape)
    cond = lam_max / (lam_min + 1e-8)  # Condition number
    
    # Mask for display
    cond_disp = np.where(sl_mask, cond, np.nan)
    
    # --- 4) Compute covariant derivative error ||∇v v|| ---
    vec = convert.read_nhdr(vector_path)
    vector_lin = vec.cpu().to(dtype=torch.float32)
    mask_t = convert.read_nhdr(mask_path).cpu().to(dtype=torch.float32)
    g_torch = torch.from_numpy(metric_3x3).to(dtype=torch.float32)
    
    nabla_vv = covariant_derivative_3d(vector_lin, g_torch, mask_t, differential_accuracy=2)
    e = torch.sqrt((nabla_vv ** 2).sum(dim=0)).cpu().numpy()  # (X,Y,Z)
    
    if axis == "z":
        sl_error = e[:, :, slice_index]
    elif axis == "y":
        sl_error = e[:, slice_index, :]
    else:
        sl_error = e[slice_index, :, :]
    
    sl_error_disp = np.where(sl_mask, sl_error, np.nan)
    
    # --- 5) Create figure with 2 subplots side by side ---
    axis_labels = {"x": "Sagittal", "y": "Coronal", "z": "Axial"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    
    # Condition number (λ_max / λ_min)
    im0 = axes[0].imshow(cond_disp.T, origin="lower", cmap="magma")
    axes[0].set_title(r"$\lambda_{\max} / \lambda_{\min}$ (Condition Number)", fontsize=14, fontweight="bold")
    axes[0].set_aspect("equal")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Condition number", fontsize=11)
    
    # ||∇v v||
    im1 = axes[1].imshow(sl_error_disp.T, origin="lower", cmap="magma")
    axes[1].set_title(r"$\| \nabla_v v \|$ (Geometric Error)", fontsize=14, fontweight="bold")
    axes[1].set_aspect("equal")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Geometric error", fontsize=11)
    
    # Add axis labels
    if axis == "z":
        for ax in axes:
            ax.set_xlabel("X (voxels)", fontsize=11)
        axes[0].set_ylabel("Y (voxels)", fontsize=11)
    elif axis == "y":
        for ax in axes:
            ax.set_xlabel("X (voxels)", fontsize=11)
        axes[0].set_ylabel("Z (voxels)", fontsize=11)
    else:
        for ax in axes:
            ax.set_xlabel("Y (voxels)", fontsize=11)
        axes[0].set_ylabel("Z (voxels)", fontsize=11)
    
    # Overall title
    
    
    # Reduce space between subplots - wspace controls horizontal spacing (0 = no space)
    plt.subplots_adjust(wspace=-0.4, left=0.05, right=0.95, top=0.92, bottom=0.1)
    plt.show()




def compute_vector_field_metric(vector_field_xyz, mask):
    """
    Compute a simple metric aligned with the vector field.
    Creates a metric that has a large eigenvalue in the direction of the vector field.
    
    Parameters:
    -----------
    vector_field_xyz : array (X, Y, Z, 3)
        Vector field
    mask : array (X, Y, Z)
        Binary mask
    
    Returns:
    --------
    metric_3x3 : array (X, Y, Z, 3, 3)
        Metric tensor aligned with vector field
    """
    X, Y, Z = mask.shape
    
    # Normalize vector field
    v_norm = np.linalg.norm(vector_field_xyz, axis=3, keepdims=True)
    v_norm = np.where(v_norm > 1e-6, v_norm, 1.0)
    v_normalized = vector_field_xyz / v_norm
    
    # Create metric: large eigenvalue in direction of vector field
    # g = I + alpha * v ⊗ v, where alpha controls anisotropy
    alpha = 10.0  # Anisotropy factor
    metric_3x3 = np.zeros((X, Y, Z, 3, 3))
    
    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                if mask[i, j, k]:
                    v = v_normalized[i, j, k, :]
                    # Outer product v ⊗ v
                    vvT = np.outer(v, v)
                    # Metric: I + alpha * v ⊗ v
                    metric_3x3[i, j, k] = np.eye(3) + alpha * vvT
                else:
                    metric_3x3[i, j, k] = np.eye(3)
    
    return metric_3x3


def compare_baseline_metrics_on_slice(vector_path,
                                      checkpoint_dirs,
                                      subject_id,
                                      mask_path,
                                      seed_start_point=0,
                                      axis="z",
                                      slice_index=None,
                                      delta_t=0.15,
                                      iter_num=8000,
                                      stop_angle=30,
                                      quiver_step=1,
                                      arrow_len=1.5,
                                      zoom=True,
                                      zoom_size=10,
                                      max_length=None):
    """
    Compare geodesics computed from different metrics projected on a 2D slice:
      - Euclidean metric (identity)
      - Vector field-based metric (aligned with vector field)
      - CNN-learned metrics from different checkpoint directories (different architectures)
    
    Parameters:
    -----------
    vector_path : str
        Path to vector field file (used for geodesic computation and vector field-based metric)
    checkpoint_dirs : list of str
        List of checkpoint directory names (e.g., ["Checkpoints_10_10_10", "Checkpoints_40_30_40"])
        Architecture is extracted from directory name
    subject_id : str
        Subject ID (e.g., "111312")
    mask_path : str
        Path to mask file
    seed_start_point : int
        Seed for selecting start point
    axis : str
        Slice axis ('x', 'y', or 'z')
    slice_index : int, optional
        Slice index. If None, uses middle slice.
    delta_t : float
        Time step for geodesic integration
    iter_num : int
        Number of iterations for geodesic computation
    stop_angle : float
        Stop angle for geodesic computation
    quiver_step : int
        Step size for quiver plot decimation
    arrow_len : float
        Arrow length for quiver plot
    zoom : bool
        Whether to zoom around start point
    zoom_size : float
        Zoom size around start point
    max_length : float, optional
        Maximum distance from start point for geodesics
    """
    # Load mask
    mask = convert.read_nhdr(mask_path).cpu().numpy().astype(bool)
    X, Y, Z = mask.shape
    
    # Load vector field
    vec_torch = convert.read_nhdr(vector_path)
    if hasattr(vec_torch, 'cpu'):
        vector_field = vec_torch.cpu().numpy()
    else:
        vector_field = vec_torch.numpy()
    
    # Convert to both formats needed
    if vector_field.shape[0] == 3:
        vector_field_xyz = np.transpose(vector_field, (1, 2, 3, 0))  # (X, Y, Z, 3)
        vector_lin = vector_field  # (3, X, Y, Z)
    elif vector_field.shape[-1] == 3:
        vector_field_xyz = vector_field  # (X, Y, Z, 3)
        vector_lin = np.transpose(vector_field, (3, 0, 1, 2))  # (3, X, Y, Z)
    else:
        raise ValueError(f"Unexpected vector field shape: {vector_field.shape}. Expected (3, X, Y, Z) or (X, Y, Z, 3)")
    
    # Select start point (center of high region, like in plot_geodesics_on_slice_from_checkpoints)
    start_point, slice_index_eff = select_oriented_point_on_slice(
        mask,
        vector_field_xyz,
        axis=axis,
        slice_index=slice_index,
        seed=seed_start_point,
        margin=2,
        min_inplane=0.0,
        use_center_high_region=True
    )
    sx, sy, sz = start_point
    print(f"Start point = {start_point}, axis={axis}, slice_index={slice_index_eff}")
    
    # Create figure
    axis_labels = {"x": "Sagittal", "y": "Coronal", "z": "Axial"}
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # Background (mask)
    if axis == "z":
        m = mask[:, :, slice_index_eff]
    elif axis == "y":
        m = mask[:, slice_index_eff, :]
    else:
        m = mask[slice_index_eff, :, :]
    ax.imshow(m.T, origin="lower", cmap="gray", alpha=0.2)
    
    # Prepare metrics to compare
    metrics_to_plot = []
    
    # 1) Euclidean metric (identity)
    id_g = np.zeros((X, Y, Z, 3, 3))
    id_g[..., 0, 0] = 1.0
    id_g[..., 1, 1] = 1.0
    id_g[..., 2, 2] = 1.0
    id_g_inv = id_g  # Identity is its own inverse
    id_6 = tensors.tens_3x3_to_tens_6(id_g_inv)
    id_tensor_lin = np.transpose(id_6, (3, 0, 1, 2))  # (6, X, Y, Z)
    metrics_to_plot.append(("Euclidean", id_tensor_lin))
    
    # 2) Vector field-based metric
    vf_metric = compute_vector_field_metric(vector_field_xyz, mask)
    vf_g_inv = np.linalg.inv(vf_metric)
    vf_6 = tensors.tens_3x3_to_tens_6(vf_g_inv)
    vf_tensor_lin = np.transpose(vf_6, (3, 0, 1, 2))  # (6, X, Y, Z)
    metrics_to_plot.append(("Vector field-based", vf_tensor_lin))
    
    # 3) CNN-learned metrics from checkpoints
    for ckpt_dir in checkpoint_dirs:
        # Extract architecture from directory name (e.g., "Checkpoints_40_30_40" -> "40_30_40" -> "[40,30,40]")
        arch_name = ckpt_dir.replace("Checkpoints_", "").replace("checkpoints_", "").replace("../", "")
        # Convert "40_30_40" to "[40,30,40]"
        arch_list = arch_name.split("_")
        arch_formatted = "[" + ",".join(arch_list) + "]"
        
        metric_path = os.path.join(ckpt_dir, subject_id, "metrics", "metric_final.nhdr")
        if not os.path.isfile(metric_path):
            print(f"Warning: {metric_path} not found, skipping...")
            continue
        
        cnn_lin = convert.read_nhdr(metric_path).cpu().numpy()
        if cnn_lin.shape[-1] == 6:
            cnn_6 = cnn_lin
        elif cnn_lin.shape[0] == 6:
            cnn_6 = np.transpose(cnn_lin, (1, 2, 3, 0))
        else:
            print(f"Warning: Unexpected shape for {metric_path}, skipping...")
            continue
        
        cnn_g = tensors.tens_6_to_tens_3x3(cnn_6)
        cnn_g_inv = np.linalg.inv(cnn_g)
        cnn_g_inv_6 = tensors.tens_3x3_to_tens_6(cnn_g_inv)
        cnn_tensor_lin = np.transpose(cnn_g_inv_6, (3, 0, 1, 2))  # (6, X, Y, Z)
        metrics_to_plot.append((f"Modèle CEDNN {arch_formatted}", cnn_tensor_lin))
    
    # Colors for different metrics - use viridis-like colors for consistency
    n_metrics = len(metrics_to_plot)
    colors = []
    # Fixed colors for baseline metrics
    colors.append('#440154')  # Dark purple for Euclidean
    colors.append('#31688e')  # Blue for Vector field-based
    # Use viridis for CNN models
    if n_metrics > 2:
        cnn_colors = plt.cm.viridis(np.linspace(0.3, 0.9, n_metrics - 2))
        colors.extend(cnn_colors)
    
    # Compute and plot geodesics for each metric
    for idx, (metric_name, tensor_lin_for_geo) in enumerate(metrics_to_plot):
        print(f"\n=== Computing geodesic for {metric_name} ===")
        
        gx, gy, gz = geodesicpath_3d(
            tensor_lin=tensor_lin_for_geo,
            vector_lin=vector_lin,
            mask_image=mask,
            start_coordinate=[sx, sy, sz],
            initial_velocity=None,
            delta_t=delta_t,
            iter_num=iter_num,
            stop_angle=stop_angle,
            both_directions=True
        )
        
        # Truncate if max_length is specified
        if max_length is not None:
            gx, gy, gz = truncate_geodesic_by_length(gx, gy, gz, start_point, max_length)
        
        # Project onto slice
        if axis == "z":
            proj_x = gx
            proj_y = gy
        elif axis == "y":
            proj_x = gx
            proj_y = gz
        else:  # 'x'
            proj_x = gy
            proj_y = gz
        
        ax.plot(proj_x, proj_y, color=colors[idx], linewidth=2, label=metric_name, alpha=0.8)
    
    # Vector field quiver plot (only once)
    if axis == "z":
        vf = vector_field_xyz[:, :, slice_index_eff, :]
        U = vf[::quiver_step, ::quiver_step, 0] * arrow_len
        V = vf[::quiver_step, ::quiver_step, 1] * arrow_len
        Xq, Yq = np.meshgrid(
            np.arange(0, X, quiver_step),
            np.arange(0, Y, quiver_step),
            indexing='ij'
        )
    elif axis == "y":
        vf = vector_field_xyz[:, slice_index_eff, :, :]
        U = vf[::quiver_step, ::quiver_step, 0] * arrow_len
        V = vf[::quiver_step, ::quiver_step, 2] * arrow_len
        Xq, Yq = np.meshgrid(
            np.arange(0, X, quiver_step),
            np.arange(0, Z, quiver_step),
            indexing='ij'
        )
    else:  # 'x'
        vf = vector_field_xyz[slice_index_eff, :, :, :]
        U = vf[::quiver_step, ::quiver_step, 1] * arrow_len
        V = vf[::quiver_step, ::quiver_step, 2] * arrow_len
        Xq, Yq = np.meshgrid(
            np.arange(0, vf.shape[0], quiver_step),
            np.arange(0, vf.shape[1], quiver_step),
            indexing='ij'
        )
    
    ax.quiver(Xq, Yq, U, V, color="gray", alpha=0.6, angles="xy", 
              scale_units="xy", scale=1.0, linewidths=0.5)
    
    # Mark start point
    if axis == "z":
        sx2, sy2 = sx, sy
    elif axis == "y":
        sx2, sy2 = sx, sz
    else:
        sx2, sy2 = sy, sz
    
    ax.scatter(sx2, sy2, color="black", s=100, marker="*", zorder=15,
               edgecolors="white", linewidths=1)
    
    # Labels and title
    ax.set_aspect("equal")
    ax.set_title(f"Geodesic Tractography: Model Comparison\n{axis_labels.get(axis, axis)} slice (index {slice_index_eff})",
                 fontsize=14, fontweight="bold", pad=15)
    
    if axis == "z":
        ax.set_xlabel("X (voxels)", fontsize=11)
        ax.set_ylabel("Y (voxels)", fontsize=11)
    elif axis == "y":
        ax.set_xlabel("X (voxels)", fontsize=11)
        ax.set_ylabel("Z (voxels)", fontsize=11)
    else:
        ax.set_xlabel("Y (voxels)", fontsize=11)
        ax.set_ylabel("Z (voxels)", fontsize=11)
    
    # Legend below figure - only show metric names
    # Calculate number of columns based on number of metrics
    n_metrics = len(metrics_to_plot)
    ncol = min(n_metrics, 4)  # Max 4 columns
    ax.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', ncol=ncol, 
              fontsize=10, framealpha=0.95, columnspacing=1.0, handlelength=1.5)
    
    # Zoom option
    if zoom:
        cx, cy = sx2, sy2
        ax.set_xlim(cx - zoom_size, cx + zoom_size)
        ax.set_ylim(cy - zoom_size, cy + zoom_size)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.show()


def debug_TDir(metrics_dir,
               mask_path,
               axis="z",
               slice_index=None,
               quiver_step=3,
               arrow_len=2.0):
    """
    Compare visuellement deux champs de directions principales :
    - TDir brut = eigenvecteur principal de g (metric_final)
    - TDir_solve = TDir renvoyé par solve_3d (après traitement géométrique)

    Affiche deux quiver plots côte à côte sur une même coupe.
    """

    # --- charger masque ---
    mask = convert.read_nhdr(mask_path).cpu().numpy().astype(bool)
    X, Y, Z = mask.shape

    # --- choisir le fichier métrique ---
    metric_final = os.path.join(metrics_dir, "metric_final.nhdr")
    if os.path.isfile(metric_final):
        metric_file = metric_final
    else:
        epoch_files = sorted(glob.glob(os.path.join(metrics_dir, "metric_epoch_*.nhdr")))
        if not epoch_files:
            raise RuntimeError(f"Aucun metric_*.nhdr trouvé dans {metrics_dir}")
        metric_file = epoch_files[-1]

    print(f"[debug_TDir] Fichier métrique utilisé : {metric_file}")

    # --- charger métrique linéarisée ---
    metric_lin = convert.read_nhdr(metric_file).cpu().numpy()
    if metric_lin.shape[-1] == 6:
        metric_6 = metric_lin  # (X,Y,Z,6)
    elif metric_lin.shape[0] == 6:
        metric_6 = np.transpose(metric_lin, (1, 2, 3, 0))  # (X,Y,Z,6)
    else:
        raise ValueError(f"Forme inattendue pour metric_lin: {metric_lin.shape}")

    metric_3x3 = tensors.tens_6_to_tens_3x3(metric_6)  # (X,Y,Z,3,3)

    # --- choisir la slice ---
    if slice_index is None:
        if axis == "z":
            slice_index = Z // 2
        elif axis == "y":
            slice_index = Y // 2
        elif axis == "x":
            slice_index = X // 2
        else:
            raise ValueError("axis doit être 'x','y' ou 'z'")

    print(f"[debug_TDir] axis={axis}, slice_index={slice_index}")

    # --- TDir brut depuis metric_final (eigenvecteur principal de g) ---
    metric_flat = metric_3x3.reshape(-1, 3, 3)
    eigvals, eigvecs = np.linalg.eigh(metric_flat)  # (N,3), (N,3,3)
    # vecteur propre associé à la plus grande valeur propre
    idx_max = np.argmax(eigvals, axis=1)           # (N,)
    TDir_raw = np.take_along_axis(
        eigvecs, idx_max[:, None, None], axis=2
    )[..., 0]                                      # (N,3)
    TDir_raw = TDir_raw.reshape(X, Y, Z, 3)

    # --- TDir_solve via solve_3d (en passant g^{-1}) ---
    metric_3x3_inv = np.linalg.inv(metric_3x3)            # (X,Y,Z,3,3)
    in_tens = tensors.tens_3x3_to_tens_6(metric_3x3_inv)  # (X,Y,Z,6)
    _, scaled_tensors, _, _, inter = solve_3d(
        in_tens=in_tens,
        in_mask=mask.astype(np.uint8),
        max_iters=50,
        save_intermediate_results=True,
        sigma=None,
    )
    TDir_solve = inter["TDir"]  # (X,Y,Z,3)

    # --- extraire slice & préparer quiver ---
    if axis == "z":
        sl_mask = mask[:, :, slice_index]
        sl_raw = TDir_raw[:, :, slice_index, :]     # (X,Y,3)
        sl_solve = TDir_solve[:, :, slice_index, :]
        U_raw = sl_raw[::quiver_step, ::quiver_step, 0] * arrow_len
        V_raw = sl_raw[::quiver_step, ::quiver_step, 1] * arrow_len
        U_sol = sl_solve[::quiver_step, ::quiver_step, 0] * arrow_len
        V_sol = sl_solve[::quiver_step, ::quiver_step, 1] * arrow_len
        Xq, Yq = np.meshgrid(
            np.arange(0, X, quiver_step),
            np.arange(0, Y, quiver_step),
            indexing="ij",
        )
    elif axis == "y":
        sl_mask = mask[:, slice_index, :]
        sl_raw = TDir_raw[:, slice_index, :, :]     # (X,Z,3)
        sl_solve = TDir_solve[:, slice_index, :, :]
        U_raw = sl_raw[::quiver_step, ::quiver_step, 0] * arrow_len
        V_raw = sl_raw[::quiver_step, ::quiver_step, 2] * arrow_len
        U_sol = sl_solve[::quiver_step, ::quiver_step, 0] * arrow_len
        V_sol = sl_solve[::quiver_step, ::quiver_step, 2] * arrow_len
        Xq, Yq = np.meshgrid(
            np.arange(0, X, quiver_step),
            np.arange(0, Z, quiver_step),
            indexing="ij",
        )
    else:  # 'x'
        sl_mask = mask[slice_index, :, :]
        sl_raw = TDir_raw[slice_index, :, :, :]     # (Y,Z,3)
        sl_solve = TDir_solve[slice_index, :, :, :]
        U_raw = sl_raw[::quiver_step, ::quiver_step, 1] * arrow_len
        V_raw = sl_raw[::quiver_step, ::quiver_step, 2] * arrow_len
        U_sol = sl_solve[::quiver_step, ::quiver_step, 1] * arrow_len
        V_sol = sl_solve[::quiver_step, ::quiver_step, 2] * arrow_len
        Xq, Yq = np.meshgrid(
            np.arange(0, sl_mask.shape[0], quiver_step),
            np.arange(0, sl_mask.shape[1], quiver_step),
            indexing="ij",
        )

    # --- stats rapides pour debug ---
    print("\n[debug_TDir] Stats TDir_raw (sur tout le volume) :")
    print("  mean:", TDir_raw.mean(axis=(0, 1, 2)))
    print("  std :", TDir_raw.std(axis=(0, 1, 2)))
    print("[debug_TDir] Stats TDir_solve (sur tout le volume) :")
    print("  mean:", TDir_solve.mean(axis=(0, 1, 2)))
    print("  std :", TDir_solve.std(axis=(0, 1, 2)))

    # --- figures ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # fond masque
    axes[0].imshow(sl_mask.T, origin="lower", cmap="gray", alpha=0.2)
    axes[1].imshow(sl_mask.T, origin="lower", cmap="gray", alpha=0.2)

    axes[0].quiver(
        Xq, Yq, U_raw, V_raw,
        color="red", alpha=0.7,
        angles="xy", scale_units="xy", scale=1.0,
        linewidths=0.5,
    )
    axes[1].quiver(
        Xq, Yq, U_sol, V_sol,
        color="blue", alpha=0.7,
        angles="xy", scale_units="xy", scale=1.0,
        linewidths=0.5,
    )

    axes[0].set_title("TDir brut (eigenvecteur g)")
    axes[1].set_title("TDir solve_3d (inter['TDir'])")

    for ax in axes:
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()


# ============================================================================
# FUNCTIONS FOR SCIENTIFIC ARTICLE FIGURES A TESTER
# ============================================================================

def compare_architectures_performance(checkpoint_dirs, 
                                     model_names,
                                     vector_path,
                                     mask_path,
                                     subject_id="111312",
                                     save_path=None):
    """
    Compare quantitativement différentes architectures (CEDNN, SIREN, RCNF, etc.)
    
    Métriques calculées :
    - Loss finale
    - Nombre de paramètres (depuis training_config.json si disponible)
    - Distance moyenne des géodésiques au ground truth
    - Condition number moyen
    - Erreur covariante moyenne
    
    Parameters:
    -----------
    checkpoint_dirs : list of str
        List of checkpoint directory paths (e.g., ["../Checkpoints_10_10_10", ...])
    model_names : list of str
        List of model names corresponding to checkpoint_dirs
    vector_path : str
        Path to vector field file (ground truth)
    mask_path : str
        Path to mask file
    subject_id : str
        Subject ID
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    results_dict : dict
        Dictionary with all computed metrics
    """
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        pass
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})
    
    results = {
        'model': [],
        'final_loss': [],
        'num_params': [],
        'mean_geodesic_distance': [],
        'mean_condition_number': [],
        'mean_covariant_error': []
    }
    
    # Load ground truth vector field
    vec_torch = convert.read_nhdr(vector_path).cpu()
    vector_field = vec_torch.numpy().astype(float)
    vector_field_xyz = np.transpose(vector_field, (1, 2, 3, 0))
    vector_lin = vec_torch.to(dtype=torch.float32)
    
    # Load mask
    mask = convert.read_nhdr(mask_path).cpu().numpy().astype(bool)
    mask_t = convert.read_nhdr(mask_path).cpu().to(dtype=torch.float32)
    
    # Select start point
    start_point = select_oriented_point_on_slice(mask, vector_field_xyz, seed=0)
    sx, sy, sz = start_point
    
    # Compute ground truth geodesic
    print("Computing ground truth geodesic...")
    gx_gt, gy_gt, gz_gt = integrate_vector_field(
        vector_field_xyz, start_point, mask, delta_t=0.15, iter_num=8000,
        both_directions=True, max_length=10
    )
    
    for checkpoint_dir, model_name in zip(checkpoint_dirs, model_names):
        print(f"\n=== Processing {model_name} ===")
        subj_dir = os.path.join(checkpoint_dir, subject_id)
        metrics_dir = os.path.join(subj_dir, "metrics")
        loss_csv = os.path.join(subj_dir, "loss.csv")
        config_file = os.path.join(subj_dir, "training_config.json")
        
        # Final loss
        if os.path.isfile(loss_csv):
            df = pd.read_csv(loss_csv)
            final_loss = df['loss'].iloc[-1]
        else:
            final_loss = np.nan
        
        # Number of parameters (try to get from config or estimate)
        num_params = np.nan
        if os.path.isfile(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Try to extract from config if available
                if 'num_params' in config:
                    num_params = config['num_params']
        
        # Load final metric
        metric_final = os.path.join(metrics_dir, "metric_final.nhdr")
        if not os.path.isfile(metric_final):
            print(f"Warning: {metric_final} not found, skipping {model_name}")
            continue
        
        metric_lin = convert.read_nhdr(metric_final).cpu().numpy()
        if metric_lin.shape[-1] == 6:
            metric_6 = metric_lin
        elif metric_lin.shape[0] == 6:
            metric_6 = np.transpose(metric_lin, (1, 2, 3, 0))
        else:
            continue
        
        metric_3x3 = tensors.tens_6_to_tens_3x3(metric_6)
        metric_lin_torch = torch.from_numpy(metric_6).permute(3, 0, 1, 2)
        
        # Compute geodesic
        print(f"  Computing geodesic for {model_name}...")
        gx, gy, gz = geodesicpath_3d(
            tensor_lin=metric_lin_torch,
            vector_lin=vector_lin,
            mask_image=mask,
            start_coordinate=[sx, sy, sz],
            initial_velocity=None,
            delta_t=0.15,
            iter_num=8000,
            stop_angle=30,
            both_directions=True
        )
        
        # Truncate to same length as ground truth
        if len(gx) > len(gx_gt):
            gx, gy, gz = gx[:len(gx_gt)], gy[:len(gx_gt)], gz[:len(gx_gt)]
        elif len(gx_gt) > len(gx):
            gx_gt_short = gx_gt[:len(gx)]
            gy_gt_short = gy_gt[:len(gy)]
            gz_gt_short = gz_gt[:len(gz)]
        else:
            gx_gt_short, gy_gt_short, gz_gt_short = gx_gt, gy_gt, gz_gt
        
        # Distance to ground truth (Hausdorff-like distance)
        geodesic_points = np.column_stack([gx, gy, gz])
        gt_points = np.column_stack([gx_gt_short, gy_gt_short, gz_gt_short])
        distances = cdist(geodesic_points, gt_points)
        mean_geod_dist = np.mean(np.min(distances, axis=1))
        
        # Condition number
        flat = metric_3x3[mask].reshape(-1, 3, 3)
        eigvals, _ = np.linalg.eigh(flat)
        eigvals = np.sort(eigvals, axis=1)
        lam_min = eigvals[:, 0]
        lam_max = eigvals[:, 2]
        cond = lam_max / (lam_min + 1e-8)
        mean_cond = np.mean(cond)
        
        # Covariant error
        g_torch = torch.from_numpy(metric_3x3).to(dtype=torch.float32)
        nabla_vv = covariant_derivative_3d(vector_lin, g_torch, mask_t, differential_accuracy=2)
        e = torch.sqrt((nabla_vv ** 2).sum(dim=0)).cpu().numpy()
        mean_cov_error = np.mean(e[mask])
        
        results['model'].append(model_name)
        results['final_loss'].append(final_loss)
        results['num_params'].append(num_params)
        results['mean_geodesic_distance'].append(mean_geod_dist)
        results['mean_condition_number'].append(mean_cond)
        results['mean_covariant_error'].append(mean_cov_error)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Architecture Performance Comparison', fontsize=16, fontweight='bold')
    
    models = results['model']
    x_pos = np.arange(len(models))
    
    # Plot 1: Final Loss
    axes[0, 0].bar(x_pos, results['final_loss'], color='steelblue', alpha=0.7)
    axes[0, 0].set_ylabel('Final Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Number of Parameters
    axes[0, 1].bar(x_pos, [p/1e6 if not np.isnan(p) else 0 for p in results['num_params']], 
                   color='coral', alpha=0.7)
    axes[0, 1].set_ylabel('Parameters (M)', fontsize=12)
    axes[0, 1].set_title('Model Complexity', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Geodesic Distance
    axes[0, 2].bar(x_pos, results['mean_geodesic_distance'], color='mediumseagreen', alpha=0.7)
    axes[0, 2].set_ylabel('Mean Distance to GT (voxels)', fontsize=12)
    axes[0, 2].set_title('Geodesic Accuracy', fontsize=13, fontweight='bold')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # Plot 4: Condition Number
    axes[1, 0].bar(x_pos, results['mean_condition_number'], color='gold', alpha=0.7)
    axes[1, 0].set_ylabel('Mean Condition Number', fontsize=12)
    axes[1, 0].set_title('Metric Conditioning', fontsize=13, fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 5: Covariant Error
    axes[1, 1].bar(x_pos, results['mean_covariant_error'], color='orchid', alpha=0.7)
    axes[1, 1].set_ylabel('Mean Covariant Error', fontsize=12)
    axes[1, 1].set_title('Geometric Consistency', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Plot 6: Loss vs Parameters (scatter)
    valid_idx = [i for i, p in enumerate(results['num_params']) if not np.isnan(p)]
    if len(valid_idx) > 0:
        axes[1, 2].scatter([results['num_params'][i]/1e6 for i in valid_idx],
                          [results['final_loss'][i] for i in valid_idx],
                          s=100, alpha=0.6, c='red')
        for i in valid_idx:
            axes[1, 2].annotate(models[i], 
                               (results['num_params'][i]/1e6, results['final_loss'][i]),
                               fontsize=9, alpha=0.8)
        axes[1, 2].set_xlabel('Parameters (M)', fontsize=12)
        axes[1, 2].set_ylabel('Final Loss', fontsize=12)
        axes[1, 2].set_title('Efficiency Trade-off', fontsize=13, fontweight='bold')
        axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return results


def plot_complexity_vs_performance(checkpoint_dirs,
                                   model_configs,
                                   vector_path,
                                   mask_path,
                                   subject_id="111312",
                                   save_path=None):
    """
    Scatter plot : nombre de paramètres vs performance (loss finale)
    
    Montre le trade-off complexité/performance
    Permet d'identifier l'architecture optimale
    
    Parameters:
    -----------
    checkpoint_dirs : list of str
        List of checkpoint directory paths
    model_configs : list of tuples
        List of (model_name, num_params, final_loss) tuples
        If None, will try to extract from checkpoint directories
    vector_path : str
        Path to vector field file
    mask_path : str
        Path to mask file
    subject_id : str
        Subject ID
    save_path : str, optional
        Path to save the figure
    """
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        pass
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    
    if model_configs is None:
        # Extract from checkpoint directories
        model_configs = []
        for checkpoint_dir in checkpoint_dirs:
            subj_dir = os.path.join(checkpoint_dir, subject_id)
            loss_csv = os.path.join(subj_dir, "loss.csv")
            config_file = os.path.join(subj_dir, "training_config.json")
            
            model_name = os.path.basename(checkpoint_dir)
            final_loss = np.nan
            num_params = np.nan
            
            if os.path.isfile(loss_csv):
                df = pd.read_csv(loss_csv)
                final_loss = df['loss'].iloc[-1]
            
            if os.path.isfile(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    if 'num_params' in config:
                        num_params = config['num_params']
            
            model_configs.append((model_name, num_params, final_loss))
    
    # Filter out invalid entries
    valid_configs = [(name, params, loss) for name, params, loss in model_configs 
                     if not (np.isnan(params) or np.isnan(loss))]
    
    if len(valid_configs) == 0:
        print("Warning: No valid configurations found")
        return
    
    names, params, losses = zip(*valid_configs)
    params = np.array(params) / 1e6  # Convert to millions
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot
    scatter = ax.scatter(params, losses, s=150, alpha=0.7, c=params, 
                        cmap='viridis', edgecolors='black', linewidths=1.5)
    
    # Annotate points
    for name, p, l in zip(names, params, losses):
        ax.annotate(name, (p, l), fontsize=10, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add Pareto frontier (lower is better for both)
    if len(valid_configs) > 1:
        # Find Pareto optimal points
        points = np.column_stack([params, losses])
        pareto_mask = np.ones(len(points), dtype=bool)
        for i, point in enumerate(points):
            for j, other in enumerate(points):
                if i != j:
                    if (other[0] <= point[0] and other[1] <= point[1] and 
                        (other[0] < point[0] or other[1] < point[1])):
                        pareto_mask[i] = False
                        break
        
        pareto_points = points[pareto_mask]
        if len(pareto_points) > 1:
            # Sort by parameters
            sort_idx = np.argsort(pareto_points[:, 0])
            pareto_points = pareto_points[sort_idx]
            ax.plot(pareto_points[:, 0], pareto_points[:, 1], 'r--', 
                   linewidth=2, alpha=0.6, label='Pareto Frontier')
    
    ax.set_xlabel('Number of Parameters (M)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Final Loss', fontsize=13, fontweight='bold')
    ax.set_title('Complexity vs Performance Trade-off', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.colorbar(scatter, ax=ax, label='Parameters (M)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def compute_geodesic_quality_metrics(metric_file,
                                     vector_path,
                                     mask_path,
                                     num_seed_points=10,
                                     delta_t=0.15,
                                     iter_num=8000):
    """
    Calcule des métriques quantitatives de qualité géométrique
    
    Métriques :
    1. Distance moyenne des géodésiques au ground truth (vector field)
    2. Angle moyen entre direction principale et vector field
    3. Longueur des géodésiques vs longueur ground truth
    4. Courbure moyenne des géodésiques
    
    Parameters:
    -----------
    metric_file : str
        Path to metric file
    vector_path : str
        Path to vector field file (ground truth)
    mask_path : str
        Path to mask file
    num_seed_points : int
        Number of seed points to test
    delta_t : float
        Time step for geodesic integration
    iter_num : int
        Number of iterations
    
    Returns:
    --------
    metrics_dict : dict
        Dictionary with all computed metrics
    """
    # Load data
    metric_lin = convert.read_nhdr(metric_file).cpu().numpy()
    if metric_lin.shape[-1] == 6:
        metric_6 = metric_lin
    elif metric_lin.shape[0] == 6:
        metric_6 = np.transpose(metric_lin, (1, 2, 3, 0))
    else:
        raise ValueError(f"Unexpected metric shape: {metric_lin.shape}")
    
    metric_3x3 = tensors.tens_6_to_tens_3x3(metric_6)
    metric_lin_torch = torch.from_numpy(metric_6).permute(3, 0, 1, 2)
    
    vec_torch = convert.read_nhdr(vector_path).cpu()
    vector_field = vec_torch.numpy().astype(float)
    vector_field_xyz = np.transpose(vector_field, (1, 2, 3, 0))
    vector_lin = vec_torch.to(dtype=torch.float32)
    
    mask = convert.read_nhdr(mask_path).cpu().numpy().astype(bool)
    
    # Select multiple seed points
    seed_points = []
    for seed in range(num_seed_points):
        try:
            pt = select_oriented_point_on_slice(mask, vector_field_xyz, seed=seed)
            seed_points.append(pt)
        except:
            continue
    
    if len(seed_points) == 0:
        raise ValueError("No valid seed points found")
    
    all_distances = []
    all_angles = []
    all_length_ratios = []
    all_curvatures = []
    
    for seed_point in seed_points:
        sx, sy, sz = seed_point
        
        # Ground truth geodesic
        gx_gt, gy_gt, gz_gt = integrate_vector_field(
            vector_field_xyz, seed_point, mask, delta_t=delta_t, 
            iter_num=iter_num, both_directions=True, max_length=10
        )
        
        # CNN-learned geodesic
        gx, gy, gz = geodesicpath_3d(
            tensor_lin=metric_lin_torch,
            vector_lin=vector_lin,
            mask_image=mask,
            start_coordinate=[sx, sy, sz],
            initial_velocity=None,
            delta_t=delta_t,
            iter_num=iter_num,
            stop_angle=30,
            both_directions=True
        )
        
        # Align lengths
        min_len = min(len(gx), len(gx_gt))
        gx, gy, gz = gx[:min_len], gy[:min_len], gz[:min_len]
        gx_gt, gy_gt, gz_gt = gx_gt[:min_len], gy_gt[:min_len], gz_gt[:min_len]
        
        # 1. Distance to ground truth
        geodesic_points = np.column_stack([gx, gy, gz])
        gt_points = np.column_stack([gx_gt, gy_gt, gz_gt])
        distances = cdist(geodesic_points, gt_points)
        mean_dist = np.mean(np.min(distances, axis=1))
        all_distances.append(mean_dist)
        
        # 2. Angle between principal direction and vector field
        # Get principal eigenvector at each point
        angles = []
        for i in range(min(len(gx), len(gx_gt))):
            x, y, z = int(gx[i]), int(gy[i]), int(gz[i])
            if (0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and 
                0 <= z < mask.shape[2] and mask[x, y, z]):
                g_mat = metric_3x3[x, y, z]
                eigvals, eigvecs = np.linalg.eigh(g_mat)
                principal_dir = eigvecs[:, np.argmax(eigvals)]
                v_field = vector_field_xyz[x, y, z]
                v_field_norm = v_field / (np.linalg.norm(v_field) + 1e-8)
                principal_dir_norm = principal_dir / (np.linalg.norm(principal_dir) + 1e-8)
                cos_angle = np.clip(np.dot(principal_dir_norm, v_field_norm), -1, 1)
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
        if len(angles) > 0:
            all_angles.append(np.mean(angles))
        
        # 3. Length ratio
        def path_length(x, y, z):
            diffs = np.diff(np.column_stack([x, y, z]), axis=0)
            return np.sum(np.linalg.norm(diffs, axis=1))
        
        len_gt = path_length(gx_gt, gy_gt, gz_gt)
        len_pred = path_length(gx, gy, gz)
        if len_gt > 0:
            all_length_ratios.append(len_pred / len_gt)
        
        # 4. Curvature (simplified: rate of change of direction)
        if len(gx) > 2:
            directions = np.diff(np.column_stack([gx, gy, gz]), axis=0)
            directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
            curvature = np.mean(np.linalg.norm(np.diff(directions, axis=0), axis=1))
            all_curvatures.append(curvature)
    
    metrics_dict = {
        'mean_distance_to_gt': np.mean(all_distances),
        'std_distance_to_gt': np.std(all_distances),
        'mean_angle': np.mean(all_angles) if len(all_angles) > 0 else np.nan,
        'std_angle': np.std(all_angles) if len(all_angles) > 0 else np.nan,
        'mean_length_ratio': np.mean(all_length_ratios),
        'std_length_ratio': np.std(all_length_ratios),
        'mean_curvature': np.mean(all_curvatures) if len(all_curvatures) > 0 else np.nan,
        'std_curvature': np.std(all_curvatures) if len(all_curvatures) > 0 else np.nan,
    }
    
    return metrics_dict


def plot_convergence_comparison(checkpoint_dirs_dict,
                               subject_id="111312",
                               save_path=None):
    """
    Compare les courbes de loss de différentes architectures
    
    Parameters:
    -----------
    checkpoint_dirs_dict : dict
        Dictionary mapping model names to checkpoint directory paths
        e.g., {"CEDNN [40,30,40]": "../Checkpoints_40_30_40", ...}
    subject_id : str
        Subject ID
    save_path : str, optional
        Path to save the figure
    """
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        pass
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(checkpoint_dirs_dict)))
    
    for (model_name, checkpoint_dir), color in zip(checkpoint_dirs_dict.items(), colors):
        subj_dir = os.path.join(checkpoint_dir, subject_id)
        loss_csv = os.path.join(subj_dir, "loss.csv")
        
        if not os.path.isfile(loss_csv):
            print(f"Warning: {loss_csv} not found, skipping {model_name}")
            continue
        
        df = pd.read_csv(loss_csv)
        epochs = df['epoch'].to_numpy()
        losses = df['loss'].to_numpy()
        
        # Normalize epochs to [0, 1] for comparison
        if len(epochs) > 1:
            epochs_norm = epochs / epochs.max()
        else:
            epochs_norm = epochs
        
        ax.plot(epochs_norm, losses, label=model_name, color=color, 
               linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Normalized Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('Training Convergence Comparison', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def generate_results_table(checkpoint_dirs_dict,
                          vector_path,
                          mask_path,
                          subject_id="111312",
                          save_path_csv=None,
                          save_path_fig=None):
    """
    Génère un tableau LaTeX/CSV avec toutes les métriques
    
    Parameters:
    -----------
    checkpoint_dirs_dict : dict
        Dictionary mapping model names to checkpoint directory paths
    vector_path : str
        Path to vector field file
    mask_path : str
        Path to mask file
    subject_id : str
        Subject ID
    save_path_csv : str, optional
        Path to save CSV file
    save_path_fig : str, optional
        Path to save figure with table
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with all results
    """
    try:
        import seaborn as sns
        sns.set_style("white")
    except ImportError:
        pass
    
    results = []
    
    vec_torch = convert.read_nhdr(vector_path).cpu()
    vector_field = vec_torch.numpy().astype(float)
    vector_field_xyz = np.transpose(vector_field, (1, 2, 3, 0))
    vector_lin = vec_torch.to(dtype=torch.float32)
    mask = convert.read_nhdr(mask_path).cpu().numpy().astype(bool)
    mask_t = convert.read_nhdr(mask_path).cpu().to(dtype=torch.float32)
    
    start_point = select_oriented_point_on_slice(mask, vector_field_xyz, seed=0)
    
    for model_name, checkpoint_dir in checkpoint_dirs_dict.items():
        print(f"\n=== Processing {model_name} ===")
        subj_dir = os.path.join(checkpoint_dir, subject_id)
        metrics_dir = os.path.join(subj_dir, "metrics")
        loss_csv = os.path.join(subj_dir, "loss.csv")
        config_file = os.path.join(subj_dir, "training_config.json")
        
        # Extract metrics
        row = {'Architecture': model_name}
        
        # Final loss
        if os.path.isfile(loss_csv):
            df_loss = pd.read_csv(loss_csv)
            row['Loss'] = df_loss['loss'].iloc[-1]
        else:
            row['Loss'] = np.nan
        
        # Parameters
        num_params = np.nan
        if os.path.isfile(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                if 'num_params' in config:
                    num_params = config['num_params']
        row['Params (M)'] = num_params / 1e6 if not np.isnan(num_params) else np.nan
        
        # Load metric and compute quality metrics
        metric_final = os.path.join(metrics_dir, "metric_final.nhdr")
        if os.path.isfile(metric_final):
            quality_metrics = compute_geodesic_quality_metrics(
                metric_final, vector_path, mask_path, num_seed_points=5
            )
            row['Geod. Dist. (vox)'] = quality_metrics['mean_distance_to_gt']
            row['Angle (deg)'] = quality_metrics['mean_angle']
            row['Length Ratio'] = quality_metrics['mean_length_ratio']
            
            # Condition number and covariant error
            metric_lin = convert.read_nhdr(metric_final).cpu().numpy()
            if metric_lin.shape[-1] == 6:
                metric_6 = metric_lin
            elif metric_lin.shape[0] == 6:
                metric_6 = np.transpose(metric_lin, (1, 2, 3, 0))
            metric_3x3 = tensors.tens_6_to_tens_3x3(metric_6)
            
            flat = metric_3x3[mask].reshape(-1, 3, 3)
            eigvals, _ = np.linalg.eigh(flat)
            eigvals = np.sort(eigvals, axis=1)
            cond = eigvals[:, 2] / (eigvals[:, 0] + 1e-8)
            row['Cond. #'] = np.mean(cond)
            
            g_torch = torch.from_numpy(metric_3x3).to(dtype=torch.float32)
            nabla_vv = covariant_derivative_3d(vector_lin, g_torch, mask_t, differential_accuracy=2)
            e = torch.sqrt((nabla_vv ** 2).sum(dim=0)).cpu().numpy()
            row['Cov. Error'] = np.mean(e[mask])
        else:
            row['Geod. Dist. (vox)'] = np.nan
            row['Angle (deg)'] = np.nan
            row['Length Ratio'] = np.nan
            row['Cond. #'] = np.nan
            row['Cov. Error'] = np.nan
        
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # Save CSV
    if save_path_csv:
        df.to_csv(save_path_csv, index=False, float_format='%.4f')
        print(f"Table saved to {save_path_csv}")
    
    # Create figure with table
    if save_path_fig:
        fig, ax = plt.subplots(figsize=(14, max(6, len(df) * 0.5)))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Quantitative Results Comparison', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(save_path_fig, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path_fig}")
        plt.close()
    
    return df


def plot_metric_tensor_visualization(metric_file,
                                     mask_path,
                                     axis="z",
                                     slice_index=None,
                                     quiver_step=5,
                                     save_path=None):
    """
    Visualise la métrique comme des ellipsoïdes 3D projetés en 2D
    
    Chaque voxel = ellipsoïde dont les axes = valeurs propres
    Orientation = vecteurs propres
    Couleur = condition number
    
    Parameters:
    -----------
    metric_file : str
        Path to metric file
    mask_path : str
        Path to mask file
    axis : str
        Slice axis ('x', 'y', or 'z')
    slice_index : int, optional
        Slice index
    quiver_step : int
        Step size for ellipsoid visualization
    save_path : str, optional
        Path to save the figure
    """
    # Load metric
    metric_lin = convert.read_nhdr(metric_file).cpu().numpy()
    if metric_lin.shape[-1] == 6:
        metric_6 = metric_lin
    elif metric_lin.shape[0] == 6:
        metric_6 = np.transpose(metric_lin, (1, 2, 3, 0))
    else:
        raise ValueError(f"Unexpected metric shape: {metric_lin.shape}")
    
    metric_3x3 = tensors.tens_6_to_tens_3x3(metric_6)
    
    # Load mask
    mask = convert.read_nhdr(mask_path).cpu().numpy().astype(bool)
    X, Y, Z = mask.shape
    
    # Select slice
    if slice_index is None:
        slice_index = {"x": X // 2, "y": Y // 2, "z": Z // 2}[axis]
    
    # Extract slice
    if axis == "z":
        metric_slice = metric_3x3[:, :, slice_index, :, :]
        mask_slice = mask[:, :, slice_index]
        x_coords, y_coords = np.meshgrid(np.arange(Y), np.arange(X))
    elif axis == "y":
        metric_slice = metric_3x3[:, slice_index, :, :, :]
        mask_slice = mask[:, slice_index, :]
        x_coords, y_coords = np.meshgrid(np.arange(Z), np.arange(X))
    else:  # 'x'
        metric_slice = metric_3x3[slice_index, :, :, :, :]
        mask_slice = mask[slice_index, :, :]
        x_coords, y_coords = np.meshgrid(np.arange(Z), np.arange(Y))
    
    # Compute eigenvalues and eigenvectors
    eigvals_slice = np.zeros((*metric_slice.shape[:2], 3))
    eigvecs_slice = np.zeros((*metric_slice.shape[:2], 3, 3))
    cond_slice = np.zeros(metric_slice.shape[:2])
    
    for i in range(metric_slice.shape[0]):
        for j in range(metric_slice.shape[1]):
            g_mat = metric_slice[i, j]
            eigvals, eigvecs = np.linalg.eigh(g_mat)
            idx = np.argsort(eigvals)
            eigvals_sorted = eigvals[idx]
            eigvecs_sorted = eigvecs[:, idx]
            
            eigvals_slice[i, j] = eigvals_sorted
            eigvecs_slice[i, j] = eigvecs_sorted
            cond_slice[i, j] = eigvals_sorted[2] / (eigvals_sorted[0] + 1e-8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Background: condition number
    im = ax.imshow(cond_slice.T, origin='lower', cmap='viridis', alpha=0.6)
    plt.colorbar(im, ax=ax, label='Condition Number')
    
    # Overlay: ellipsoids as arrows (principal direction)
    for i in range(0, metric_slice.shape[0], quiver_step):
        for j in range(0, metric_slice.shape[1], quiver_step):
            if mask_slice[i, j]:
                # Principal eigenvector (largest eigenvalue)
                principal_dir = eigvecs_slice[i, j, :, 2]
                principal_val = eigvals_slice[i, j, 2]
                
                # Project to 2D (simplified: use first two components)
                if axis == "z":
                    u, v = principal_dir[0], principal_dir[1]
                elif axis == "y":
                    u, v = principal_dir[0], principal_dir[2]
                else:
                    u, v = principal_dir[1], principal_dir[2]
                
                # Scale by eigenvalue
                scale = np.sqrt(principal_val) * 2
                ax.arrow(j, i, v * scale, u * scale, 
                        head_width=0.5, head_length=0.5, 
                        fc='red', ec='red', alpha=0.7, linewidth=1)
    
    ax.set_title('Metric Tensor Visualization\n(Ellipsoids: Principal Directions)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_metric_structure_analysis(metric_file,
                                  mask_path,
                                  save_path=None):
    """
    Analyse la structure géométrique de la métrique
    
    1. Distribution des valeurs propres (histogrammes)
    2. Distribution des angles entre vecteurs propres et vector field
    3. Carte de l'anisotropie (lambda_max / lambda_min)
    4. Carte de la "directionnalité" (alignement avec vector field)
    
    Parameters:
    -----------
    metric_file : str
        Path to metric file
    mask_path : str
        Path to mask file
    save_path : str, optional
        Path to save the figure
    """
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        pass
    
    # Load metric
    metric_lin = convert.read_nhdr(metric_file).cpu().numpy()
    if metric_lin.shape[-1] == 6:
        metric_6 = metric_lin
    elif metric_lin.shape[0] == 6:
        metric_6 = np.transpose(metric_lin, (1, 2, 3, 0))
    else:
        raise ValueError(f"Unexpected metric shape: {metric_lin.shape}")
    
    metric_3x3 = tensors.tens_6_to_tens_3x3(metric_6)
    
    # Load mask
    mask = convert.read_nhdr(mask_path).cpu().numpy().astype(bool)
    
    # Flatten and filter by mask
    flat = metric_3x3[mask].reshape(-1, 3, 3)
    
    # Eigenvalues
    eigvals, eigvecs = np.linalg.eigh(flat)
    eigvals = np.sort(eigvals, axis=1)
    lam_min = eigvals[:, 0]
    lam_mid = eigvals[:, 1]
    lam_max = eigvals[:, 2]
    
    # Condition number
    cond = lam_max / (lam_min + 1e-8)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Metric Structure Analysis', fontsize=16, fontweight='bold')
    
    # 1. Eigenvalue distributions
    axes[0, 0].hist(lam_min, bins=50, alpha=0.6, label='λ_min', color='blue')
    axes[0, 0].hist(lam_mid, bins=50, alpha=0.6, label='λ_mid', color='green')
    axes[0, 0].hist(lam_max, bins=50, alpha=0.6, label='λ_max', color='red')
    axes[0, 0].set_xlabel('Eigenvalue', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Eigenvalue Distributions', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Condition number distribution
    axes[0, 1].hist(cond, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[0, 1].set_xlabel('Condition Number (λ_max / λ_min)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Anisotropy Distribution', fontsize=13, fontweight='bold')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].axvline(cond.mean(), color='red', linestyle='--', 
                      label=f'Mean: {cond.mean():.2f}')
    axes[0, 1].legend()
    
    # 3. Anisotropy map (slice)
    X, Y, Z = mask.shape
    slice_idx = Z // 2
    cond_3d = np.zeros_like(mask, dtype=float)
    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                if mask[i, j, k]:
                    g_mat = metric_3x3[i, j, k]
                    eigvals_local, _ = np.linalg.eigh(g_mat)
                    eigvals_local = np.sort(eigvals_local)
                    cond_3d[i, j, k] = eigvals_local[2] / (eigvals_local[0] + 1e-8)
    
    im = axes[1, 0].imshow(cond_3d[:, :, slice_idx].T, origin='lower', 
                           cmap='hot', interpolation='nearest')
    axes[1, 0].set_title('Anisotropy Map (Slice)', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('X', fontsize=12)
    axes[1, 0].set_ylabel('Y', fontsize=12)
    plt.colorbar(im, ax=axes[1, 0], label='Condition Number')
    
    # 4. Eigenvalue ratios
    ratio_min_mid = lam_mid / (lam_min + 1e-8)
    ratio_max_mid = lam_max / (lam_mid + 1e-8)
    
    axes[1, 1].scatter(ratio_min_mid, ratio_max_mid, alpha=0.3, s=1, c=cond, 
                      cmap='viridis')
    axes[1, 1].set_xlabel('λ_mid / λ_min', fontsize=12)
    axes[1, 1].set_ylabel('λ_max / λ_mid', fontsize=12)
    axes[1, 1].set_title('Eigenvalue Ratio Analysis', fontsize=13, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Condition Number')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    
    """plot_geodesics_from_checkpoints(
        metrics_dir="../Checkpoints_10_10_10/111312/metrics",
        mask_path="../Brains/111312/111312_shrinktensor_filt_mask.nhdr",
        seed_start_point=0,
        init_velocity= None,
        num_intermediate=3,
        DIR = True,
        vector_path="../Brains/111312/111312_shrinktensor_principal_vector_field.nhdr",
        max_length=7
    )"""

    """plot_geodesics_on_slice_from_checkpoints(
    metrics_dir="../Checkpoints_10_10_10/111312/metrics",
    mask_path="../Brains/111312/111312_shrinktensor_filt_mask.nhdr",
    vector_path="../Brains/111312/111312_shrinktensor_principal_vector_field.nhdr",
    seed_start_point =0,
    axis="z",             # 'x', 'y' ou 'z'
    slice_index=None,
    num_intermediate=3,
    zoom=True,
    zoom_size=10,
    iter_num=8000,
    quiver_step=1,
    arrow_len=1.5,
    max_length=7
    )"""
    
    """analyze_metric_field("../Checkpoints_10_10_10/111312/metrics",
     mask_path="../Brains/111312/111312_shrinktensor_filt_mask.nhdr")"""

    """debug_TDir(metrics_dir = "../Checkpoints_10_10_10/111312/metrics",
               mask_path= "../Brains/111312/111312_shrinktensor_filt_mask.nhdr",
               axis="z",
               slice_index=None,
               quiver_step=3,
               arrow_len=2.0)"""

    """plot_metric_eigs_and_covariant_error(
        metrics_dir="../Checkpoints_10_10_10/111312/metrics",
        vector_path="../Brains/111312/111312_shrinktensor_principal_vector_field.nhdr",
        mask_path="../Brains/111312/111312_shrinktensor_filt_mask.nhdr",
        axis="z",
        slice_index=None
    )"""
    compare_baseline_metrics_on_slice(
        vector_path="../Brains/111312/111312_shrinktensor_principal_vector_field.nhdr",
        checkpoint_dirs=["../Checkpoints_10_10_10", "../Checkpoints_40_30_40"],
        subject_id="111312",
        mask_path="../Brains/111312/111312_shrinktensor_filt_mask.nhdr",
        axis="z"
    )

    """plot_loss_vs_anisotropy(checkpoint_dir="../Checkpoints_10_10_10",
                            subject_id="111312",
                            mask_path="../Brains/111312/111312_shrinktensor_filt_mask.nhdr")"""
