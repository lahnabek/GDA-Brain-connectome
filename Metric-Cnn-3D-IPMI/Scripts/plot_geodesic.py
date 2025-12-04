import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

sys.path.insert(0, PROJECT_ROOT)

import glob
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import Packages.data.convert as convert
from Packages.util import tensors
from Packages.algo.geodesic import geodesicpath_3d
from Packages.algo.metricModSolver import solve_3d  



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
        vector_path=None):

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
    # 4) Couleurs + Figure
    # -------------------------------
    cmap = plt.cm.viridis(np.linspace(0,1,len(selected_files)))
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # -------------------------------
    # 5) Pour chaque métrique : solve_3d → vector_lin → geodesic
    # -------------------------------
    for k, metric_file in enumerate(selected_files):

        print(f"\n=== Processing {metric_file} ===")

        # Lire metric (format NHDR) – ce fichier contient la métrique g sous forme linéarisée (6 composantes)
        tens_lin = convert.read_nhdr(metric_file).cpu().numpy()  # (X,Y,Z,6) ou (6,X,Y,Z)
        if tens_lin.shape[-1] == 6:
            # (6,X,Y,Z)
            metric_6 = tens_lin
            tensor_lin_g = np.transpose(tens_lin, (3, 0, 1, 2))  # (6,X,Y,Z)
        else:
            tensor_lin_g = tens_lin                             # (6,X,Y,Z)
            metric_6 = np.transpose(tens_lin, (1, 2, 3, 0))     # (X,Y,Z,6)

        # -------------------------------
        # 1) Construire g(x) en 3x3
        # -------------------------------
        metric_3x3 = tensors.tens_6_to_tens_3x3(metric_6)       # (X,Y,Z,3,3)

        # -------------------------------
        # 2) Champ de directions : champ de vecteurs d'entrée (indépendant de g)
        # -------------------------------
        vector_lin = vector_field  # (3,X,Y,Z)

        # -------------------------------
        # 3) Inverse de la métrique pour geodesicpath_3d
        # -------------------------------
        g_inv_3x3 = np.linalg.inv(metric_3x3)                  # (X,Y,Z,3,3)
        tensor_lin_for_geo_6 = tensors.tens_3x3_to_tens_6(g_inv_3x3)  # (X,Y,Z,6)
        tensor_lin_for_geo = np.transpose(tensor_lin_for_geo_6, (3, 0, 1, 2))  # (6,X,Y,Z)

        #select start point:

        start_point = select_seeded_mask_point(mask, seed=seed_start_point, margin=2)
        # -------------------------------
        # Géodésique
        # -------------------------------
        gx, gy, gz = geodesicpath_3d(
            tensor_lin=tensor_lin_for_geo,
            vector_lin=vector_lin,
            mask_image=mask,
            start_coordinate=start_point,
            initial_velocity=init_velocity,
            delta_t=delta_t,
            iter_num=iter_num,
            stop_angle=stop_angle,
            both_directions = DIR
        )

        ax.plot(gx, gy, gz, color=cmap[k], linewidth=2,
                label=os.path.basename(metric_file))
    # --- afficher START POINT sur la coupe ---
    sx, sy, sz = start_point

    ax.scatter(sx, sy, sz, color="black", s=80, marker="*", zorder=15)


    ax.set_title("Évolution des géodésiques au cours du training")
    ax.legend()
    plt.tight_layout()
    plt.show()


def select_oriented_point_on_slice(mask, vec_field, axis="z",
                                   slice_index=None, seed=0,
                                   margin=2, min_inplane=0.3):
    """
    Sélectionne un point (x,y,z) :
      - dans le mask,
      - dans la slice (axis, slice_index),
      - où la direction principale TDir est bien dans le plan
        (grande composante dans le plan),
      - à au moins `margin` voxels des bords,
      - de façon reproductible via `seed`.
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
        y, z = valid[rng.integers(0, len(valid))]
        x = slice_index

    return (int(x), int(y), int(z)), int(slice_index)



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
    fig = plt.figure(figsize=(7,7))
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
                min_inplane=0.7
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
            stop_angle=stop_angle
        )

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

        ax.plot(
            proj_x, proj_y,
            color=cmap[k],
            linewidth=2,
            label=os.path.basename(metric_file)
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
                linewidths=0.5
            )

    # --- 5) Marquer le point de départ ---
    sx, sy, sz = start_point
    if axis == "z":
        sx2, sy2 = sx, sy
    elif axis == "y":
        sx2, sy2 = sx, sz
    else:
        sx2, sy2 = sy, sz

    ax.scatter(sx2, sy2, color="black", s=80, marker="*", zorder=15)

    ax.set_aspect("equal")
    ax.set_title(f"Géodésiques projetées — Coupe {axis}={slice_index_eff}")
    ax.legend()

    # --- Option de zoom autour du point de départ ---
    if zoom:
        cx, cy = sx2, sy2
        ax.set_xlim(cx - zoom_size, cx + zoom_size)
        ax.set_ylim(cy - zoom_size, cy + zoom_size)

    plt.tight_layout()
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


if __name__ == "__main__":
    """
    plot_geodesics_from_checkpoints(
        metrics_dir="../Checkpoints_10_10_10/111312/metrics",
        mask_path="../Brains/111312/111312_shrinktensor_filt_mask.nhdr",
        seed_start_point=0,
        init_velocity= None,
        num_intermediate=3,
        DIR = True
    )"""

    plot_geodesics_on_slice_from_checkpoints(
    metrics_dir="../Checkpoints_10_10_10/111312/metrics",
    mask_path="../Brains/111312/111312_shrinktensor_filt_mask.nhdr",
    seed_start_point =5,
    axis="z",             # 'x', 'y' ou 'z'
    slice_index=None,
    num_intermediate=3,
    )
    
    """analyze_metric_field("../Checkpoints_10_10_10/111312/metrics",
     mask_path="../Brains/111312/111312_shrinktensor_filt_mask.nhdr")"""

    """debug_TDir(metrics_dir = "../Checkpoints_10_10_10/111312/metrics",
               mask_path= "../Brains/111312/111312_shrinktensor_filt_mask.nhdr",
               axis="z",
               slice_index=None,
               quiver_step=3,
               arrow_len=2.0)"""
