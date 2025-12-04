import os, sys
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# ---- Fix imports ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from Packages.disp.vis import show_slice

def lire_image():
    # ====================================================
    # 1. Lire l’image
    # ====================================================
    img_path = os.path.join(PROJECT_ROOT,
                            "Brains/111312/111312_shrinktensor_principal_vector_field.nhdr")

    img = sitk.ReadImage(img_path)

    # RAW SITK shape = (C, Z, Y, X)
    raw = sitk.GetArrayFromImage(img)
    print("RAW shape SITK:", raw.shape)

    # ====================================================
    # 2. REMETTRE DANS L’ORDRE CORRECT (X, Y, Z, C)
    # ====================================================
    # raw = (C, Z, Y, X)
    # Réordonner → (X, Y, Z, C)
    np_img = np.transpose(raw, (3, 2, 1, 0))

    print("Shape corrigée :", np_img.shape)   # (48,64,48,3)


    # ====================================================
    # 3. Afficher une slice
    # ====================================================
    mid = np_img.shape[2] // 2

    show_slice(np_img, slc_no=mid, ax=2, title="Slice Z du champ de vecteurs")








