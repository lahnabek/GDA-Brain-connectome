import nibabel as nib
import numpy as np
import SimpleITK as sitk

fiber_file = "HCP1065_fiber.nii.gz"

output_3d = "vector_fields.mha"
output_2d = "vector_field_2D.npy"

slice_index = 60  # coupe pour le modèle 2D

print("Chargement du fichier fiber.nii.gz ...")
img = nib.load(fiber_file)
data = img.get_fdata()   # shape (H, W, D, C)
H, W, D, C = data.shape

print(f"Shape détecté : (H={H}, W={W}, D={D}, C={C})")

# On prend seulement les 3 premiers peaks = 9 canaux
if C < 9:
    raise ValueError("Pas assez de composantes pour extraire 3 peaks (9 canaux nécessaires).")

data = data[:, :, :, :9]  # (H, W, D, 9)

# reshape : (H,W,D,9) → (H,W,D,3,3)
data = data.reshape(H, W, D, 3, 3)

# reorder axes → (m=3, 3, H, W, D)
data = np.moveaxis(data, (3,4,0,1,2), (0,1,2,3,4))
data = data.astype(np.float32)

# Sauvegarde 3D
img_mha = sitk.GetImageFromArray(data)
sitk.WriteImage(img_mha, output_3d)
print(f"Fichier 3D créé : {output_3d}")

# Sauvegarde 2D
slice2D = data[0, :, :, slice_index]  # peak1, couche slice_index
np.save(output_2d, slice2D)
print(f"Fichier 2D créé : {output_2d}")

print("Terminé.")
