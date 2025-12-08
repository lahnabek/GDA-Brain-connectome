# Modeling the Shape of the Brain Connectome via Deep Neural Networks

Implementation (2D and 3D) of a **Metric CNN** to estimate a **Riemannian manifold** that faithfully represents the brain connectome from vector fields derived from tractography data.

Based on the paper:

> Haocheng Dai, Martin Bauer, P. Thomas Fletcher, Sarang C. Joshi  
> **Deep Learning the Shape of the Brain Connectome**  
> *arXiv:2203.06122*

---

## Repository Overview

- **`Metric-Cnn-3D-IPMI/`** – 3D implementation (full 3D connectome).  
- **`Metric-Cnn-2D-IPMI/`** – 2D implementation (toy case / 2D slices and synthetic simulations).  
- **Global data and scripts:**
  - `HCP1065_fiber.nii*`, `HCP1065.1.25mm*.fib*`: example HCP connectome data
  - `vector_field_2D.npy`, `vector_fields.mha`: example vector fields
  - `prepare_metriccnn_data.py`: data preprocessing and conversion script for MetricCNN
  - `setup_metriccnn_env.py`: helper for creating the Conda environment

---

## Quick Setup

Both 2D and 3D projects share the same core libraries (PyTorch, SimpleITK, scikit‑image, numba, etc.).  
Each subfolder contains its own `environment.yml` and `requirements.txt`.

Example for the 3D version (similar for 2D):

```bash
cd Metric-Cnn-3D-IPMI
conda env create -f environment.yml
conda activate metcnn
```

---

## Structure of `Metric-Cnn-3D-IPMI`

**Goal:** Learn a 3D Riemannian metric field (SPD 3×3 per voxel) from principal vector fields (tractography) by enforcing a **geometric PDE loss** based on the covariant derivative \(\nabla_v v\).

### Main structure

- **`Brains/`** – Input HCP 3D data  
  - `*_shrinktensor_principal_vector_field.nhdr/.raw`: principal vector field  
  - `*_shrinktensor_filt_mask.nhdr/.raw`: brain mask  

- **`Checkpoints_*/`** – Results of multiple experiments  
  - One subfolder per subject (`111312/`, etc.)  
  - Contains `checkpoint_epoch_XXXX.pth.tar`, `model_final.pth.tar`,  
    associated `metrics/metric_epoch_XXXX.nhdr`, `loss.csv`, and `training_config.json`  

- **`Figures/`** – Figures included in the paper (architecture, eigen‑decomposition, performance)

- **`environment.yml`, `requirements.txt`** – Environment dependencies

- **`README.md`** – 3D‑specific documentation

### 3D Scripts (`Metric-Cnn-3D-IPMI/Scripts/`)

- **`runMetricCnnTrainingInference.sh`** – main entry point (training + inference using default config)
- **`MetricCnnTrainingInference.py`** – main Python script  
  - Defines `brain_id`, input/output paths, training parameters  
  - Creates model `DenseED` (3D) from `model3D.py`  
  - Uses dataset `DatasetHCP` from `dataset.py`  
  - Loss function: PDE residual \(\nabla_v v - \sigma v\) weighted by mask  
  - Saves losses, checkpoints, and final learned metric field
- **`dataset.py`** – class `DatasetHCP` for loading vector fields and masks
- **`model3D.py`** – defines the **Dense Encoder–Decoder 3D** architecture
- **`pde.py`** – defines `eigen_composite()` and `pde()` functions implementing the geometric loss
- **Additional utilities:**  
  - `plot_geodesic.py`, `test_vis.py`, `lazy_imports.py`

### 3D Packages (`Metric-Cnn-3D-IPMI/Packages/`)

- **`algo/`** – algorithms for geodesics (`geodesic.py`, `euler.py`, `dijkstra.py`, `metricModSolver.py`)
- **`data/`** – data I/O and conversion for NRRD/NHDR formats (`convert.py`, `io.py`, `nrrd.py`)
- **`disp/`** – visualization helpers (`vis.py`)
- **`util/`** – Riemannian and tensor utilities (`riemann.py`, `tensors.py`, `diff.py`, `maskops.py`, etc.)

---

## Structure of `Metric-Cnn-2D-IPMI`

**Goal:** 2D version of MetricCNN for simplified experiments (toy data / 2D slices).  
Learns a 2D Riemannian metric (SPD 2×2) with a PDE‑based geometric loss.

### Main structure

- **`Brains/`** – Example 2D data (HCP subjects and sinusoidal vector fields)
  - `100610/`, `cos/`, `sin/`, etc.  

- **`Figures/`** – 2D architecture and performance figures

- **`environment.yml`, `requirements.txt`** – 2D dependencies  
- **`README.md`** – 2D‑specific readme

### 2D Scripts (`Metric-Cnn-2D-IPMI/Scripts/`)

- **`runMetricCnnTrainingInference.sh`** – shell launcher (training + inference)
- **`MetricCnnTrainingInference.py`** – main 2D training/inference script  
  - Builds `DenseED(in_channels=2, out_channels=3)`  
  - Loads dataset via `ImageDataset` from `dataset.py`  
  - Loss PDE via `pde(u, vector_lin, mask)` from `pde.py`  
  - Saves `loss.txt` + `model.pth.tar`  
  - Produces `*_learned_metric_final.nhdr`
- **`model.py`** – defines the **Dense Encoder–Decoder 2D** CNN
- **`dataset.py`** – class `ImageDataset` to load 2D vector fields and masks
- **`pde.py`** – defines matrix exponential and 2D PDE residual  
- **Additional scripts:**  
  - `plot.py`, `GeodesicPlottingBraid.ipynb`, `MetricCnnTrainingInferenceBraid.ipynb`, etc.

### 2D Packages (`Metric-Cnn-2D-IPMI/Packages/`)

- **`algo/`** – algorithms for 2D geodesic computation
- **`apps/`** – high‑level application modules (`metricEstimation.py`, `geodesicShooting.py`)
- **`sims/`** – synthetic simulation modules (`simTensors2D.py`, `metricEstSim.py`, `testCase.py`)
- **`test/`** – test scripts and examples (`test2DTensorGen.py`, `testGeodesic.py`, etc.)
- **`data/`, `disp/`, `util/`** – I/O, visualization, and Riemannian utility modules (shared with 3D version)

---

## Key Differences Between 2D and 3D Versions

| Aspect | 2D Version | 3D Version |
|--------|-------------|-------------|
| **Dimension** | SPD 2×2 metric, vector fields 2D | SPD 3×3 metric, vector fields 3D |
| **Parametrization** | 3 output channels → matrix exponential | 7 output channels → eigen‑decomposition |
| **Applications** | Mainly synthetic data & toy simulations | Real HCP connectome data |
| **Extras** | Includes `apps/`, `sims/`, `test/` folders for additional experiments | Focused on PDE training and geodesic visualization |

---

## How to Navigate

- **To train on 3D HCP data:**  
  See `Metric-Cnn-3D-IPMI/README.md` and `Scripts/runMetricCnnTrainingInference.sh`.

- **To train or test in 2D:**  
  See `Metric-Cnn-2D-IPMI/README.md` and `Scripts/runMetricCnnTrainingInference.sh`.
  Use accompanying notebooks (`MetricCnnTrainingInferenceBraid.ipynb`, etc.) for visualization.

- **To explore the Riemannian geometry core:**  
  Review `Packages/util/riemann.py` and `Scripts/pde.py`.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{dai2022deep,
  title   = {Deep Learning the Shape of the Brain Connectome},
  author  = {Dai, Haocheng and Bauer, Martin and Fletcher, P. Thomas and Joshi, Sarang C},
  journal = {arXiv preprint arXiv:2203.06122},
  year    = {2022}
}
```