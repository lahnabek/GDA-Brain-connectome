# Modeling the Shape of the Brain Connectome via Deep Neural Networks

Implementation (2D et 3D) du **Metric CNN** pour estimer une **variété riemannienne** représentant le connectome cérébral à partir de champs de vecteurs dérivés de données de tractographie.

Basé sur l’article :

> Haocheng Dai, Martin Bauer, P. Thomas Fletcher, Sarang C. Joshi  
> **Deep Learning the Shape of the Brain Connectome**  
> arXiv:2203.06122

---

## Contenu du dépôt

- **`Metric-Cnn-3D-IPMI/`** : implémentation 3D (connectome complet 3D).
- **`Metric-Cnn-2D-IPMI/`** : implémentation 2D (cas jouet / coupes 2D, simulations).
- **Données et scripts globaux :**
  - `HCP1065_fiber.nii*`, `HCP1065.1.25mm*.fib*` : exemples de données connectome HCP.
  - `vector_field_2D.npy`, `vector_fields.mha` : exemples de champs de vecteurs.
  - `prepare_metriccnn_data.py` : préparation / conversion de données pour MetricCNN.
  - `setup_metriccnn_env.py` : script d’aide pour créer l’environnement Conda.

---

## Installation rapide

Les deux projets (2D et 3D) ont leur propre environnement, mais utilisent les mêmes librairies de base (PyTorch, SimpleITK, scikit-image, numba, etc.).

Pour le 3D (similaire pour le 2D) :

cd Metric-Cnn-3D-IPMI
conda env create -f environment.yml
conda activate metcnnLes fichiers `environment.yml` et `requirements.txt` dans chaque sous-dossier listent toutes les dépendances nécessaires.

---

## Structure de `Metric-Cnn-3D-IPMI`

**But :** apprendre un champ de métriques riemanniennes 3D (SPD 3×3 par voxel) à partir de champs de vecteurs principaux (tractographie) et imposer une **perte géométrique** via une PDE (dérivée covariante \(\nabla_v v\)).

### Arborescence principale

- **`Brains/`**
  - Données d’entrée HCP 3D (ex. sujet `111312/`) :
    - `*_shrinktensor_principal_vector_field.nhdr/.raw` : champ de vecteurs principal.
    - `*_shrinktensor_filt_mask.nhdr/.raw` : masque cérébral filtré.

- **`Checkpoints_*/`**
  - Résultats de plusieurs expériences (ne pas nécessairement modifier) :
    - `111312/` : sous-dossier par sujet.
      - `checkpoint_epoch_XXXX.pth.tar` : checkpoints PyTorch.
      - `model_final.pth.tar` : modèle final.
      - `metrics/metric_epoch_XXXX.nhdr/.raw`, `metric_final.nhdr/.raw` : métriques apprises.
      - `loss.csv` : historique de loss.
      - `training_config.json` : configuration du run.

- **`Figures/`**
  - `architecture.png` : architecture du Metric CNN 3D.
  - `eigencomposition.png`, `performance.png` : figures pour le papier / présentation.

- **`environment.yml`, `requirements.txt`**
  - Dépendances Python pour reproduire l’environnement 3D.

- **`README.md`**
  - README spécifique à la version 3D (quickstart, explications, liens vers papier).

### Scripts 3D (dossier `Metric-Cnn-3D-IPMI/Scripts/`)

- **`runMetricCnnTrainingInference.sh`**
  - Point d’entrée recommandé : lance l’entraînement + inference avec une config par défaut.

- **`MetricCnnTrainingInference.py`**
  - **Script principal de training/inference 3D.**
  - Paramètres en haut de fichier :
    - `brain_id`, `input_dir` (souvent `../Brains`), `output_dir` (souvent `../Checkpoints_*`),
    - `epoch_num`, `learning_rate`, `BLOCKS = [40, 30, 40]`, etc.
  - Crée un modèle `DenseED` (3D) défini dans `model3D.py` :
    - `in_channels = 3` (3 composantes du champ de vecteurs),
    - `out_channels = 7` (paramètres pour une métrique SPD 3×3 via `eigen_composite`).
  - Utilise le dataset `DatasetHCP` (dans `dataset.py`) pour charger un sujet HCP.
  - Boucle d’entraînement :
    - `u_pred = model(input)` → paramétrisation de la métrique,
    - perte PDE `pde(u_pred, input, mask)` (dans `pde.py`),
    - loss MSE entre résidu PDE pondéré par le masque et 0,
    - écriture de `loss.csv`, checkpoints et métriques intermédiaires (`metrics/metric_epoch_XXXX.nhdr`).
  - À la fin, sauvegarde `model_final.pth.tar` et `metric_final.nhdr`.

- **`dataset.py`**
  - Classe `DatasetHCP` :
    - charge une fois par sujet le champ `*_shrinktensor_principal_vector_field.nhdr` (mis à l’échelle ×1000),
    - charge le masque `*_shrinktensor_filt_mask.nhdr` et met les bords à 0,
    - renvoie des `dict` :
      - `'vector_field'` : champ de vecteurs 3D,
      - `'mask'` : masque binaire 3D (1 canal).

- **`model3D.py`**
  - Architecture **Dense Encoder–Decoder 3D** :
    - `_DenseLayer`, `_DenseBlock` (style DenseNet),
    - `_Transition` (downsampling/upsampling 3D),
    - `DenseED` : réseau complet, entièrement convolutionnel 3D (`Conv3d`, `BatchNorm3d`, upsampling nearest/trilinear),
    - `Decoder` : décoder 3D autonome pour PDE.
  - Utilisé par `MetricCnnTrainingInference.py` pour prédire les 7 canaux paramétrisant la métrique.

- **`pde.py`**
  - `eigen_composite(u)` :
    - recompose une matrice SPD 3×3 à partir des 7 canaux de sortie `u` (rotation \(R(\theta, K)\) + valeurs propres exponentiées).
  - `pde(u, vector_lin, mask, differential_accuracy=2)` :
    - calcule \(\nabla_v v\) via `riemann.covariant_derivative_3d` (dans `Packages/util/riemann.py`),
    - projette / normalise pour obtenir un résidu PDE \(\nabla_v v - \sigma v\),
    - sert de loss (MSE -> 0).

- **Autres scripts** :
  - `plot_geodesic.py` : visualisation de géodésiques sur la métrique apprise.
  - `test_vis.py` : tests de visualisation.
  - `lazy_imports.py` : utilitaires d’import paresseux.

### Packages 3D (dossier `Metric-Cnn-3D-IPMI/Packages/`)

- **`algo/`**
  - `geodesic.py` : calcul / intégration de géodésiques sur la métrique 3D.
  - `euler.py` : intégrateurs (méthodes d’Euler, etc.) pour ODE géodésiques.
  - `dijkstra.py` : géodésiques de type shortest‑path (Dijkstra).
  - `metricModSolver.py` : solveur / modificateur de métriques (scénarios 3D spécifiques).

- **`data/`**
  - `convert.py` : lecture/écriture NRRD/NHDR et conversions vers tensors PyTorch.
  - `io.py`, `nrrd.py` : I/O bas niveau pour formats NRRD/NHDR.
  - `constants.py`, `gen.py`, `fileManager.py` : constantes, génération de données, gestion fichiers.

- **`disp/`**
  - `vis.py` : fonctions de visualisation (slices, champs de vecteurs, tenseurs…).

- **`util/`**
  - `riemann.py` : routines riemanniennes (dérivée covariante 2D/3D, connexions, etc.).
  - `tensors.py` : conversions lin↔mat pour tenseurs symétriques.
  - `diff.py` : opérateurs de différences finies.
  - `maskops.py` : opérations sur masques.
  - `monitor.py`, `parsers.py`, `YAMLcfg.py` : utils divers.

---

## Structure de `Metric-Cnn-2D-IPMI`

**But :** version 2D du MetricCNN pour des expériences plus simples (coupes 2D, données synthétiques), partageant la même philosophie : apprendre une métrique riemannienne 2D (SPD 2×2) avec une perte géodésique via PDE.

### Arborescence principale

- **`Brains/`**
  - `100610/` : sujet HCP 2D (plusieurs fichiers `*_vector_field`, `*_filt_mask`, tenseurs bruts et scalés).
  - `cos/`, `sin/` : exemples de champs de vecteurs 2D (cosinus, sinus) + masques.

- **`Figures/`**
  - `architecture.png`, `eigencomposition.png`, `performance.png` : figures pour la version 2D.

- **`environment.yml`, `requirements.txt`**
  - Dépendances Python nécessaires à la version 2D.

- **`README.md`**
  - README spécifique à la version 2D.

### Scripts 2D (dossier `Metric-Cnn-2D-IPMI/Scripts/`)

- **`runMetricCnnTrainingInference.sh`**
  - Script shell de lancement (training + inference 2D).

- **`MetricCnnTrainingInference.py`**
  - Script principal d’entraînement 2D :
    - crée un `DenseED(in_channels=2, out_channels=3, imsize=100, blocks=[6,8,6], ...)`,
    - dataset : `ImageDataset` (défini dans `dataset.py`),
    - loss PDE 2D : `pde(u, vector_lin, mask)` (dans `pde.py`),
    - écrit `loss.txt` + `model.pth.tar` dans `output_dir/brain_id/`.
  - En fin de script :
    - recharge le modèle,
    - applique sur le champ complet,
    - reconstruit la métrique SPD 2×2 via `matrix_exp_2d` et écrit `*_learned_metric_final.nhdr`.

- **`model.py`**
  - Architecture **Dense Encoder–Decoder 2D** (même logique que `model3D.py` mais en 2D) :
    - `_DenseLayer`, `_DenseBlock`, `_Transition`,
    - `DenseED` et `Decoder` en 2D (`Conv2d`, `BatchNorm2d`, upsampling bilinear/nearest).

- **`dataset.py`**
  - `ImageDataset` :
    - lit `*_vector_field.nhdr` (vecteurs 2D),
    - lit `*_filt_mask.nhdr`, calcule un mask de frontière par `filters.laplace`, enlève la frontière,
    - renvoie `'vector_field'` et `'mask'`.

- **`pde.py`**
  - `matrix_exp_2d(A)` : construit une matrice SPD 2×2 (exponentielle de matrice fermée-forme).
  - `pde(u, vector_lin, mask, differential_accuracy=2)` :
    - `s = tensors.lin2mat(u)`,
    - `metric_mat = matrix_exp_2d(s)`,
    - `nabla_vv = riemann.covariant_derivative_2d(vector_lin, metric_mat, mask, ...)`,
    - renvoie le résidu PDE \(\nabla_v v - \sigma v\) (2 composantes).

- **Autres scripts** :
  - `plot.py` : visualisation 2D.
  - `GeodesicPlottingBraid.ipynb`, `GeodesicPlottingHCP.ipynb` : notebooks de tracé de géodésiques.
  - `MetricCnnTrainingInferenceBraid.ipynb`, `MetricCnnTrainingInferenceHCP.ipynb` : notebooks d’entraînement/inférence.

### Packages 2D (dossier `Metric-Cnn-2D-IPMI/Packages/`)

- **`algo/`**
  - `geodesic.py`, `euler.py`, `dijkstra.py` : analogues 2D des algorithmes géodésiques.

- **`apps/`** (présent en 2D, **pas** en 3D)
  - `appTypes.py` : définitions de types / structures haut niveau.
  - `geodesicShooting.py` : shooting géodésique 2D.
  - `metricEstimation.py` : pipeline d’estimation de métrique 2D.

- **`sims/`** (présent en 2D, **pas** en 3D)
  - `simTensors2D.py` : génération de champs de tenseurs 2D synthétiques.
  - `metricEstSim.py`, `testCase.py` : scénarios de simulation de métrique.

- **`test/`** (présent en 2D, **pas** en 3D)
  - `test2DTensorGen.py`, `test3DTensorGen.py`, `test3DBrainMetricEst.py`, `test3DCubicMetricEst.py` :
    tests de génération / estimation de tenseurs 2D/3D.
  - `testGeodesic.py`, `testMetricEstimation.py`, `testMetricEstSim.py` :
    tests de géodésiques et d’estimation de métrique.
  - `testSITKVis.py` : tests I/O & visualisation SimpleITK.
  - `input/annulus_tensors.mat` : exemple de données synthétiques.

- **`data/`, `disp/`, `util/`**
  - Très proches de la version 3D (conversion, I/O, visualisation, riemann, tenseurs, etc.).
  - En plus, dans `util/`, un module historique `tensorstrash.py` (fonctions de tenseurs dépréciées / expérimentales).

---

## Différences clés entre `Metric-Cnn-3D-IPMI` et `Metric-Cnn-2D-IPMI`

- **Dimension**
  - 3D : métriques SPD 3×3, champs de vecteurs 3D, conv 3D, dérivée covariante 3D.
  - 2D : métriques SPD 2×2, champs de vecteurs 2D, conv 2D, dérivée covariante 2D.

- **Paramétrisation de la métrique**
  - 3D : sortie réseau de 7 canaux → rotation \(R(\theta, K)\) + valeurs propres exponentiées (`eigen_composite`).
  - 2D : sortie réseau de 3 canaux → exponentielle de matrice 2×2 (`matrix_exp_2d`).

- **Écosystème**
  - 3D : plus minimal, ciblé sur connectome réel HCP + visualisation géodésique.
  - 2D : plus riche en modules d’**applications, simulations et tests** (`apps/`, `sims/`, `test/`).

---

## Comment s’y retrouver

- **Pour entraîner un modèle 3D sur un sujet HCP :**
  - Voir `Metric-Cnn-3D-IPMI/README.md` et `Metric-Cnn-3D-IPMI/Scripts/runMetricCnnTrainingInference.sh`.
  - Config et boucle d’entraînement : `Metric-Cnn-3D-IPMI/Scripts/MetricCnnTrainingInference.py`.

- **Pour entraîner / tester en 2D :**
  - Voir `Metric-Cnn-2D-IPMI/README.md` et `Metric-Cnn-2D-IPMI/Scripts/runMetricCnnTrainingInference.sh`.
  - Scripts d’expérimentation/visualisation supplémentaires : notebooks et `plot.py`.

- **Pour comprendre la géométrie riemannienne / PDE :**
  - Routines de base : `Metric-Cnn-*/Packages/util/riemann.py` et `Metric-Cnn-*/Scripts/pde.py`.
  - Conversion tenseurs ↔ matrices : `Metric-Cnn-*/Packages/util/tensors.py`.

---

## Citation

Si vous utilisez ce code dans un travail scientifique, merci de citer :
tex
@article{dai2022deep,
  title   = {Deep Learning the Shape of the Brain Connectome},
  author  = {Dai, Haocheng and Bauer, Martin and Fletcher, P Thomas and Joshi, Sarang C},
  journal = {arXiv preprint arXiv:2203.06122},
  year    = {2022}
}