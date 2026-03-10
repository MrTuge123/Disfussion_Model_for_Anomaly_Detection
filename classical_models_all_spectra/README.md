# Anomaly Detection on Spectroscopy Data — Classical Models

This directory benchmarks five unsupervised anomaly detection models on spectral data. The goal is to identify **ambient (non-polymer) particles** that were never seen during training, using only the spectral signature of known polymer samples to define what "normal" looks like.

---

## Problem Setup

This is a **one-class classification** (novelty detection) problem, not a standard supervised classification problem. The key distinction:

- Models are trained **only on normal polymer spectra** — no anomaly examples are used during training
- At evaluation time, the model scores both normal and ambient samples
- Ambient samples that score above a learned threshold are flagged as anomalies
- The true labels of the ambient samples are used **only to evaluate** model performance (AUC, recall, etc.) — they are never seen during training

This setup reflects a realistic deployment scenario: you know what normal polymers look like, but you cannot enumerate every possible ambient particle type in advance.

---

## Dataset

| File | Samples | Features | Label |
|------|---------|----------|-------|
| `data/all_spectra.csv` | 1,215 | 535 wavelengths (981–2969 nm) | 21 polymer classes |
| `data/ambient_spectra.csv` | 55 | 535 wavelengths (981–2969 nm) | AMB (ambient/anomaly) |

Both datasets are **z-score normalised** (mean = 0, std = 1) — no additional preprocessing is required before modelling. There are no missing values in either dataset.

`all_spectra.csv` is class-imbalanced: the three most common classes (Polyethylene: 297, Polypropylene: 213, Polystyrene: 188) account for ~57% of all samples. Several classes have fewer than 10 samples.

**Evaluation test set** is the concatenation of both files (1,270 samples total):
```
true_labels = [0, 0, ..., 0,  1, 1, ..., 1]
               ←— 1215 normal —→ ←— 55 anomaly —→
```

---

## Pipeline

Every model notebook follows the same workflow:

```
1. Load Data
       ↓
2. Train model on X_normal (1,215 samples) only — no labels used
       ↓
3. Compute anomaly score for all samples
   (normal + ambient)
       ↓
4. Set decision threshold
   = 95th percentile of normal training scores
   → by design, flags ~5% × 1215 ≈ 61 normal samples (FP = 61, fixed)
       ↓
5. Predict labels: score > threshold → anomaly (1), else normal (0)
       ↓
6. Evaluate: AUC, Recall, Confusion Matrix, ROC Curve
```

The anomaly score definition varies by model (see below). The threshold strategy and evaluation are identical across all models.

---

## Models

### k-Nearest Neighbours (kNN)
**Notebooks:** `models/knn_5_euclidean.ipynb`, `models/knn_hyperparameter_comparison.ipynb`

Fits on normal training samples. The anomaly score for a new sample is the mean Euclidean distance to its k nearest neighbours in the training set. Samples far from all normal neighbours receive a high score.

- **Best config:** k=3, Euclidean distance
- **Score:** mean distance to k nearest normal neighbours

### Isolation Forest
**Notebooks:** `models/isolation_forest.ipynb`, `models/isolation_forest_hyperparameter_comparison.ipynb`

Builds an ensemble of random trees that recursively partition the feature space. Anomalies are easier to isolate (require fewer splits), so they have shorter average path lengths. The anomaly score is the negated mean path length.

- **Best config:** n_estimators=100, max_features=0.25
- **Score:** negated mean isolation path length

### One-Class SVM (OC-SVM)
**Notebooks:** `models/one_class_svm.ipynb`, `models/one_class_svm_hyperparameter_comparison.ipynb`

Learns a nonlinear closed boundary around the normal data in a kernel-induced feature space (RBF kernel). The anomaly score is the negated signed distance from the decision boundary — positive scores indicate the sample lies outside the boundary.

- **Best config:** nu=0.05, gamma=0.001, kernel=rbf
- **Score:** negated decision function value

### Autoencoder
**Notebooks:** `models/autoencoder.ipynb`, `models/autoencoder_hyperparameter_comparison.ipynb`

A neural network trained to compress each spectrum to a low-dimensional latent representation and reconstruct it back. Trained only on normal spectra, it learns to reconstruct normal patterns efficiently. Anomalies, which share different spectral structure, reconstruct poorly.

- **Architecture:** 535 → 512 → 256 → 128 → 64 → 128 → 256 → 512 → 535
- **Score:** mean squared reconstruction error (MSE)

### PCA Reconstruction
**Notebook:** `models/pca_reconstruction.ipynb`

Projects each spectrum into the top principal components learned from normal data, then reconstructs it back to the original space. Normal spectra lie close to this linear subspace and reconstruct accurately. Anomalies that lie off the subspace reconstruct poorly.

- **Best config:** n_components=50 (captures 93.6% of normal variance)
- **Score:** mean squared reconstruction error (MSE)

---

## Results

Ranked by **Recall** (primary metric — fraction of the 55 ambient anomalies detected):

| Rank | Model | AUC | Recall | TP | FN | FP |
|------|-------|-----|--------|----|----|-----|
| 1 | Autoencoder | 0.972 | 0.818 | 45 | 10 | 61 |
| 2 | OC-SVM (nu=0.05, γ=0.001) | 0.947 | 0.727 | 40 | 15 | 61 |
| 3 | kNN (k=3, Euclidean) | 0.922 | 0.455 | 25 | 30 | 61 |
| 4 | Isolation Forest (n=100, feat=0.25) | 0.754 | 0.127 | 7 | 48 | 61 |
| 5 | PCA Reconstruction (n=50) | 0.795 | 0.073 | 4 | 51 | 61 |

**FP = 61 is identical across all models** — a fixed consequence of the p95 threshold, not a property of any individual model (5% × 1215 ≈ 61).

**Why Recall is the primary metric:** A missed anomaly (FN) is more costly than a false alarm (FP) in this application. Accuracy is misleading here because the dataset is 96%/4% normal/anomaly — a model that predicts "normal" for everything achieves 96% accuracy while catching zero anomalies.

**Key finding:** Models that learn a compact nonlinear representation of normal data (Autoencoder, OC-SVM) substantially outperform those relying on linear subspaces (PCA) or feature-wise statistics (Isolation Forest). The strong inter-feature correlations in Raman spectra favour models that learn the full joint structure of the normal class.

---

## Directory Structure

```
classical_models_all_spectra/
├── data/
│   ├── all_spectra.csv                          # 1,215 normal polymer spectra
│   ├── ambient_spectra.csv                      # 55 ambient anomaly spectra
│   └── explore_data.ipynb                       # data exploration and visualisation
│
├── models/
│   ├── knn_5_euclidean.ipynb                    # kNN baseline (k=5)
│   ├── knn_hyperparameter_comparison.ipynb      # kNN — k and distance metric sweep
│   ├── isolation_forest.ipynb                   # Isolation Forest baseline
│   ├── isolation_forest_hyperparameter_comparison.ipynb
│   ├── one_class_svm.ipynb                      # OC-SVM baseline
│   ├── one_class_svm_hyperparameter_comparison.ipynb
│   ├── autoencoder.ipynb                        # Autoencoder baseline
│   ├── autoencoder_hyperparameter_comparison.ipynb
│   ├── pca_reconstruction.ipynb                 # PCA reconstruction
│   └── save_model_comparison_table.py           # generates figures/model_comparison_table.png
│
└── figures/
    ├── model_comparison_table.png               # summary table of all models
    ├── knn_*.png
    ├── isolation_forest_*.png
    ├── ocsvm_*.png
    ├── autoencoder_*.png
    └── pca_*.png
```

Each model directory follows the same notebook structure: hyperparameters → imports → load data → train → score → threshold → distribution plot → evaluation → confusion matrix → ROC curve.
