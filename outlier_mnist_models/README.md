# Outlier Detection Models on MNIST (one-digit normal, rest outliers)

## Setup
From `outlier_mnist_models/`:
- Create / activate a Python environment (recommended)
- Install deps:
  - `pip install -r requirements.txt`

## How to run
Each folder contains a standalone notebook:
- `knn_distance/knn_distance.ipynb`
- `gmm_density/gmm_density.ipynb`
- `kde_density/kde_density.ipynb`
- `ocsvm_oneclass/ocsvm_oneclass.ipynb`
- `vae/vae.ipynb`
- `ensemble/ensemble.ipynb`

Outliers are automatically defined as all digits not in `NORMAL_DIGITS`.

## Outputs (per notebook)
- Test AUROC
- ROC curve plot
- Confusion matrix (threshold chosen via Youden J on validation set)
- Score distribution histogram
