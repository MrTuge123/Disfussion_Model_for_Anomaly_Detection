# Variational Autoencoder (VAE) Anomaly Detection on MNIST (Normal Digit = 0…9)

This experiment evaluates a **Variational Autoencoder (VAE)** as an anomaly detector on MNIST using a **one-vs-rest** protocol. The same pipeline is run **10 times**, where each digit `d ∈ {0,…,9}` is treated as the **normal** class and all other digits are treated as **outliers**. For each `d`, the notebook produces three outputs arranged in **2×5 grids**:
1) **ROC curves with AUROC**,  
2) **confusion matrices**,  
3) **score distribution plots**.

---

## 1) Code and pipeline overview

### A. Data construction (per normal digit)
For each normal digit `d`:

- **Training set:** only MNIST training images labeled `d` (normal-only training).
- **Validation set:** a mix of:
  - held-out normal images (`d`),
  - a size-matched, randomly sampled set of outliers (digits ≠ `d`) from the training set.
- **Test set:** the MNIST test set, relabeled as:
  - `y = 0` if digit is `d` (normal),
  - `y = 1` otherwise (outlier).

All images are flattened into 784-dimensional vectors and loaded via PyTorch `DataLoader`s.

---

### B. Model training (VAE on normals only)
For each digit `d`, a **new VAE** is trained **from scratch** using only normal training samples:

- Encoder → latent mean and log-variance (`μ`, `log σ²`)
- Reparameterization trick to sample latent `z`
- Decoder reconstructs the input
- Latent dimension = 16
- Trained for 5 epochs

The VAE therefore models the distribution of **normal digit `d` only**.

---

## 2) Anomaly score definition (highlighted)

For a given input image \( x \), the VAE defines the anomaly score as the **negative Evidence Lower Bound (negative ELBO)** computed per sample.

### A. Per-sample loss components

1. **Reconstruction term** (binary cross-entropy, summed over pixels):
\[
\text{Recon}(x) = \sum_{i=1}^{784} \text{BCE}(x_i, \hat{x}_i)
\]

2. **KL divergence term** (regularization of latent space):
\[
\text{KL}(x) = -\frac{1}{2} \sum_j \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
\]

### B. Final anomaly score
\[
\boxed{
\text{AnomalyScore}(x) = \text{Recon}(x) + \beta \cdot \text{KL}(x)
}
\]
with \( \beta = 1.0 \) in this experiment.

**Interpretation:**
- **Low score** → image is well reconstructed and lies near the learned latent prior → *looks normal*.
- **High score** → poor reconstruction and/or atypical latent encoding → *looks anomalous*.

This scalar anomaly score is the **only quantity** used for ROC curves, AUROC, thresholding, confusion matrices, and score distribution plots.

---

## 3) How the score is used for anomaly detection

### A. ROC curves and AUROC
- The ROC curve is computed on the **test set** by sweeping a threshold over the continuous anomaly score.
- The decision rule is:
\[
\hat{y} = \mathbb{1}[\text{AnomalyScore} \ge t]
\]
- **AUROC** summarizes how well the anomaly score ranks outliers above normals, independent of any single threshold.

---

### B. Threshold selection and confusion matrices
- A **single threshold** is selected using **Youden’s J statistic** on the **validation set**:
\[
J = \text{TPR} - \text{FPR}
\]
- The threshold that maximizes \( J \) is then applied to **test scores**.
- Confusion matrices report counts of:
  - True Negatives (normal accepted),
  - False Positives (normal rejected),
  - False Negatives (outlier accepted),
  - True Positives (outlier rejected).

The value shown as `thr=...` above each confusion matrix is this validation-derived threshold.

---

### C. Score distribution plots
- Each subplot shows the empirical distribution of the anomaly score on the test set:
  - Blue: normal digit `d`,
  - Orange: outliers (all other digits).
- The dashed vertical line marks the chosen threshold.
- Separation between the two distributions explains both the AUROC values and the confusion matrix behavior.

---

## 4) Interpretation of the outputs

### ROC / AUROC grid
- Each subplot corresponds to a different choice of normal digit.
- Higher AUROC indicates stronger separation between normal and outlier scores.
- Performance varies by digit, reflecting how well the VAE can model that digit’s structure.

### Confusion matrix grid
- Shows test-set classification performance at the validation-chosen operating point.
- Differences across digits reflect varying overlap in score distributions.

### Score distribution grid
- Visualizes the underlying reason for success or failure:
  - well-separated distributions → reliable anomaly detection,
  - overlapping distributions → unavoidable trade-offs between false positives and false negatives.

---

**Summary:**  
The VAE is trained only on normal data for each digit. Anomaly detection is driven entirely by the **negative ELBO (reconstruction error + KL divergence)**. Higher negative ELBO indicates poorer fit under the normal model and is treated as stronger evidence of an anomaly across all reported metrics and plots.
