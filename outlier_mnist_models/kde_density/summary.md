# Kernel Density Estimation (KDE) Anomaly Detection on MNIST (Normal Digit = 0…9)

This notebook evaluates a **Kernel Density Estimation (KDE)**–based anomaly detector on MNIST using a **one-vs-rest** protocol. The experiment is repeated **10 times**, each time treating a different digit `d ∈ {0,…,9}` as the **normal** class and all other digits as **outliers**. For each `d`, the notebook produces three outputs arranged in **2×5 grids**:
1) **ROC curves with AUROC**,  
2) **confusion matrices**,  
3) **score distribution plots**.

---

## 1) Code and pipeline summary

### A. Data construction (per normal digit)
For each digit `d`:
- **Training set:** only MNIST training images labeled `d` (normal-only training).
- **Validation set:** a balanced mixture of:
  - held-out normal samples of digit `d`,
  - randomly sampled outliers from digits ≠ `d`.
- **Test set:** MNIST test images relabeled as:
  - `y = 0` if digit is `d` (normal),
  - `y = 1` otherwise (outlier).

All images are flattened into 784-dimensional vectors.

---

### B. Representation (PCA)
If `PCA=True`, PCA is:
- **fit only on normal training data** for digit `d`,
- then applied to validation and test data.
This yields a lower-dimensional embedding used by KDE.

---

### C. Density model (KDE)
For each digit `d`, a KDE is fit **only on the normal training embeddings**:
- kernel = Gaussian  
- bandwidth = 1.0  

The KDE models the probability density of the normal digit in feature space.

---

## 2) Anomaly score definition (highlighted)

For an embedded sample \( z \), KDE provides:
\[
\log p(z) = \text{kde.score\_samples}(z)
\]

The notebook defines the anomaly score as:
\[
\boxed{
\text{AnomalyScore}(z) = - \log p(z)
}
\]

**Interpretation:**
- **Low score** → high density under the normal KDE → sample looks normal.
- **High score** → low density under the normal KDE → sample looks anomalous.

This scalar anomaly score is the **sole quantity** used for all evaluation steps.

---

## 3) How the score is used for anomaly detection

### A. ROC curves and AUROC
- ROC curves are computed on the **test set** by sweeping a threshold over the anomaly score.
- Binary decision rule:
\[
\hat{y} = \mathbb{1}[\text{AnomalyScore} \ge t]
\]
- **AUROC** measures how well the score ranks outliers above normals, independent of any fixed threshold.

---

### B. Threshold selection and confusion matrices
- A single operating threshold is selected on the **validation set** using **Youden’s J statistic**:
\[
J = \text{TPR} - \text{FPR}
\]
- The threshold maximizing \( J \) is applied to the **test set** scores.
- Confusion matrices report counts of:
  - True Negatives (normal accepted),
  - False Positives (normal rejected),
  - False Negatives (outlier accepted),
  - True Positives (outlier rejected).

The value shown as `thr=...` above each confusion matrix is this validation-derived threshold.

---

### C. Score distribution plots
- Each subplot shows test-set anomaly score distributions:
  - Blue: normal digit `d`,
  - Orange: outliers (all other digits).
- The dashed vertical line indicates the selected threshold.
- The degree of separation between the two histograms explains both AUROC values and confusion matrix outcomes.

---

## 4) Interpretation of the outputs

### ROC / AUROC grid
- Each subplot corresponds to a different normal digit.
- Higher AUROC indicates stronger separability between normal and outlier score distributions for that digit.

### Confusion matrix grid
- Shows test-set classification performance at the validation-chosen operating point.
- Variability across digits reflects how well KDE models each digit’s distribution.

### Score distribution grid
- Provides a direct visualization of density-based anomaly detection:
  - well-separated distributions → reliable anomaly detection,
  - overlapping distributions → unavoidable false-positive/false-negative trade-offs.

---

**Summary:**  
For each digit, KDE is trained exclusively on normal data to estimate a probability density. Anomaly detection is driven entirely by the **negative log density** under this model. Higher values indicate lower likelihood of belonging to the normal class and are treated as stronger evidence of an anomaly across all reported plots and metrics.
