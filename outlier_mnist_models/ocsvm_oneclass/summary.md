# One-Class SVM (OCSVM) Baseline on MNIST (Normal Digit = 0…9)

This notebook evaluates a **One-Class SVM (OCSVM)** anomaly detector on MNIST under a **one-vs-rest** protocol. The experiment is repeated **10 times**, where each digit `d ∈ {0,…,9}` is treated as the **normal** class and all other digits are treated as **outliers**. For each `d`, the notebook produces three outputs in **2×5 grids**:
1) **ROC curve + AUROC**, 2) **confusion matrix**, 3) **score distributions**.

---

## 1) What the code does (pipeline)

### A. Data construction (per normal digit)
For each `normal_digit = d`:

- **Train set (normal-only):** all training images with label `d`.
- **Validation set (mixed):** held-out normal images of `d` plus a randomly sampled, size-matched set of outliers from digits `≠ d`.  
  Validation is used to choose a threshold for binary decisions.
- **Test set (mixed):** MNIST test images labeled as:
  - `y=0` if digit is `d` (normal)
  - `y=1` otherwise (outlier)

All images are flattened (784-dim vectors).

---

### B. Representation (PCA on normals only)
If `PCA=True`, the code fits PCA using **only normal training samples** and projects all splits:
- Train normals → fit PCA
- Val/test → transform using the same PCA
This produces a lower-dimensional feature vector `z` (e.g., 50-D).

---

### C. Model fitting (OCSVM on normals only)
The One-Class SVM is trained **only on the normal training embeddings**:
- kernel = RBF
- `nu` controls an upper bound on the training fraction treated as outliers and a lower bound on support vectors
- `gamma` controls RBF kernel width (often `"scale"`)

The model learns a boundary around the normal data in feature space.

---

## 2) How the anomaly score is calculated (highlight)

### A. Raw OCSVM output: `decision_function`
For an input embedding `z`, scikit-learn computes:
\[
f(z) = \text{decision\_function}(z)
\]
Interpretation (standard for OCSVM):
- **larger \( f(z) \)** → sample is more confidently **inside** the learned normal region (more normal)
- **smaller \( f(z) \)** → sample is closer to / outside the boundary (less normal)

### B. Anomaly score used in the notebook
The notebook converts this into a score where **higher means more anomalous**:
\[
\boxed{\text{anomaly\_score}(z) = - f(z)}
\]

So:
- **low anomaly score** → likely normal
- **high anomaly score** → likely outlier

This scalar score is the single value used for ROC/AUROC, thresholding, confusion matrices, and histograms.

---

## 3) How the score is used for anomaly detection

### A. ROC curve + AUROC (test set)
- The ROC curve sweeps a threshold over the **continuous anomaly score**.
- For each threshold \( t \), the classifier is:
  \[
  \hat{y} = \mathbb{1}[\text{anomaly\_score} \ge t]
  \]
- AUROC summarizes how well the anomaly score ranks outliers above normals:
  - **AUROC ≈ 1.0**: strong separability
  - **AUROC ≈ 0.5**: near-random ranking
  - **AUROC < 0.5**: ranking is inverted relative to labels/score direction

---

### B. Threshold selection (validation set) and confusion matrix (test set)
To produce a single confusion matrix per digit, the notebook selects a threshold using **Youden’s J statistic** on validation:
- Compute ROC on `(y_val, anomaly_score_val)`
- Choose threshold maximizing:
  \[
  J = \text{TPR} - \text{FPR}
  \]

Then apply that threshold to test scores:
\[
\hat{y}_{test} = \mathbb{1}[\text{anomaly\_score}_{test} \ge t^*]
\]
and compute the confusion matrix with:
- **TN**: true normals classified as normal
- **FP**: true normals classified as outlier
- **FN**: true outliers classified as normal
- **TP**: true outliers classified as outlier

The value shown as `thr=...` above each confusion matrix is this selected \( t^* \) (in anomaly-score units).

---

### C. Score distribution plots
Each subplot shows the empirical distributions of the **anomaly score** on the test set:
- Blue histogram: `anomaly_score` for **normal** test samples (`y=0`)
- Orange histogram: `anomaly_score` for **outlier** test samples (`y=1`)
- Dashed vertical line: the chosen threshold \( t^* \)

Interpretation:
- Good separation: blue concentrated left (lower scores), orange shifted right (higher scores)
- Overlap indicates unavoidable trade-offs between FP and FN at any single threshold

---

## 4) What the outputs mean

### A. ROC/AUROC grid
Each subplot corresponds to one normal digit `d`. The reported AUROC is computed on the test set using the anomaly score `-decision_function`. Higher AUROC indicates that test outliers generally receive larger anomaly scores than test normals.

### B. Confusion matrix grid
Each confusion matrix shows test-set classification results after converting the continuous anomaly score into a binary decision using the validation-chosen threshold (`thr=...`). The counts quantify how many normals/outliers are accepted or rejected at that operating point.

### C. Score distribution grid
These plots directly visualize how the anomaly score behaves for normals vs outliers and how the threshold divides the two. The farther apart the two distributions are, the more reliable a single threshold is for anomaly detection.
