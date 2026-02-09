# kNN-based Anomaly Detection on MNIST

This document explains the **k-Nearest Neighbors (kNN) anomaly detection notebook** used to evaluate one-vs-rest outlier detection on the MNIST dataset. It covers the pipeline, the role of each data split, how scores and thresholds are computed, and how to interpret the resulting plots.

---

## 1. Problem Setup

We frame MNIST anomaly detection as a **one-vs-rest** task:

* Choose one digit (d \in {0,\dots,9}) as the **normal class**
* Treat **all other digits** as **outliers**
* Repeat the full experiment for each digit

This matches the standard **semi-supervised anomaly detection** setting: only normal data is used for training.

---

## 2. Data Splits and Their Roles

The dataset is split into three conceptually distinct parts **per normal digit**:

### Training set

* Contains **only normal samples** (digit = (d))
* Used to define what *normality* looks like
* No labels are needed (`y_train` is omitted by design)

### Validation set

* Balanced mix of:

  * normal samples (digit = (d))
  * outliers (digits (\neq d))
* Used **only** to select a decision threshold

### Test set

* Full MNIST test set
* Relabeled as:

  * `0` = normal (digit = (d))
  * `1` = outlier (digit (\neq d))
* Used **only** for final evaluation and visualization

This separation prevents data leakage and mirrors proper evaluation practice.

---

## 3. Feature Representation

* Images are normalized to the range ([0,1])
* Each image is flattened from (28 \times 28) into a **784-dimensional vector**
* All distances are computed in this raw pixel space

---

## 4. kNN Anomaly Detection Model

### Model fitting

For each normal digit (d):

* Fit a kNN structure using only **normal training samples**
* Distance metric: **Euclidean**
* Number of neighbors: **K = 10**

The model does **not** perform classification. It only stores normal examples.

---

### Anomaly score definition

For a sample (x), the anomaly score is:

[
\text{score}(x) = \frac{1}{K} \sum_{i=1}^{K} \lVert x - \text{NN}_i(x) \rVert_2
]

Interpretation:

* **Low score** → looks similar to normal digit
* **High score** → far from normal data → likely outlier

Scores are continuous, unbounded, and represent distances (not probabilities).

---

## 5. Threshold Selection: Youden’s J

To convert scores into binary predictions, we select a threshold using the **validation set**.

### Youden’s J statistic

[
J = \text{TPR} - \text{FPR}
]

* TPR: true positive rate (outliers correctly detected)
* FPR: false positive rate (normals incorrectly flagged)

The chosen threshold maximizes (J), i.e. the point on the ROC curve farthest from the diagonal.

Important properties:

* Independent of class imbalance
* Treats false positives and false negatives symmetrically
* Threshold is **digit-specific**

---

## 6. Use of the Test Set

The test set is **never** used for training or threshold selection.

It is used only for:

1. **ROC curves and AUROC** (threshold-free evaluation)
2. **Confusion matrices** (using validation-derived threshold)
3. **Score distribution plots** (diagnostic visualization)

---

## 7. Evaluation Plots and How to Interpret Them

### 7.1 ROC Curves

**Axes**:

* x-axis: False Positive Rate (FPR)
* y-axis: True Positive Rate (TPR)

**What they show**:

* How well anomaly scores rank normals vs outliers
* AUROC close to 1.0 indicates strong separability

**Key observation**:

* Digits like `0` and `1` achieve near-perfect AUROC
* Digits like `8` perform worse due to visual similarity with many other digits

---

### 7.2 Confusion Matrices

Computed on the **test set** using the **validation-derived Youden threshold**.

They show:

* True normals vs predicted normals/outliers
* True outliers vs predicted normals/outliers

Interpretation:

* High false negatives → outliers visually similar to the normal digit
* High false positives → normal digit has high intra-class variability

---

### 7.3 Score Distribution Plots

Each subplot shows two normalized histograms (test set only):

* **x-axis**: kNN anomaly score (mean distance)
* **y-axis**: probability density (area = 1)

Colors:

* Blue: true normal samples
* Orange: true outlier samples

The dashed vertical line is the **Youden threshold**.

These plots explain *why* confusion matrices look the way they do:

* Clear separation → strong performance
* Heavy overlap → unavoidable errors

---

## 8. Key Findings

* kNN anomaly detection works very well for compact digits (`0`, `1`, `6`)
* Performance degrades for visually diverse or ambiguous digits (`2`, `3`, `8`)
* AUROC alone is insufficient; confusion matrices reveal operating-point behavior
* Limitations stem from **raw pixel distance in high dimensions**, not bugs in the pipeline

---

## 9. Takeaways

* The notebook implements a **correct, leakage-free anomaly detection pipeline**
* Results are internally consistent across ROC curves, thresholds, and score distributions
* Digit-dependent behavior is expected and informative

This notebook provides a strong baseline for comparing more expressive models (e.g. KDE, GMM, One-Class SVM, learned embeddings).
