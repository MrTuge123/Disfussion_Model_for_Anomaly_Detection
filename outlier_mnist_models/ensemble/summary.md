# Ensemble Outlier Detection on MNIST (Normal Digit = 0…9)

This notebook cell runs the **same anomaly-detection pipeline 10 times**, each time treating **one digit** as the *normal* class and all other digits as *outliers*. For each normal digit, it produces:

- a **ROC curve** and **AUROC**
- a **confusion matrix** (after choosing a decision threshold on a validation set)
- a **score distribution plot** (normal vs outlier scores)

---

## 1) What the code is doing (end-to-end pipeline)

### Step A — Define the “one-vs-rest” anomaly task (10 runs)
For each `normal_digit ∈ {0,1,...,9}`:

- **Training set:** uses **only images of that digit** (normal-only training).
- **Validation set:** uses a **mix** of
  - normals from the same digit, plus
  - a matched number of sampled outliers from other digits.
- **Test set:** uses the full MNIST test set, labeled as:
  - `y=0` if the image is the normal digit
  - `y=1` otherwise (outlier)

This yields 10 separate anomaly-detection problems: “is this digit *d* or not?”

---

### Step B — Fit three unsupervised detectors on normal-only training data
All models train **only on normal data**, as standard in outlier detection.

1. **kNN distance score**
   - Fit kNN on flattened normal images.
   - Score each sample by the **mean distance** to its `K` nearest neighbors.
   - Interpretation: **larger distance ⇒ more anomalous**.

2. **PCA → Gaussian Mixture Model (GMM)**
   - Compress images with PCA to `PCA_DIM` dimensions.
   - Fit a GMM on PCA-transformed normal embeddings.
   - Score each sample as **negative log-likelihood**:  
     `score = -log p(x)`
   - Interpretation: **lower likelihood ⇒ more anomalous**.

3. **PCA → One-Class SVM (OCSVM)**
   - Fit One-Class SVM on PCA embeddings.
   - Use `-decision_function(x)` as the anomaly score.
   - Interpretation: **more negative / smaller margin ⇒ more anomalous**.

Each detector outputs a 1D anomaly score per sample.

---

### Step C — Standardize and ensemble the detector scores
Raw detector scores live on different scales, so the code:

1. Stacks validation scores into a matrix `S_val` of shape `(n_val, 3)`  
2. Computes per-detector mean/std on validation: `μ, σ`
3. Converts both validation and test scores to **z-scores**:
   - `S_z = (S - μ) / (σ + eps)`
4. Defines the **ensemble score** as the **mean of the 3 z-scores**:
   - `ensemble_score(x) = mean(z_knn, z_gmm, z_ocsvm)`

This makes scores comparable across detectors and reduces sensitivity to one detector dominating due to scale.

---

### Step D — Threshold selection (validation) + evaluation (test)
- **ROC curve / AUROC:** computed on the **test set**, using the continuous ensemble score.  
  AUROC is **threshold-free**.

- **Confusion matrix:** requires a threshold. The code chooses it using **Youden’s J statistic** on validation:
  - Compute ROC on `(y_val, score_val)`
  - Choose threshold maximizing `J = TPR - FPR`

Then apply that threshold to **test scores** to get predicted labels and compute the confusion matrix.

---

## 2) How to read the outputs

## A) ROC Curves (2×5 grid)
Each subplot corresponds to a normal digit. The curve shows the **TPR vs FPR** tradeoff across all thresholds.

- **AUROC close to 1.0** means normals and outliers are well-separated by the score.
- In your plots:
  - **Normal=1** is extremely strong (**AUROC ≈ 0.998**).
  - **Normal=0** is also strong (**≈ 0.980**).
  - **Normal=2 and 5** are noticeably harder (**≈ 0.866, 0.832**).

**Interpretation:** the pipeline can learn a tight “normal manifold” for some digits (like 1), but for digits with higher shape variability or overlap with other digits (often 2, 5), the detectors have more ambiguity.

---

## B) Confusion Matrices (2×5 grid)
Each confusion matrix uses the **Youden threshold** selected on validation, then applied on test.

The axes are:
- Rows = true label (Normal=0 row, Outlier=1 row)
- Columns = predicted label (Normal=0 col, Outlier=1 col)

So:
- Top-left = **True Negatives (TN)**: correct normals
- Top-right = **False Positives (FP)**: normals flagged as outliers
- Bottom-left = **False Negatives (FN)**: outliers incorrectly accepted as normal
- Bottom-right = **True Positives (TP)**: correct outliers

### What your confusion matrices show
A common pattern across digits:
- **TP is very large** (many outliers correctly detected),
- but there can be substantial **FN** for difficult normal digits.

Examples visible in your plots:
- **Normal=1:** very strong
  - TN ~ 1100, FP ~ 35 (few normal mistakes)
  - FN ~ 61, TP ~ 8804 (few outliers missed)
- **Normal=2:** much weaker
  - TN ~ 839, FP ~ 193
  - FN ~ 2292 (many outliers misclassified as normal)
  - TP ~ 6676

That matches the AUROC story: normal=2 is harder, so even the “best” threshold found on validation still yields many missed outliers.

### Why FN can be large even when AUROC is decent
AUROC measures ranking quality, not a single operating point. If the score distributions overlap, you can still get a decent AUROC but no threshold gives simultaneously low FP and low FN. Youden’s J picks a balanced ROC point, but your application might want:
- fewer FN (catch more outliers) at the cost of more FP, or
- fewer FP at the cost of more FN.

---

## C) Score Distributions (2×5 grid)
These plots show the **density of ensemble scores** for:
- normals (blue)
- outliers (orange)
and the dashed vertical line is the chosen threshold.

**Desired shape:** blue mass left of the threshold, orange mass right of it.

Your plots show:
- For **normal=1**, the blue and orange distributions are very separated → high AUROC and clean confusion matrix.
- For **normal=2 and 5**, there is more overlap → more FN and/or FP.

This plot is the most intuitive explanation of *why* each digit’s performance differs:
- More overlap ⇒ the models cannot clearly separate normal-vs-rest.

---

## 3) Key takeaways from your results

1. **Digit choice matters a lot.**  
   Normal digits like **1 and 0** behave as “easy normals” for this ensemble; digits like **5 and 2** are harder.

2. **AUROC and confusion matrix tell different things.**
   - AUROC: “are scores generally higher for outliers than normals?”
   - confusion matrix: “at this chosen threshold, how many errors do we make?”

3. **Thresholding is the knob that changes FP vs FN tradeoff.**  
   Youden’s J is a reasonable default, but if your goal is “catch as many outliers as possible,” you’d likely shift the threshold left (accept higher FP).

---