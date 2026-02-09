# GMM Density Baseline on MNIST (Normal Digit = 0…9)

This experiment evaluates a **Gaussian Mixture Model (GMM) density estimator** as an outlier detector on MNIST under a **one-vs-rest** setup. It runs the same pipeline **10 times**, where each digit `d ∈ {0,…,9}` is treated as the **normal** class and all other digits are treated as **outliers**. For each `d`, the notebook produces three outputs arranged in **2×5 grids**:
1) **ROC curve + AUROC**, 2) **confusion matrix**, 3) **score distributions**.

---

## 1) What the code does (high-level)

### A. Data setup (repeated for each normal digit)
For each `normal_digit = d`:

- **Train set (normal-only):**  
  Uses only training images whose label is `d`.  
  This is standard in unsupervised anomaly detection: the model learns “what normal looks like.”

- **Validation set (mixed):**  
  Combines:
  - normals = digit `d` (held-out from the normal pool),
  - outliers = randomly sampled images from digits `{0,…,9} \ {d}` (matched count to normals).  
  Validation is used to **choose a threshold** for the confusion matrix.

- **Test set (mixed):**  
  Uses the MNIST test set, labeled as:
  - `y=0` if digit is `d` (normal),
  - `y=1` otherwise (outlier).

---

### B. Feature pipeline: PCA (fit only on normals)
Each image is flattened to a vector (784 dims), then optionally reduced using:

- **PCA** to `PCA_DIM=50` (in your run, `PCA=True`)

PCA is fit **only on normal training data**, to avoid leaking outlier structure into the representation.

---

### C. Model: GMM fit on normal-only embeddings
A **Gaussian Mixture Model** is fit on the PCA embeddings of normal training samples:

- `n_components = 10`
- `covariance_type = "diag"`

This learns a density model of the normal class in the reduced feature space.

---

### D. Scoring rule: negative log-likelihood (anomaly score)
For a sample `x`, the GMM assigns a likelihood `p(x)`. The notebook uses:

- **Anomaly score:** `score(x) = -log p(x)`

Interpretation:
- **low score** ⇒ high likelihood under the normal model ⇒ “looks normal”
- **high score** ⇒ low likelihood under the normal model ⇒ “looks anomalous / outlier”

---

### E. Evaluation + thresholding
Two types of evaluation are produced:

1) **ROC + AUROC (test set)**  
   Uses the continuous anomaly score and sweeps all thresholds.  
   AUROC is **threshold-free** and measures how well scores rank outliers above normals.

2) **Confusion matrix (test set at one chosen threshold)**  
   A single threshold is chosen **from the validation set** using **Youden’s J statistic**:
   - Find threshold that maximizes `J = TPR − FPR` on validation  
   Then apply that threshold to test to get predicted labels and compute the confusion matrix.

---

## 2) How to interpret the outputs

## A) ROC curves + AUROC (2×5 grid)
Each subplot corresponds to one choice of normal digit `d`.  
- The ROC curve plots **TPR vs FPR** for all possible thresholds.
- The dashed diagonal is random guessing.
- **Higher curve / larger AUROC** means the detector better separates normals from outliers.

In your plots:
- **Normal=1** is easiest (AUROC ≈ 0.997): digit “1” has a relatively consistent shape.
- **Normal=2** is hardest (AUROC ≈ 0.858): the normal manifold overlaps more with other digits, so likelihoods are less separable.
- Most digits are in the **0.90–0.97** range, indicating the density model is reasonably effective but digit-dependent.

**Key point:** AUROC reflects *ranking quality* across thresholds. It does not commit to a single operating point.

---

## B) Confusion matrices (2×5 grid)
Each confusion matrix uses the **validation-chosen threshold** (shown as `thr=...`) and reports test-set counts:

- Rows = true label (`Normal(0)`, `Outlier(1)`)
- Cols = predicted label (`Normal(0)`, `Outlier(1)`)

So the entries are:
- **TN** (top-left): normals correctly accepted
- **FP** (top-right): normals incorrectly flagged as outliers
- **FN** (bottom-left): outliers incorrectly accepted as normal
- **TP** (bottom-right): outliers correctly flagged

What your matrices show:
- Many digits achieve high TP (most outliers detected),
- but harder normals (notably **digit 2**) have a larger FN count (more outliers “look normal” under the model).

**Important:** these confusion matrices depend strongly on the thresholding strategy. Youden’s J targets a balanced ROC point; if your use-case prioritizes catching outliers (low FN), you’d typically choose a *more aggressive* threshold.

---

## C) Score distributions (2×5 grid)
Each subplot overlays the density of anomaly scores on the test set:
- Blue = normal scores (digit `d`)
- Orange = outlier scores (all other digits)
- Dashed vertical line = selected threshold (Youden on validation)

Interpretation:
- If blue and orange are well separated, there exists a threshold with low FP and low FN.
- If they overlap heavily, any threshold must trade FP vs FN.

In your plots:
- For **Normal=1**, the blue distribution is concentrated at lower scores and the orange distribution is clearly shifted right → strong separability.
- For **Normal=2**, overlap is larger → consistent with lower AUROC and larger FN.

These plots are the “mechanistic explanation” behind both AUROC and the confusion matrix:
- **More overlap ⇒ worse separability ⇒ lower AUROC and/or more errors at any fixed threshold.**

---

## 3) Takeaways from this run

1) **Digit choice drives difficulty.**  
   Some digits (e.g., 1, 0) form a tighter normal distribution than others (e.g., 2), making density modeling easier.

2) **AUROC vs confusion matrix tell different stories.**  
   - AUROC: “does the score rank outliers higher than normals?”
   - Confusion matrix: “at this specific threshold, how many mistakes do we make?”

3) **GMM density is a reasonable baseline, but not uniformly robust.**  
   It performs well for simple, consistent digits, but struggles more when the normal digit has high intra-class variation or overlaps with other digits in feature space.

---