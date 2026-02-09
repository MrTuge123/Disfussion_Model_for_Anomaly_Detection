"""
Common utilities for MNIST one-class / one-vs-rest outlier detection experiments.

Conventions:
- y_true: 1 = outlier (anomaly), 0 = normal
- scores: higher = more anomalous
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

def set_global_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)

def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    return float(roc_auc_score(y_true, scores))

def plot_roc_curve(y_true: np.ndarray, scores: np.ndarray, title: str = "ROC Curve") -> None:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = compute_auroc(y_true, scores)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} (AUROC={auc:.4f})")
    plt.grid(True, alpha=0.3)
    plt.show()

def choose_threshold_youden(y_true_val: np.ndarray, scores_val: np.ndarray) -> float:
    """
    Choose threshold that maximizes Youden's J statistic = TPR - FPR on validation data.
    Returns a threshold on 'scores' (higher = more anomalous).
    """
    y_true_val = np.asarray(y_true_val).astype(int)
    scores_val = np.asarray(scores_val).astype(float)
    fpr, tpr, thresholds = roc_curve(y_true_val, scores_val)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx])

def confusion_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float):
    """
    Predict outlier if score >= threshold.
    Returns confusion matrix and dict with TN, FP, FN, TP.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return cm, {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

def plot_confusion_matrix(cm, title: str = "Confusion Matrix") -> None:
    cm = np.asarray(cm)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Normal(0)", "Outlier(1)"])
    plt.yticks(tick_marks, ["Normal(0)", "Outlier(1)"])
    # annotate
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

def plot_score_hist(scores: np.ndarray, y_true: np.ndarray, title: str = "Anomaly Score Distribution") -> None:
    scores = np.asarray(scores).astype(float)
    y_true = np.asarray(y_true).astype(int)
    plt.figure()
    plt.hist(scores[y_true==0], bins=50, alpha=0.6, label="Normal")
    plt.hist(scores[y_true==1], bins=50, alpha=0.6, label="Outlier")
    plt.title(title)
    plt.xlabel("Anomaly score (higher = more anomalous)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
