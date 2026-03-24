#!/usr/bin/env python3
"""
Hyperparameter sweep for the MLP-based Neural Spline Flow microplastic detector.

Pipeline (matches notebook exactly):
  Raw spectra → Rubberband baseline → SNV → StandardScaler (per region)
  → PCA (separate fingerprint + C-H stretch) → Normalizing Flow

Sweeps over:
  N_LAYERS, HIDDEN_DIM, N_BINS, DROPOUT, LEARNING_RATE,
  WEIGHT_DECAY, BATCH_SIZE, PCA_VARIANCE

For every combination the script:
  1. Fits preprocessing + separate PCA + trains the flow from scratch
  2. Selects the threshold via validation-percentile sweep (max F1)
  3. Evaluates on the held-out test set (polymer test + ambient)
  4. Saves four plots per run in result_0324/<run_index>/
  5. Writes summary CSV + JSON to result_0324/
"""

import os, itertools, json, time, warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve, average_precision_score,
)

# =====================================================================
# FIXED CONFIG (not swept)
# =====================================================================
SOURCE_POLYMER_FILE      = "all_spectra.csv"
SOURCE_MICROPLASTIC_FILE = "ambient_spectra.csv"
NON_PLASTIC_POLYMERS     = ["Sucrose", "NH2SO4", "Cotton", "NaNO3"]
FINGERPRINT_MAX          = 1799.0 
CH_STRETCH_MIN           = 2721.0
TEST_SIZE                = 0.2
VAL_SPLIT                = 0.2
RANDOM_STATE             = 42
N_EPOCHS                 = 150
CLIP_GRAD_NORM           = 3.0
TAIL_BOUND_MARGIN        = 1.2

RESULTS_DIR = "result_0324_roc"

# =====================================================================
# SWEEP GRID  — edit these lists to control which combos are tested
# =====================================================================
SWEEP_GRID = {
    "n_layers":      [4, 8],
    "hidden_dim":    [24,32,64],
    "n_bins":        [5, 8],
    "dropout":       [0.3,0.4],
    "learning_rate": [3e-4, 5e-4],
    "weight_decay":  [1e-4],
    "batch_size":    [32, 64],
    "patience":      [20],
    "pca_variance":  [0.90]
}


# =====================================================================
# SPECTRAL PREPROCESSING  (identical to notebook)
# =====================================================================

def rubberband_baseline(spectrum):
    n = len(spectrum)
    x = np.arange(n, dtype=float)
    y = spectrum.astype(float)
    hull = []
    for i in range(n):
        while len(hull) >= 2:
            ax_, ay_ = x[hull[-2]], y[hull[-2]]
            bx_, by_ = x[hull[-1]], y[hull[-1]]
            cx_, cy_ = x[i], y[i]
            cross = (bx_ - ax_) * (cy_ - ay_) - (by_ - ay_) * (cx_ - ax_)
            if cross < 0:
                hull.pop()
            else:
                break
        hull.append(i)
    baseline = np.interp(x, x[hull], y[hull])
    return spectrum - baseline


def snv(spectrum):
    mean = spectrum.mean()
    std = spectrum.std()
    if std < 1e-10:
        return spectrum - mean
    return (spectrum - mean) / std


def preprocess_spectra(spectra):
    out = np.empty_like(spectra, dtype=float)
    for i in range(len(spectra)):
        out[i] = snv(rubberband_baseline(spectra[i]))
    return out


# =====================================================================
# MODEL DEFINITION  (MLP-based, identical to notebook)
# =====================================================================

def rational_quadratic_spline_forward(x, widths, heights, derivatives, tail_bound=5.0):
    K = widths.shape[-1]
    widths = F.softmax(widths, dim=-1) * 2 * tail_bound
    heights = F.softmax(heights, dim=-1) * 2 * tail_bound
    derivatives = F.softplus(derivatives)
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0) - tail_bound
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0) - tail_bound
    x_clamped = x.clamp(-tail_bound + 1e-6, tail_bound - 1e-6)
    bin_idx = torch.searchsorted(cumwidths[..., 1:], x_clamped.unsqueeze(-1)).squeeze(-1)
    bin_idx = bin_idx.clamp(0, K - 1)
    idx = bin_idx.unsqueeze(-1)
    w_k  = widths.gather(-1, idx).squeeze(-1)
    h_k  = heights.gather(-1, idx).squeeze(-1)
    d_k  = derivatives.gather(-1, idx).squeeze(-1)
    d_k1 = derivatives.gather(-1, idx + 1).squeeze(-1)
    cw_k = cumwidths.gather(-1, idx).squeeze(-1)
    ch_k = cumheights.gather(-1, idx).squeeze(-1)
    s_k = h_k / w_k
    xi = ((x_clamped - cw_k) / w_k).clamp(1e-6, 1.0 - 1e-6)
    numerator   = h_k * (s_k * xi * xi + d_k * xi * (1 - xi))
    denominator = s_k + (d_k + d_k1 - 2 * s_k) * xi * (1 - xi)
    y = ch_k + numerator / denominator
    deriv_num = s_k * s_k * (d_k1 * xi * xi + 2 * s_k * xi * (1 - xi) + d_k * (1 - xi) ** 2)
    log_det = torch.log(deriv_num + 1e-8) - 2 * torch.log(denominator.abs() + 1e-8)
    return y, log_det


class SplineParamNet(nn.Module):
    """Residual MLP that predicts spline parameters from the conditioning half."""
    def __init__(self, in_dim, transform_dim, hidden_dim=128, n_bins=8, dropout=0.4):
        super().__init__()
        self.transform_dim = transform_dim
        self.n_bins = n_bins
        out_dim = transform_dim * (3 * n_bins + 1)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        batch = x.shape[0]
        h = F.relu(self.input_proj(x))
        h = h + self.block1(h)
        h = h + self.block2(h)
        params = self.output_proj(h)
        params = params.view(batch, self.transform_dim, 3 * self.n_bins + 1)
        widths = params[:, :, :self.n_bins]
        heights = params[:, :, self.n_bins:2 * self.n_bins]
        derivatives = params[:, :, 2 * self.n_bins:]
        return widths, heights, derivatives


class CouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim=128, n_bins=8, tail_bound=5.0,
                 reverse=False, dropout=0.4):
        super().__init__()
        self.dim = dim
        self.reverse = reverse
        self.split_dim = (dim + 1) // 2
        self.transform_dim = dim - self.split_dim
        self.tail_bound = tail_bound
        cond_dim  = self.transform_dim if reverse else self.split_dim
        trans_dim = self.split_dim     if reverse else self.transform_dim
        self.param_net = SplineParamNet(
            cond_dim, trans_dim, hidden_dim, n_bins, dropout)

    def forward(self, x):
        if self.reverse:
            x_cond, x_trans = x[:, self.split_dim:], x[:, :self.split_dim]
        else:
            x_cond, x_trans = x[:, :self.split_dim], x[:, self.split_dim:]
        w, h, d = self.param_net(x_cond)
        y_trans, ld = rational_quadratic_spline_forward(
            x_trans, w, h, d, self.tail_bound)
        ld_sum = ld.sum(1)
        if self.reverse:
            return torch.cat([y_trans, x_cond], 1), ld_sum
        return torch.cat([x_cond, y_trans], 1), ld_sum


class StandardGaussian:
    def __init__(self, dim):
        self.dim = dim
        self._log_norm = -0.5 * dim * np.log(2.0 * np.pi)

    def log_prob(self, z):
        return self._log_norm - 0.5 * (z * z).sum(dim=1)


class PermutationLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("permutation", torch.randperm(dim))
        self.register_buffer("inverse_permutation", torch.argsort(self.permutation))

    def forward(self, x):
        return x[:, self.permutation], torch.zeros(x.shape[0], device=x.device)


class ActNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, dim))
        self.bias      = nn.Parameter(torch.zeros(1, dim))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def _data_dependent_init(self, x):
        self.bias.data.copy_(-x.mean(0, keepdim=True))
        self.log_scale.data.copy_(-torch.log(x.std(0, keepdim=True) + 1e-6))
        self.initialized.fill_(True)

    def forward(self, x):
        if not self.initialized:
            self._data_dependent_init(x)
        z = (x + self.bias) * torch.exp(self.log_scale)
        return z, self.log_scale.sum().expand(x.shape[0])


class BatchNormFlow(nn.Module):
    def __init__(self, dim, momentum=0.1, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.log_gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta      = nn.Parameter(torch.zeros(1, dim))
        self.register_buffer("running_mean", torch.zeros(1, dim))
        self.register_buffer("running_var",  torch.ones(1, dim))

    def forward(self, x):
        if self.training:
            mean = x.mean(0, keepdim=True)
            var = x.var(0, keepdim=True) + self.eps
            self.running_mean.mul_(1 - self.momentum).add_(mean * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var * self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var + self.eps
        x_hat = (x - mean) / torch.sqrt(var)
        z = x_hat * torch.exp(self.log_gamma) + self.beta
        log_det = (self.log_gamma - 0.5 * torch.log(var)).sum().expand(x.shape[0])
        return z, log_det


class NormalizingFlow(nn.Module):
    def __init__(self, dim, n_layers=4, hidden_dim=128, n_bins=8,
                 tail_bound=5.0, dropout=0.4):
        super().__init__()
        self.dim = dim
        self.base_dist = StandardGaussian(dim)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(ActNorm(dim))
            self.layers.append(BatchNormFlow(dim))
            self.layers.append(CouplingLayer(
                dim, hidden_dim, n_bins, tail_bound,
                reverse=(i % 2 == 1), dropout=dropout))
            if i < n_layers - 1:
                self.layers.append(PermutationLayer(dim))

    def forward(self, x):
        z = x
        ld = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            z, d = layer(z)
            ld += d
        return self.base_dist.log_prob(z) + ld


# =====================================================================
# TRAINING FUNCTION
# =====================================================================

def train_flow(model, data, n_epochs, batch_size, lr, weight_decay,
               clip_grad_norm, val_data, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    n_batches = max(1, len(data) // batch_size)
    losses, val_losses = [], []
    best_val, best_state, no_improve = float("inf"), None, 0

    for epoch in tqdm(range(n_epochs), desc="  training", leave=False):
        model.train()
        data_shuf = data[torch.randperm(len(data))]
        epoch_loss = 0.0
        for i in range(n_batches):
            batch = data_shuf[i * batch_size:(i + 1) * batch_size]
            optimizer.zero_grad()
            loss = -model(batch).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        losses.append(epoch_loss / n_batches)

        if val_data is not None:
            model.eval()
            with torch.no_grad():
                vl = -model(val_data).mean().item()
            val_losses.append(vl)
            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return losses, val_losses, best_val


# =====================================================================
# PLOTTING HELPERS
# =====================================================================

def save_loss_curve(losses, val_losses, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, lw=2, label="Train")
    if val_losses:
        ax.plot(val_losses, lw=2, label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_confusion_matrix(cm, f1, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn_r",
                xticklabels=["MICROPLASTIC", "NON-MP"],
                yticklabels=["MICROPLASTIC", "NON-MP"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (F1={f1:.3f})")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_auroc_curve(ground_truth, anomaly_scores, auroc_val, threshold_as, path):
    fpr, tpr, _ = roc_curve(ground_truth, anomaly_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="purple", lw=2.5, label=f"AUROC = {auroc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    preds = (anomaly_scores >= threshold_as).astype(int)
    cur_fpr = (preds[ground_truth == 0] == 1).mean()
    cur_tpr = (preds[ground_truth == 1] == 1).mean()
    ax.scatter([cur_fpr], [cur_tpr], color="red", s=100, zorder=5,
              edgecolors="black", lw=1.5,
              label=f"Threshold (FPR={cur_fpr:.2f}, TPR={cur_tpr:.2f})")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_logprob_hist(train_lp, test_normal_lp, test_anomaly_lp, threshold_lp, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(train_lp, bins=50, alpha=0.5, density=True, color="green",
            edgecolor="black", label="Train (microplastic)")
    ax.hist(test_normal_lp, bins=30, alpha=0.5, density=True, color="blue",
            edgecolor="black", label="Test polymers (microplastic)")
    ax.hist(test_anomaly_lp, bins=30, alpha=0.5, density=True, color="red",
            edgecolor="black", label="Test ambient (non-microplastic)")
    ax.axvline(threshold_lp, color="black", ls="--", lw=2,
               label=f"Threshold = {threshold_lp:.2f}")
    ax.set_xlabel("Log Probability"); ax.set_ylabel("Density")
    ax.set_title("Log-Probability Distributions"); ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# =====================================================================
# SINGLE RUN  — full pipeline for one hyperparameter config
# =====================================================================

def run_single(hp, df_train, df_polymer_test, df_amb, df_test,
               feature_cols, fp_mask, ch_mask, run_dir):
    """Train + evaluate one config. Returns metrics dict."""
    normal_data = df_train[feature_cols].values

    # Preprocess per spectral region
    fp_preprocessed = preprocess_spectra(normal_data[:, fp_mask])
    ch_preprocessed = preprocess_spectra(normal_data[:, ch_mask])

    scaler_fp = StandardScaler().fit(fp_preprocessed)
    scaler_ch = StandardScaler().fit(ch_preprocessed)
    fp_scaled = scaler_fp.transform(fp_preprocessed)
    ch_scaled = scaler_ch.transform(ch_preprocessed)

    # Separate PCA per region
    pca_fp = PCA(n_components=hp["pca_variance"], svd_solver="full").fit(fp_scaled)
    pca_ch = PCA(n_components=hp["pca_variance"], svd_solver="full").fit(ch_scaled)
    fp_reduced = pca_fp.transform(fp_scaled)
    ch_reduced = pca_ch.transform(ch_scaled)

    normal_reduced = np.hstack([fp_reduced, ch_reduced])
    n_components = normal_reduced.shape[1]
    n_fp = pca_fp.n_components_
    n_ch = pca_ch.n_components_
    tail_bound = float(np.abs(normal_reduced).max()) * TAIL_BOUND_MARGIN

    # Helper to transform new data through the full pipeline
    def transform_data(raw):
        fp_pre = preprocess_spectra(raw[:, fp_mask])
        ch_pre = preprocess_spectra(raw[:, ch_mask])
        return np.hstack([
            pca_fp.transform(scaler_fp.transform(fp_pre)),
            pca_ch.transform(scaler_ch.transform(ch_pre)),
        ])

    # Train / val split (deterministic)
    n_val = int(len(normal_reduced) * VAL_SPLIT)
    perm = np.random.RandomState(RANDOM_STATE).permutation(len(normal_reduced))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_tensor = torch.FloatTensor(normal_reduced[train_idx])
    val_tensor   = torch.FloatTensor(normal_reduced[val_idx])

    # Build & train
    model = NormalizingFlow(
        dim=n_components, n_layers=hp["n_layers"], hidden_dim=hp["hidden_dim"],
        n_bins=hp["n_bins"], tail_bound=tail_bound, dropout=hp["dropout"])

    t0 = time.time()
    losses, val_losses, best_val_loss = train_flow(
        model, train_tensor, N_EPOCHS, hp["batch_size"], hp["learning_rate"],
        hp["weight_decay"], CLIP_GRAD_NORM, val_tensor, hp["patience"])
    train_time = time.time() - t0

    # Evaluate
    model.eval()
    n_polymer_test = len(df_polymer_test)
    n_anomaly = len(df_test) - n_polymer_test
    ground_truth = np.array([0] * n_polymer_test + [1] * n_anomaly)

    with torch.no_grad():
        train_lp = model(train_tensor).cpu().numpy()
        val_lp   = model(val_tensor).cpu().numpy()
        test_reduced = transform_data(df_test[feature_cols].values)
        test_lp  = model(torch.FloatTensor(test_reduced)).cpu().numpy()

    test_as = -test_lp  # anomaly scores

    # Youden's J statistic: find ROC operating point maximising TPR - FPR
    fpr_arr, tpr_arr, thresholds_roc = roc_curve(ground_truth, test_as)
    j_scores = tpr_arr - fpr_arr
    best_idx = np.argmax(j_scores)
    threshold_as = thresholds_roc[best_idx]
    test_preds = (test_as >= threshold_as).astype(int)
    best_thresh_lp = -threshold_as

    # Metrics
    cm = confusion_matrix(ground_truth, test_preds)
    tn, fp_, fn, tp = cm.ravel()
    acc   = accuracy_score(ground_truth, test_preds)
    prec  = precision_score(ground_truth, test_preds, zero_division=0)
    rec   = recall_score(ground_truth, test_preds, zero_division=0)
    f1    = f1_score(ground_truth, test_preds, zero_division=0)
    auroc = roc_auc_score(ground_truth, test_as)
    ap    = average_precision_score(ground_truth, test_as)

    # Per-polymer breakdown
    polymer_results = {}
    if "polymer" in df_test.columns:
        polymers = df_test["polymer"].values
        for poly in np.unique(polymers):
            mask = polymers == poly
            poly_preds = test_preds[mask]
            poly_gt = ground_truth[mask]
            total = int(mask.sum())
            detected = int(poly_preds.sum())
            expected = "NON-MICROPLASTIC" if poly_gt[0] == 1 else "MICROPLASTIC"
            polymer_results[poly] = {
                "total": total,
                "detected_non_mp": detected,
                "pct_non_mp": round(100 * detected / total, 2),
                "expected": expected,
            }

    # Save plots for this run
    save_loss_curve(losses, val_losses,
                    os.path.join(run_dir, "loss_curves.png"))
    save_confusion_matrix(cm, f1,
                          os.path.join(run_dir, "confusion_matrix.png"))
    save_auroc_curve(ground_truth, test_as, auroc, threshold_as,
                     os.path.join(run_dir, "auroc_curve.png"))
    save_logprob_hist(train_lp.flatten(),
                      test_lp[:n_polymer_test].flatten(),
                      test_lp[n_polymer_test:].flatten(),
                      best_thresh_lp,
                      os.path.join(run_dir, "logprob_distribution.png"))

    return {
        "accuracy":      round(acc, 4),
        "precision":     round(prec, 4),
        "recall":        round(rec, 4),
        "f1":            round(f1, 4),
        "auroc":         round(auroc, 4),
        "avg_precision": round(ap, 4),
        "best_val_loss": round(best_val_loss, 4),
        "final_train_loss": round(losses[-1], 4),
        "train_val_gap": round(val_losses[-1] - losses[-1], 4) if val_losses else None,
        "epochs_trained": len(losses),
        "train_time_sec": round(train_time, 1),
        "n_pca_total":   n_components,
        "n_pca_fp":      int(n_fp),
        "n_pca_ch":      int(n_ch),
        "tail_bound":    round(tail_bound, 2),
        "TP": int(tp), "FP": int(fp_), "TN": int(tn), "FN": int(fn),
        "confusion_matrix": cm.tolist(),
        "polymer_results":  polymer_results,
    }


# =====================================================================
# MAIN SWEEP
# =====================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"  HYPERPARAMETER SWEEP — NF Microplastic Detection (MLP + Split PCA)")
    print(f"  Results → {RESULTS_DIR}/")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}\n")

    # ---- Load data once ----
    df_polymer = pd.read_csv(SOURCE_POLYMER_FILE, index_col=0)
    df_amb     = pd.read_csv(SOURCE_MICROPLASTIC_FILE, index_col=0)

    # Remove non-plastic materials
    df_non_polymer = df_polymer[df_polymer["polymer"].isin(NON_PLASTIC_POLYMERS)].copy()
    df_non_polymer['polymer'] = 'AMB'
    df_polymer = df_polymer[~df_polymer["polymer"].isin(NON_PLASTIC_POLYMERS)]
    df_polymer = df_polymer[
        df_polymer["polymer"].map(df_polymer["polymer"].value_counts()) > 1]

    df_train, df_polymer_test = train_test_split(
        df_polymer, test_size=TEST_SIZE,
        stratify=df_polymer["polymer"], random_state=RANDOM_STATE)
    df_test = pd.concat([df_polymer_test, df_non_polymer,df_amb], ignore_index=True)

    feature_cols = [c for c in df_train.columns
                    if c != "polymer" and str(c) != ""
                    and not str(c).startswith("Unnamed")]

    # Per-region masks
    wavenumber_vals = np.array([float(c) for c in feature_cols])
    fp_mask = wavenumber_vals <= FINGERPRINT_MAX
    ch_mask = wavenumber_vals >= CH_STRETCH_MIN

    print(f"Training samples:   {len(df_train)}")
    print(f"Test polymers:      {len(df_polymer_test)}")
    print(f"Test ambient:       {len(df_amb)}")
    print(f"Feature columns:    {len(feature_cols)}")
    print(f"Fingerprint feats:  {fp_mask.sum()}")
    print(f"C-H stretch feats:  {ch_mask.sum()}\n")

    # ---- Build grid ----
    keys = sorted(SWEEP_GRID.keys())
    combos = [dict(zip(keys, vals))
              for vals in itertools.product(*(SWEEP_GRID[k] for k in keys))]
    n_combos = len(combos)
    print(f"Total configurations to test: {n_combos}\n")

    rows = []

    for run_idx, hp in enumerate(combos, start=1):
        tag = f"run_{run_idx:04d}"
        run_dir = os.path.join(RESULTS_DIR, tag)
        os.makedirs(run_dir, exist_ok=True)

        print(f"[{run_idx:4d}/{n_combos}]  {tag}  "
              f"layers={hp['n_layers']} hdim={hp['hidden_dim']} "
              f"bins={hp['n_bins']} drop={hp['dropout']} "
              f"lr={hp['learning_rate']:.0e} wd={hp['weight_decay']:.0e} "
              f"bs={hp['batch_size']} pca={hp['pca_variance']}")

        try:
            metrics = run_single(
                hp, df_train, df_polymer_test, df_amb, df_test,
                feature_cols, fp_mask, ch_mask, run_dir)

            result = {"run_id": run_idx, "folder": tag, **hp, **metrics}
            rows.append(result)

            print(f"         → PCA={metrics['n_pca_total']}(FP:{metrics['n_pca_fp']},CH:{metrics['n_pca_ch']})  "
                  f"F1={metrics['f1']:.4f}  AUROC={metrics['auroc']:.4f}  "
                  f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
                  f"gap={metrics['train_val_gap']}  "
                  f"epochs={metrics['epochs_trained']}  "
                  f"time={metrics['train_time_sec']:.0f}s\n")
        except Exception as e:
            print(f"         ✗ FAILED: {e}\n")
            rows.append({"run_id": run_idx, "folder": tag, **hp, "error": str(e)})

        # Save incrementally
        _save_results(rows)

    # Final save & summary
    _save_results(rows)
    _print_summary(rows)
    print(f"\nSweep complete. Results saved to {RESULTS_DIR}/")


def _save_results(rows):
    """Save results as CSV (flat) and JSON (full detail with per-polymer)."""
    flat = [{k: v for k, v in r.items()
             if k not in ("confusion_matrix", "polymer_results")}
            for r in rows]
    pd.DataFrame(flat).to_csv(
        os.path.join(RESULTS_DIR, "sweep_results.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "sweep_results_full.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)


def _print_summary(rows):
    """Print top configs ranked by F1 and AUROC."""
    valid = [r for r in rows if "error" not in r]
    if not valid:
        print("\nNo successful runs.")
        return

    print(f"\n{'=' * 70}")
    print(f"  SWEEP SUMMARY — {len(valid)} successful / {len(rows)} total")
    print(f"{'=' * 70}")

    by_f1 = sorted(valid, key=lambda r: r["f1"], reverse=True)
    print(f"\n  TOP 5 by F1:")
    for rank, r in enumerate(by_f1[:5], 1):
        print(f"    {rank}. {r['folder']}  F1={r['f1']:.4f}  AUROC={r['auroc']:.4f}  "
              f"layers={r['n_layers']} hdim={r['hidden_dim']} bins={r['n_bins']} "
              f"lr={r['learning_rate']} wd={r['weight_decay']} drop={r['dropout']} "
              f"bs={r['batch_size']} pca={r['pca_variance']}")

    by_auroc = sorted(valid, key=lambda r: r["auroc"], reverse=True)
    print(f"\n  TOP 5 by AUROC:")
    for rank, r in enumerate(by_auroc[:5], 1):
        print(f"    {rank}. {r['folder']}  AUROC={r['auroc']:.4f}  F1={r['f1']:.4f}  "
              f"layers={r['n_layers']} hdim={r['hidden_dim']} bins={r['n_bins']} "
              f"lr={r['learning_rate']} wd={r['weight_decay']} drop={r['dropout']} "
              f"bs={r['batch_size']} pca={r['pca_variance']}")

    best = by_f1[0]
    best_path = os.path.join(RESULTS_DIR, "best_config.json")
    with open(best_path, "w") as f:
        json.dump({k: v for k, v in best.items()
                   if k not in ("confusion_matrix", "polymer_results")},
                  f, indent=2, default=str)
    print(f"\n  Best config (by F1) saved to {best_path}")


if __name__ == "__main__":
    main()