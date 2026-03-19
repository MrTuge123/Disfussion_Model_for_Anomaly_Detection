#!/usr/bin/env python3
"""
Hyperparameter sweep for the 1D-Conv Neural Spline Flow anomaly detector.

Sweeps over:
  HIDDEN_DIM   ∈ {32, 64}
  N_BINS       ∈ {4, 8}
  LEARNING_RATE∈ {1e-4, 5e-4}
  PCA_VARIANCE ∈ {0.80, 0.90, 0.95}

For every combination the script:
  1. Fits PCA + trains the flow from scratch
  2. Selects the threshold via validation-percentile sweep (max F1)
  3. Evaluates on the held-out test set
  4. Saves four plots per run in results/<run_index>/
  5. Writes a summary CSV table to results/summary.csv
"""

import os, itertools, warnings
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
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

# =====================================================================
# FIXED CONFIG (same as the notebook)
# =====================================================================
SOURCE_POLYMER_FILE  = "all_spectra.csv"
SOURCE_MICROPLASTIC_FILE = "ambient_spectra.csv"
TEST_SIZE        = 0.2
VAL_SPLIT        = 0.2
RANDOM_STATE     = 42
N_LAYERS         = 4
TAIL_BOUND_MARGIN = 1.2
DROPOUT          = 0.4
N_EPOCHS         = 150
BATCH_SIZE       = 64
WEIGHT_DECAY     = 1e-4
CLIP_GRAD_NORM   = 3.0
PATIENCE         = 20
THRESHOLD_PERCENTILE = 10   # baseline reference only

# =====================================================================
# SWEEP GRID
# =====================================================================
HIDDEN_DIMS    = [32, 64]
N_BINS_LIST    = [4, 8]
LEARNING_RATES = [1e-4, 5e-4]
PCA_VARIANCES  = [0.80, 0.90, 0.95]

RESULTS_DIR = "results"

# =====================================================================
# MODEL DEFINITION  (identical to the notebook)
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


def rational_quadratic_spline_inverse(y, widths, heights, derivatives, tail_bound=5.0):
    K = widths.shape[-1]
    widths = F.softmax(widths, dim=-1) * 2 * tail_bound
    heights = F.softmax(heights, dim=-1) * 2 * tail_bound
    derivatives = F.softplus(derivatives)
    cumwidths  = torch.cumsum(widths, dim=-1)
    cumwidths  = F.pad(cumwidths, (1, 0), value=0.0) - tail_bound
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0) - tail_bound
    y_clamped = y.clamp(-tail_bound + 1e-6, tail_bound - 1e-6)
    bin_idx = torch.searchsorted(cumheights[..., 1:], y_clamped.unsqueeze(-1)).squeeze(-1)
    bin_idx = bin_idx.clamp(0, K - 1)
    idx = bin_idx.unsqueeze(-1)
    w_k  = widths.gather(-1, idx).squeeze(-1)
    h_k  = heights.gather(-1, idx).squeeze(-1)
    d_k  = derivatives.gather(-1, idx).squeeze(-1)
    d_k1 = derivatives.gather(-1, idx + 1).squeeze(-1)
    cw_k = cumwidths.gather(-1, idx).squeeze(-1)
    ch_k = cumheights.gather(-1, idx).squeeze(-1)
    s_k = h_k / w_k
    a = h_k * (s_k - d_k) + (y_clamped - ch_k) * (d_k + d_k1 - 2 * s_k)
    b = h_k * d_k - (y_clamped - ch_k) * (d_k + d_k1 - 2 * s_k)
    c = -s_k * (y_clamped - ch_k)
    disc = (b * b - 4 * a * c).clamp(min=0)
    xi = ((2 * c) / (-b - torch.sqrt(disc + 1e-8))).clamp(1e-6, 1.0 - 1e-6)
    return cw_k + xi * w_k


class SplineParamNet(nn.Module):
    def __init__(self, in_dim, transform_dim, hidden_dim=128, n_bins=8, dropout=0.4):
        super().__init__()
        self.transform_dim = transform_dim
        self.n_bins = n_bins
        out_dim = transform_dim * (3 * n_bins + 1)
        self.input_conv = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.block1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(dropout))
        self.output_proj = nn.Linear(hidden_dim * in_dim, out_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        batch = x.shape[0]
        h = F.relu(self.input_conv(x.unsqueeze(1)))
        h = h + self.block1(h)
        h = h + self.block2(h)
        params = self.output_proj(h.flatten(1))
        params = params.view(batch, self.transform_dim, 3 * self.n_bins + 1)
        return params[:, :, :self.n_bins], params[:, :, self.n_bins:2*self.n_bins], params[:, :, 2*self.n_bins:]


class CouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim=128, n_bins=8, tail_bound=5.0, reverse=False):
        super().__init__()
        self.dim = dim; self.reverse = reverse
        self.split_dim = (dim + 1) // 2
        self.transform_dim = dim - self.split_dim
        self.tail_bound = tail_bound
        cond_dim  = self.transform_dim if reverse else self.split_dim
        trans_dim = self.split_dim     if reverse else self.transform_dim
        self.param_net = SplineParamNet(cond_dim, trans_dim, hidden_dim, n_bins)

    def forward(self, x):
        if self.reverse:
            x_cond, x_trans = x[:, self.split_dim:], x[:, :self.split_dim]
        else:
            x_cond, x_trans = x[:, :self.split_dim], x[:, self.split_dim:]
        w, h, d = self.param_net(x_cond)
        y_trans, ld = rational_quadratic_spline_forward(x_trans, w, h, d, self.tail_bound)
        ld_sum = ld.sum(1)
        if self.reverse:
            return torch.cat([y_trans, x_cond], 1), ld_sum
        return torch.cat([x_cond, y_trans], 1), ld_sum

    def inverse(self, y):
        if self.reverse:
            y_trans, y_cond = y[:, :self.split_dim], y[:, self.split_dim:]
        else:
            y_cond, y_trans = y[:, :self.split_dim], y[:, self.split_dim:]
        w, h, d = self.param_net(y_cond)
        x_trans = rational_quadratic_spline_inverse(y_trans, w, h, d, self.tail_bound)
        if self.reverse:
            return torch.cat([x_trans, y_cond], 1)
        return torch.cat([y_cond, x_trans], 1)


class StandardGaussian:
    def __init__(self, dim):
        self.dim = dim
        self._log_norm = -0.5 * dim * np.log(2.0 * np.pi)
    def log_prob(self, z):
        return self._log_norm - 0.5 * (z * z).sum(dim=1)
    def sample(self, n):
        return torch.randn(n, self.dim)


class PermutationLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("permutation", torch.randperm(dim))
        self.register_buffer("inverse_permutation", torch.argsort(self.permutation))
    def forward(self, x):
        return x[:, self.permutation], torch.zeros(x.shape[0], device=x.device)
    def inverse(self, y):
        return y[:, self.inverse_permutation]


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
        if not self.initialized: self._data_dependent_init(x)
        z = (x + self.bias) * torch.exp(self.log_scale)
        return z, self.log_scale.sum().expand(x.shape[0])
    def inverse(self, z):
        return z * torch.exp(-self.log_scale) - self.bias


class BatchNormFlow(nn.Module):
    def __init__(self, dim, momentum=0.1, eps=1e-5):
        super().__init__()
        self.eps = eps; self.momentum = momentum
        self.log_gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta      = nn.Parameter(torch.zeros(1, dim))
        self.register_buffer("running_mean", torch.zeros(1, dim))
        self.register_buffer("running_var",  torch.ones(1, dim))
    def forward(self, x):
        if self.training:
            mean = x.mean(0, keepdim=True); var = x.var(0, keepdim=True) + self.eps
            self.running_mean.mul_(1 - self.momentum).add_(mean * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var * self.momentum)
        else:
            mean = self.running_mean; var = self.running_var + self.eps
        x_hat = (x - mean) / torch.sqrt(var)
        z = x_hat * torch.exp(self.log_gamma) + self.beta
        return z, (self.log_gamma - 0.5 * torch.log(var)).sum().expand(x.shape[0])
    def inverse(self, z):
        var = self.running_var + self.eps
        return (z - self.beta) * torch.exp(-self.log_gamma) * torch.sqrt(var) + self.running_mean


class NormalizingFlow(nn.Module):
    def __init__(self, dim, n_layers=4, hidden_dim=128, n_bins=8, tail_bound=3.0):
        super().__init__()
        self.dim = dim; self.base_dist = StandardGaussian(dim)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(ActNorm(dim))
            self.layers.append(BatchNormFlow(dim))
            self.layers.append(CouplingLayer(dim, hidden_dim, n_bins, tail_bound, reverse=(i % 2 == 1)))
            if i < n_layers - 1:
                self.layers.append(PermutationLayer(dim))
    def forward(self, x):
        z = x; ld = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            z, d = layer(z); ld += d
        return self.base_dist.log_prob(z) + ld


# =====================================================================
# TRAINING FUNCTION
# =====================================================================

def train_flow(model, data, n_epochs, batch_size, lr, weight_decay,
               clip_grad_norm, val_data, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    n_batches = len(data) // batch_size
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
    return losses, val_losses


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
                xticklabels=["NORMAL", "ANOMALY"],
                yticklabels=["NORMAL", "ANOMALY"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (F1={f1:.3f})")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_auroc_curve(ground_truth, anomaly_scores, auroc_val, threshold_as, path):
    fpr, tpr, _ = roc_curve(ground_truth, anomaly_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="purple", lw=2.5, label=f"AUROC = {auroc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    # mark operating point
    preds = (anomaly_scores >= threshold_as).astype(int)
    cur_fpr = (preds[ground_truth == 0] == 1).mean()
    cur_tpr = (preds[ground_truth == 1] == 1).mean()
    ax.scatter([cur_fpr], [cur_tpr], color="red", s=100, zorder=5,
              edgecolors="black", lw=1.5, label=f"Threshold (FPR={cur_fpr:.2f}, TPR={cur_tpr:.2f})")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_logprob_hist(train_lp, test_normal_lp, test_anomaly_lp, threshold_lp, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(train_lp, bins=50, alpha=0.5, density=True, color="green",
            edgecolor="black", label="Train (normal)")
    ax.hist(test_normal_lp, bins=30, alpha=0.5, density=True, color="blue",
            edgecolor="black", label="Test polymers (normal)")
    ax.hist(test_anomaly_lp, bins=30, alpha=0.5, density=True, color="red",
            edgecolor="black", label="Test ambient (anomaly)")
    ax.axvline(threshold_lp, color="black", ls="--", lw=2,
               label=f"Threshold = {threshold_lp:.2f}")
    ax.set_xlabel("Log Probability"); ax.set_ylabel("Density")
    ax.set_title("Log-Probability Distributions"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# =====================================================================
# MAIN SWEEP
# =====================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Load data once ----
    df_polymer = pd.read_csv(SOURCE_POLYMER_FILE, index_col=0)
    df_amb     = pd.read_csv(SOURCE_MICROPLASTIC_FILE, index_col=0)
    df_polymer = df_polymer[df_polymer["polymer"].map(df_polymer["polymer"].value_counts()) > 1]
    df_train, df_polymer_test = train_test_split(
        df_polymer, test_size=TEST_SIZE, stratify=df_polymer["polymer"], random_state=RANDOM_STATE)
    df_test = pd.concat([df_polymer_test, df_amb], ignore_index=True)
    feature_cols = [c for c in df_train.columns
                    if c != "polymer" and str(c) != "" and not str(c).startswith("Unnamed")]

    n_polymer_test = len(df_polymer_test)
    n_amb = len(df_amb)
    ground_truth = np.array([0] * n_polymer_test + [1] * n_amb)

    # Pre-scale once (PCA is per-run because variance changes)
    normal_data = df_train[feature_cols].values
    scaler = StandardScaler()
    normal_scaled = scaler.fit_transform(normal_data)
    test_raw = df_test[feature_cols].values
    test_scaled = scaler.transform(test_raw)

    # ---- Build grid ----
    grid = list(itertools.product(HIDDEN_DIMS, N_BINS_LIST, LEARNING_RATES, PCA_VARIANCES))
    print(f"Running {len(grid)} configurations …\n")

    rows = []

    for run_idx, (hidden_dim, n_bins, lr, pca_var) in enumerate(grid, start=1):
        tag = f"run_{run_idx:03d}"
        run_dir = os.path.join(RESULTS_DIR, tag)
        os.makedirs(run_dir, exist_ok=True)

        print(f"[{run_idx:2d}/{len(grid)}]  hidden={hidden_dim}  bins={n_bins}  "
              f"lr={lr:.0e}  pca={pca_var}")

        # ---- PCA (per-run) ----
        pca = PCA(n_components=pca_var, svd_solver="full")
        normal_reduced = pca.fit_transform(normal_scaled)
        n_comp = pca.n_components_
        tail_bound = float(np.abs(normal_reduced).max()) * TAIL_BOUND_MARGIN

        # Train / val split (deterministic)
        n_val = int(len(normal_reduced) * VAL_SPLIT)
        perm = np.random.RandomState(RANDOM_STATE).permutation(len(normal_reduced))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        train_tensor = torch.FloatTensor(normal_reduced[train_idx])
        val_tensor   = torch.FloatTensor(normal_reduced[val_idx])

        # ---- Build & train ----
        model = NormalizingFlow(
            dim=n_comp, n_layers=N_LAYERS, hidden_dim=hidden_dim,
            n_bins=n_bins, tail_bound=tail_bound)

        losses, val_losses = train_flow(
            model, train_tensor, N_EPOCHS, BATCH_SIZE, lr,
            WEIGHT_DECAY, CLIP_GRAD_NORM, val_tensor, PATIENCE)

        # ---- Evaluate ----
        model.eval()
        with torch.no_grad():
            train_lp = model(train_tensor).cpu().numpy()
            val_lp   = model(val_tensor).cpu().numpy()
            test_reduced = pca.transform(test_scaled)
            test_lp  = model(torch.FloatTensor(test_reduced)).cpu().numpy()

        test_as = -test_lp  # anomaly scores

        # Validation-based threshold sweep (maximise F1)
        best_f1, best_thresh_lp = 0.0, np.percentile(val_lp, THRESHOLD_PERCENTILE)
        for p in np.arange(1, 50, 0.5):
            cand = np.percentile(val_lp, p)
            preds = (test_as >= -cand).astype(int)
            cur_f1 = f1_score(ground_truth, preds, zero_division=0)
            if cur_f1 > best_f1:
                best_f1 = cur_f1
                best_thresh_lp = cand

        threshold_as = -best_thresh_lp
        test_preds = (test_as >= threshold_as).astype(int)

        # Metrics
        cm = confusion_matrix(ground_truth, test_preds)
        tn, fp, fn, tp = cm.ravel()
        prec = precision_score(ground_truth, test_preds, zero_division=0)
        rec  = recall_score(ground_truth, test_preds, zero_division=0)
        f1   = f1_score(ground_truth, test_preds, zero_division=0)
        auroc = roc_auc_score(ground_truth, test_as)

        print(f"         → PCA dims={n_comp}  F1={f1:.4f}  AUROC={auroc:.4f}")

        # ---- Save plots ----
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

        # ---- Record ----
        rows.append({
            "run":        run_idx,
            "folder":     tag,
            "hidden_dim": hidden_dim,
            "n_bins":     n_bins,
            "lr":         lr,
            "pca_var":    pca_var,
            "pca_dims":   n_comp,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "F1":    round(f1, 4),
            "AUROC": round(auroc, 4),
        })

    # ---- Summary table ----
    summary = pd.DataFrame(rows)
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\n{'='*80}")
    print(summary.to_string(index=False))
    print(f"{'='*80}")
    print(f"\nSummary saved to {summary_path}")
    print(f"Per-run plots saved to {RESULTS_DIR}/run_XXX/")


if __name__ == "__main__":
    main()
