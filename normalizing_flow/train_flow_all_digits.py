#!/usr/bin/env python3
"""
Train RealNVP normalizing flow models for all MNIST digits (0-9).
Saves checkpoints, generated samples, ROC curves, and anomaly distributions.
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
DATA_ROOT = '../data'
BASELINE_DIGITS = list(range(10))  # Train on digits 0-9

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Training hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
LOG_EVERY = 50

# Flow architecture
INPUT_DIM = 28 * 28  # Flattened MNIST
HIDDEN_DIM = 256
NUM_BLOCKS = 4

# Output directories
CHECKPOINT_DIR = Path('./flow_checkpoints')
SAMPLE_DIR = Path('./flow_gen_images')
ROC_DIR = Path('./flow_anomaly_distribution/roc_curves')
ANOMALY_DIST_DIR = Path('./flow_anomaly_distribution/anomaly_scores')

# Create directories
for d in [CHECKPOINT_DIR, SAMPLE_DIR, ROC_DIR, ANOMALY_DIST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Checkpoint dir: {CHECKPOINT_DIR}")
print(f"Sample dir: {SAMPLE_DIR}")
print(f"ROC dir: {ROC_DIR}")
print(f"Anomaly dist dir: {ANOMALY_DIST_DIR}")

# ============================================================================
# Model Architecture
# ============================================================================

class MLP(nn.Module):
    """Simple MLP for coupling transformation."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # Initialize last layer to zero so coupling starts as identity (s=0,t=0)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class CouplingLayer(nn.Module):
    """Coupling layer for RealNVP."""
    def __init__(self, input_dim, hidden_dim, mask_type='even'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        assert input_dim % 2 == 0, "input_dim must be even"
        self.half = input_dim // 2

        if mask_type == 'even':
            mask = (torch.arange(input_dim) % 2 == 0)
        else:  # odd
            mask = (torch.arange(input_dim) % 2 == 1)
        self.register_buffer('mask', mask)

        self.transform_net = MLP(self.half, hidden_dim, 2 * self.half)

    def forward(self, x):
        mask = self.mask.to(x.device)
        x_id = x[:, mask]
        x_tr = x[:, ~mask]

        st = self.transform_net(x_id)
        s, t = st.chunk(2, dim=1)
        s = 0.1 * torch.tanh(s)

        y_tr = x_tr * torch.exp(s) + t
        y = x.clone()
        y[:, mask] = x_id
        y[:, ~mask] = y_tr

        log_det = s.sum(dim=1)
        return y, log_det

    def inverse(self, y):
        mask = self.mask.to(y.device)
        y_id = y[:, mask]
        y_tr = y[:, ~mask]

        st = self.transform_net(y_id)
        s, t = st.chunk(2, dim=1)
        s = 0.1 * torch.tanh(s)

        x_tr = (y_tr - t) * torch.exp(-s)
        x = y.clone()
        x[:, mask] = y_id
        x[:, ~mask] = x_tr

        log_det = (-s).sum(dim=1)
        return x, log_det


class RealNVP(nn.Module):
    """RealNVP normalizing flow."""
    def __init__(self, input_dim, hidden_dim, num_blocks):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.layers = nn.ModuleList([
            CouplingLayer(input_dim, hidden_dim, mask_type='even' if i % 2 == 0 else 'odd')
            for i in range(num_blocks)
        ])

        self.register_buffer('base_mean', torch.zeros(input_dim))
        self.register_buffer('base_logstd', torch.zeros(input_dim))

    def forward(self, x):
        z = x.view(-1, self.input_dim)
        log_det_total = torch.zeros(z.shape[0], device=z.device)

        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total += log_det

        return z, log_det_total

    def inverse(self, z):
        x = z.view(-1, self.input_dim)
        log_det_total = torch.zeros(x.shape[0], device=x.device)

        for layer in reversed(self.layers):
            x, log_det = layer.inverse(x)
            log_det_total += log_det

        return x, log_det_total

    def log_prob(self, x):
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.input_dim * np.log(2 * np.pi)
        log_px = log_pz + log_det
        return log_px

    def sample(self, batch_size):
        z = torch.randn(batch_size, self.input_dim, device=self.base_mean.device)
        x, _ = self.inverse(z)
        return (x + 0.5).clamp(0, 1)


# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_mnist_digit(root, digit, train=True, download=True):
    """Load MNIST data filtered to a single digit."""
    ds = datasets.MNIST(root=root, train=train, download=download, transform=None)
    targets = ds.targets
    data = ds.data
    idxs = (targets == digit).nonzero(as_tuple=True)[0]
    filtered_data = data[idxs].float().div(255.0) - 0.5  # Center around zero
    return filtered_data


# ============================================================================
# Training Function
# ============================================================================

def train_flow_for_digit(baseline_digit, num_epochs=NUM_EPOCHS):
    """Train a normalizing flow model for a given baseline digit."""
    print(f"\n{'='*70}")
    print(f"Training flow for digit {baseline_digit}")
    print(f"{'='*70}\n")

    # Load data
    train_data = load_mnist_digit(DATA_ROOT, baseline_digit, train=True)
    print(f"Train data shape: {train_data.shape}")

    # Train/val split
    n_train = int(0.8 * len(train_data))
    perm = torch.randperm(len(train_data))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_data_split = train_data[train_idx]
    val_data_split = train_data[val_idx]

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data_split),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_data_split),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # Initialize model
    model = RealNVP(INPUT_DIM, HIDDEN_DIM, NUM_BLOCKS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training loop
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for step, (images,) in enumerate(train_loader):
            images = images.to(DEVICE)
            log_prob = model.log_prob(images)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

            if (step + 1) % LOG_EVERY == 0:
                print(f"Epoch {epoch} Step {step+1}/{len(train_loader)} Loss={train_loss/n_train_batches:.4f}")

        train_loss /= n_train_batches
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for (images,) in val_loader:
                images = images.to(DEVICE)
                log_prob = model.log_prob(images)
                loss = -log_prob.mean()
                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= n_val_batches
        val_losses.append(val_loss)

        print(f"Epoch {epoch} Train Loss={train_loss:.4f} Val Loss={val_loss:.4f}")

    # Save model
    checkpoint_path = CHECKPOINT_DIR / f'flow_digit_{baseline_digit}.pt'
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to: {checkpoint_path}\n")

    return model, train_losses, val_losses


# ============================================================================
# Evaluation and Visualization
# ============================================================================

def evaluate_and_save_results(baseline_digit, model):
    """Evaluate model on test set and save results."""
    print(f"Evaluating flow for digit {baseline_digit}...")

    # Load test data
    test_ds = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=None)
    test_data_full = test_ds.data.float().div(255.0) - 0.5
    test_labels_full = test_ds.targets

    # Compute anomaly scores
    model.eval()
    test_log_probs = []

    with torch.no_grad():
        for i in range(0, len(test_data_full), BATCH_SIZE):
            batch_data = test_data_full[i:i+BATCH_SIZE].to(DEVICE)
            log_prob = model.log_prob(batch_data)
            test_log_probs.append(log_prob.cpu())

    test_log_probs = torch.cat(test_log_probs).numpy()
    anomaly_scores = -test_log_probs
    test_labels = test_labels_full.numpy()

    # Binary labels
    binary_labels = (test_labels != baseline_digit).astype(int)

    # Compute metrics
    auroc = roc_auc_score(binary_labels, anomaly_scores)
    fpr, tpr, _ = roc_curve(binary_labels, anomaly_scores)
    precision, recall, _ = precision_recall_curve(binary_labels, anomaly_scores)

    print(f"AUROC: {auroc:.4f}")

    # Save anomaly score distribution
    anomaly_dist_path = ANOMALY_DIST_DIR / f'anomaly_scores_digit_{baseline_digit}.npz'
    np.savez(
        anomaly_dist_path,
        anomaly_scores=anomaly_scores,
        binary_labels=binary_labels,
        normal_mean=anomaly_scores[binary_labels==0].mean(),
        anomaly_mean=anomaly_scores[binary_labels==1].mean(),
        auroc=auroc
    )
    print(f"Anomaly scores saved to: {anomaly_dist_path}")

    # Save ROC curve
    roc_path = ROC_DIR / f'roc_curve_digit_{baseline_digit}.png'
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC curve
    axes[0].plot(fpr, tpr, lw=2, label=f'Flow (AUROC={auroc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'ROC Curve - RealNVP (Digit {baseline_digit} as Normal)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])

    # Anomaly score distribution
    axes[1].hist(anomaly_scores[binary_labels==0], bins=50, alpha=0.6, label=f'Normal (digit {baseline_digit})', color='green')
    axes[1].hist(anomaly_scores[binary_labels==1], bins=50, alpha=0.6, label='Anomalies (others)', color='red')
    axes[1].set_xlabel('Anomaly Score (Negative Log Likelihood)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Anomaly Score Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {roc_path}")

    return auroc


def generate_and_save_samples(baseline_digit, model):
    """Generate samples and save them."""
    print(f"Generating samples for digit {baseline_digit}...")

    model.eval()
    with torch.no_grad():
        samples = model.sample(batch_size=10)

    # Plot and save
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].view(28, 28).cpu(), cmap='gray')
        ax.axis('off')
    plt.suptitle(f'Samples Generated from Learned Flow (Digit {baseline_digit})')
    plt.tight_layout()

    sample_path = SAMPLE_DIR / f'samples_digit_{baseline_digit}.png'
    plt.savefig(sample_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Samples saved to: {sample_path}\n")


# ============================================================================
# Main Loop
# ============================================================================

def main():
    """Train flow models for all digits and save results."""
    results = {}

    for baseline_digit in BASELINE_DIGITS:
        # Train
        model, train_losses, val_losses = train_flow_for_digit(baseline_digit, num_epochs=NUM_EPOCHS)

        # Evaluate and save results
        auroc = evaluate_and_save_results(baseline_digit, model)
        results[baseline_digit] = auroc

        # Generate and save samples
        generate_and_save_samples(baseline_digit, model)

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY: AUROC for all digits")
    print(f"{'='*70}")
    for digit in sorted(results.keys()):
        print(f"Digit {digit}: AUROC = {results[digit]:.4f}")
    print(f"Mean AUROC: {np.mean(list(results.values())):.4f}")
    print(f"Std AUROC:  {np.std(list(results.values())):.4f}")
    print(f"{'='*70}\n")

    # Save summary
    summary_path = ANOMALY_DIST_DIR / 'auroc_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("AUROC Summary for All Digits\n")
        f.write("="*50 + "\n")
        for digit in sorted(results.keys()):
            f.write(f"Digit {digit}: {results[digit]:.4f}\n")
        f.write("="*50 + "\n")
        f.write(f"Mean AUROC: {np.mean(list(results.values())):.4f}\n")
        f.write(f"Std AUROC:  {np.std(list(results.values())):.4f}\n")
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
