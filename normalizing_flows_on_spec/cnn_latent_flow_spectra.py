import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


DEFAULT_SOURCE_FILE = "all_spectra.csv"
DEFAULT_NORMAL_POLYMER = "Polyethylene"


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [
        col
        for col in df.columns
        if col != "polymer" and str(col) != "" and not str(col).startswith("Unnamed")
    ]


class AffineCouplingLayer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 64, scale_factor: float = 0.5):
        super().__init__()
        self.dim = dim
        self.split_dim = dim // 2
        self.scale_factor = scale_factor

        self.scale_net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.split_dim, hidden_dim)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, dim - self.split_dim)),
            nn.Tanh(),
        )

        self.shift_net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.split_dim, hidden_dim)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, dim - self.split_dim)),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x[:, : self.split_dim], x[:, self.split_dim :]

        s = self.scale_net(x1) * self.scale_factor
        t = self.shift_net(x1)

        y1 = x1
        y2 = x2 * torch.exp(s) + t
        log_det = s.sum(dim=1)

        return torch.cat([y1, y2], dim=1), log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y1, y2 = y[:, : self.split_dim], y[:, self.split_dim :]

        s = self.scale_net(y1) * self.scale_factor
        t = self.shift_net(y1)

        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)

        return torch.cat([x1, x2], dim=1)


class PermutationLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("permutation", torch.randperm(dim))
        self.register_buffer("inverse_permutation", torch.argsort(self.permutation))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x[:, self.permutation], torch.zeros(x.shape[0], device=x.device)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y[:, self.inverse_permutation]


class StandardGaussian:
    def __init__(self, dim: int):
        self.dim = dim

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        log_norm = 0.5 * self.dim * np.log(2.0 * np.pi)
        return -0.5 * torch.sum(z * z, dim=1) - log_norm

    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        return torch.randn(n_samples, self.dim, device=device)


class NormalizingFlow(nn.Module):
    def __init__(self, dim: int, n_layers: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.base_dist = StandardGaussian(dim)
        self.layers = nn.ModuleList()

        for layer_idx in range(n_layers):
            self.layers.append(AffineCouplingLayer(dim, hidden_dim=hidden_dim))
            if layer_idx < n_layers - 1:
                self.layers.append(PermutationLayer(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        log_det_sum = torch.zeros(x.shape[0], device=x.device)

        for layer in self.layers:
            z, log_det = layer(z)
            log_det_sum += log_det

        return self.base_dist.log_prob(z) + log_det_sum

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        z = self.base_dist.sample(n_samples, device=device)
        return self.inverse(z)


class SpectralCNNEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, pooled_length: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(pooled_length),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * pooled_length, 128),
            nn.SiLU(),
            nn.Linear(128, latent_dim),
        )
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.projection(x)


class SpectralCNNDecoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, seed_length: int = 16):
        super().__init__()
        self.seed_length = seed_length
        self.input_dim = input_dim
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128 * seed_length),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.projection(z).view(z.shape[0], 128, self.seed_length)
        x = F.interpolate(x, size=self.input_dim, mode="linear", align_corners=False)
        x = self.decoder(x)
        return x.squeeze(1)


class CNNLatentFlowModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        flow_layers: int = 6,
        flow_hidden_dim: int = 64,
    ):
        super().__init__()
        self.encoder = SpectralCNNEncoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = SpectralCNNDecoder(input_dim=input_dim, latent_dim=latent_dim)
        self.flow = NormalizingFlow(
            dim=latent_dim,
            n_layers=flow_layers,
            hidden_dim=flow_hidden_dim,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(x)
        reconstruction = self.decode(z)
        log_prob = self.flow(z)
        recon_error = F.mse_loss(reconstruction, x, reduction="none").mean(dim=1)

        return {
            "latent": z,
            "reconstruction": reconstruction,
            "log_prob": log_prob,
            "reconstruction_error": recon_error,
        }

    def anomaly_score(self, x: torch.Tensor, reconstruction_weight: float = 1.0) -> torch.Tensor:
        outputs = self.forward(x)
        return -outputs["log_prob"] + reconstruction_weight * outputs["reconstruction_error"]


def load_training_data(
    source_file: str,
    normal_polymer: str,
) -> Tuple[pd.DataFrame, List[str], StandardScaler, torch.Tensor]:
    df = pd.read_csv(source_file, index_col=False)
    feature_cols = get_feature_columns(df)

    normal_data = df.loc[df["polymer"] == normal_polymer, feature_cols].values.astype(np.float32)
    if len(normal_data) == 0:
        raise ValueError(f"No training rows found for polymer '{normal_polymer}' in {source_file}")

    scaler = StandardScaler()
    normal_scaled = scaler.fit_transform(normal_data).astype(np.float32)

    return df, feature_cols, scaler, torch.from_numpy(normal_scaled)


def train_latent_flow(
    model: CNNLatentFlowModel,
    data: torch.Tensor,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    reconstruction_weight: float = 10.0,
    flow_weight: float = 1.0,
    device: str = "cpu",
) -> Dict[str, List[float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True, drop_last=False)

    history = {"total_loss": [], "reconstruction_loss": [], "flow_loss": []}

    for epoch in tqdm(range(n_epochs), desc="Training"):
        model.train()
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        flow_loss_sum = 0.0

        for (batch,) in loader:
            batch = batch.to(device)
            outputs = model(batch)

            recon_loss = outputs["reconstruction_error"].mean()
            flow_loss = -outputs["log_prob"].mean()
            loss = reconstruction_weight * recon_loss + flow_weight * flow_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_sum += loss.item()
            recon_loss_sum += recon_loss.item()
            flow_loss_sum += flow_loss.item()

        num_batches = len(loader)
        history["total_loss"].append(total_loss_sum / num_batches)
        history["reconstruction_loss"].append(recon_loss_sum / num_batches)
        history["flow_loss"].append(flow_loss_sum / num_batches)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:>3}/{n_epochs} | "
                f"total={history['total_loss'][-1]:.4f} | "
                f"recon={history['reconstruction_loss'][-1]:.4f} | "
                f"flow={history['flow_loss'][-1]:.4f}"
            )

    return history


def score_dataframe(
    model: CNNLatentFlowModel,
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: StandardScaler,
    device: str,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    results: Dict[str, Dict[str, float]] = {}

    with torch.no_grad():
        for polymer in df["polymer"].unique():
            polymer_data = df.loc[df["polymer"] == polymer, feature_cols].values.astype(np.float32)
            polymer_scaled = scaler.transform(polymer_data).astype(np.float32)
            polymer_tensor = torch.from_numpy(polymer_scaled).to(device)

            outputs = model(polymer_tensor)
            anomaly_score = -outputs["log_prob"] + outputs["reconstruction_error"]

            results[polymer] = {
                "mean_log_prob": outputs["log_prob"].mean().item(),
                "std_log_prob": outputs["log_prob"].std(unbiased=False).item(),
                "mean_reconstruction_error": outputs["reconstruction_error"].mean().item(),
                "mean_anomaly_score": anomaly_score.mean().item(),
            }

    return results


def print_results(results: Dict[str, Dict[str, float]], normal_polymer: str) -> None:
    polymers = sorted(results, key=lambda name: results[name]["mean_log_prob"], reverse=True)

    print("\n" + "=" * 104)
    print(
        f"{'Polymer':<40} "
        f"{'Mean Log Prob':>15} "
        f"{'Std Log Prob':>15} "
        f"{'Mean Recon MSE':>15} "
        f"{'Mean Anomaly':>15}"
    )
    print("=" * 104)

    for polymer in polymers:
        marker = " <- NORMAL" if polymer == normal_polymer else ""
        metrics = results[polymer]
        print(
            f"{polymer:<40} "
            f"{metrics['mean_log_prob']:>15.4f} "
            f"{metrics['std_log_prob']:>15.4f} "
            f"{metrics['mean_reconstruction_error']:>15.4f} "
            f"{metrics['mean_anomaly_score']:>15.4f}"
            f"{marker}"
        )

    print("=" * 104)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a CNN encoder that compresses each 535-point spectrum into a latent "
            "vector, then fit a normalizing flow in that latent space."
        )
    )
    parser.add_argument("--source-file", default=DEFAULT_SOURCE_FILE)
    parser.add_argument("--normal-polymer", default=DEFAULT_NORMAL_POLYMER)
    parser.add_argument("--latent-dim", type=int, default=16, help="Recommended range: 8-32")
    parser.add_argument("--flow-layers", type=int, default=6)
    parser.add_argument("--flow-hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reconstruction-weight", type=float, default=10.0)
    parser.add_argument("--flow-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not 8 <= args.latent_dim <= 32:
        print(
            f"Warning: latent_dim={args.latent_dim} is outside the requested 8-32 range. "
            "Training will still run."
        )

    df, feature_cols, scaler, train_tensor = load_training_data(
        source_file=args.source_file,
        normal_polymer=args.normal_polymer,
    )

    model = CNNLatentFlowModel(
        input_dim=train_tensor.shape[1],
        latent_dim=args.latent_dim,
        flow_layers=args.flow_layers,
        flow_hidden_dim=args.flow_hidden_dim,
    )

    print(f"Training on {train_tensor.shape[0]} samples with {train_tensor.shape[1]} spectral features")
    print(f"Compressed latent dimension: {args.latent_dim}")
    print(f"Device: {device}")

    train_latent_flow(
        model=model,
        data=train_tensor,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        reconstruction_weight=args.reconstruction_weight,
        flow_weight=args.flow_weight,
        device=device,
    )

    results = score_dataframe(
        model=model,
        df=df,
        feature_cols=feature_cols,
        scaler=scaler,
        device=device,
    )
    print_results(results, normal_polymer=args.normal_polymer)


if __name__ == "__main__":
    main()
