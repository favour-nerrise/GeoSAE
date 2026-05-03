"""Training code for GeoSAE."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from geosae import data
from geosae import io
from geosae.manifold import build_knn_graph
from geosae.manifold import KnnGraph
from geosae.models import SparseAutoencoder


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Configuration for GeoSAE training."""

    feature_dir: Path
    output_dir: Path
    expansion_factor: int = 2
    topk: int = 16
    activation: str = "topk"
    knn_k: int = 15
    lambda_manifold: float = 0.1
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.1
    seed: int = 42


class ManifoldDataset(Dataset):
    """Dataset that pairs each sample with its manifold neighbors."""

    def __init__(
        self,
        features: np.ndarray,
        knn_indices: np.ndarray,
        knn_distances: np.ndarray,
        all_features: np.ndarray,
    ) -> None:
        self.features = torch.from_numpy(features)
        self.knn_indices = knn_indices
        self.knn_distances = torch.from_numpy(knn_distances.astype(np.float32))
        self.all_features = torch.from_numpy(all_features)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        neighbors = self.all_features[self.knn_indices[index]]
        return self.features[index], neighbors, self.knn_distances[index]


def _split_indices(n_samples: int, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Creates a reproducible train-validation split."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    n_val = max(1, int(validation_fraction * n_samples))
    return indices[n_val:], indices[:n_val]


def train_geosae(features: np.ndarray, graph: KnnGraph, config: TrainConfig) -> dict[str, object]:
    """Trains a GeoSAE model from embeddings and a precomputed graph.

    Args:
      features: Input embedding matrix.
      graph: Manifold graph built on `features`.
      config: Training configuration.

    Returns:
      Training summary dictionary.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_indices, val_indices = _split_indices(
        len(features), config.validation_fraction, config.seed)
    train_dataset = ManifoldDataset(
        features[train_indices],
        graph.indices[train_indices],
        graph.distances[train_indices],
        features,
    )
    val_tensor = torch.from_numpy(features[val_indices]).to(device)

    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    model = SparseAutoencoder(
        input_dim=features.shape[1],
        expansion_factor=config.expansion_factor,
        topk=config.topk,
        activation=config.activation,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state = None
    best_val_mse = float("inf")
    history = []

    for epoch in range(config.epochs):
        model.train()
        epoch_mse = 0.0
        epoch_manifold = 0.0
        epoch_count = 0
        for batch_features, batch_neighbors, batch_distances in loader:
            batch_features = batch_features.to(device)
            batch_neighbors = batch_neighbors.to(device)
            batch_distances = batch_distances.to(device)

            outputs = model(batch_features)
            batch_size = batch_features.shape[0]
            neighbor_count = batch_neighbors.shape[1]

            batch_pre = model.pre_activations(batch_features)
            with torch.no_grad():
                neighbor_pre = model.pre_activations(
                    batch_neighbors.reshape(batch_size * neighbor_count, -1))
            neighbor_pre = neighbor_pre.reshape(batch_size, neighbor_count, -1)

            weights = torch.exp(-batch_distances / max(graph.sigma, 1e-8))
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            squared_distance = (
                (batch_pre.unsqueeze(1) - neighbor_pre).pow(2).mean(dim=-1))
            manifold_loss = (weights * squared_distance).sum(dim=1).mean()
            total_loss = outputs["mse"] + config.lambda_manifold * manifold_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            model.normalize_decoder()

            epoch_mse += outputs["mse"].item() * batch_size
            epoch_manifold += manifold_loss.item() * batch_size
            epoch_count += batch_size

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tensor)
            residual_var = (val_tensor - val_outputs["x_hat"]).var().item()
            total_var = val_tensor.var().item()
            explained_variance = 1.0 - residual_var / max(total_var, 1e-8)
            alive_features = int((val_outputs["latents"] > 0).any(dim=0).sum().item())
            val_mse = float(val_outputs["mse"].item())

        history.append({
            "epoch": epoch,
            "train_mse": epoch_mse / epoch_count,
            "train_manifold": epoch_manifold / epoch_count,
            "val_mse": val_mse,
            "explained_variance": explained_variance,
            "alive_features": alive_features,
        })
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {
                "model_state_dict": model.state_dict(),
                "history": history,
            }

    io.ensure_dir(config.output_dir)
    assert best_state is not None
    checkpoint = {
        "config": dataclasses.asdict(config),
        "graph": {
            "k": config.knn_k,
            "sigma": graph.sigma,
        },
        **best_state,
    }
    torch.save(checkpoint, config.output_dir / "best_model.pt")
    io.write_json(
        {
            "input_dim": int(features.shape[1]),
            "n_samples": int(len(features)),
            "train_samples": int(len(train_indices)),
            "val_samples": int(len(val_indices)),
            "best_val_mse": best_val_mse,
            "history": history,
        },
        config.output_dir / "train_summary.json",
    )
    return checkpoint


def main() -> None:
    """Parses arguments and trains a GeoSAE model."""
    parser = argparse.ArgumentParser(description="Train a GeoSAE model.")
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expansion-factor", type=int, default=2)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--activation", type=str, default="topk",
                        choices=["topk", "relu", "jumprelu"])
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--lambda-manifold", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        expansion_factor=args.expansion_factor,
        topk=args.topk,
        activation=args.activation,
        knn_k=args.knn_k,
        lambda_manifold=args.lambda_manifold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    table = data.load_feature_directory(args.feature_dir, args.metadata_csv)
    graph = build_knn_graph(table.features, k=args.knn_k)
    train_geosae(table.features, graph, config)


if __name__ == "__main__":
    main()
