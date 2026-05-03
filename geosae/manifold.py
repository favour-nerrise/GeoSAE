"""Manifold graph construction utilities."""

from __future__ import annotations

import dataclasses

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclasses.dataclass(frozen=True)
class KnnGraph:
    """k-nearest-neighbor graph for manifold regularization.

    Attributes:
      indices: Neighbor indices with shape `(n_samples, k)`.
      distances: Neighbor distances with shape `(n_samples, k)`.
      sigma: Median neighbor distance used as the kernel bandwidth.
    """

    indices: np.ndarray
    distances: np.ndarray
    sigma: float


def build_knn_graph(features: np.ndarray, k: int) -> KnnGraph:
    """Builds a k-nearest-neighbor graph on the embedding space.

    Args:
      features: Array of shape `(n_samples, d_model)`.
      k: Number of neighbors per sample.

    Returns:
      A `KnnGraph` instance.
    """
    neighbors = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
    neighbors.fit(features)
    distances, indices = neighbors.kneighbors(features)
    distances = distances[:, 1:].astype(np.float32)
    indices = indices[:, 1:].astype(np.int64)
    sigma = float(np.median(distances))
    return KnnGraph(indices=indices, distances=distances, sigma=sigma)


def normalized_kernel_weights(distances: np.ndarray, sigma: float) -> np.ndarray:
    """Computes normalized Gaussian weights from neighbor distances.

    Args:
      distances: Neighbor distances with shape `(batch_size, k)`.
      sigma: Kernel bandwidth.

    Returns:
      Row-normalized neighbor weights.
    """
    weights = np.exp(-distances / max(sigma, 1e-8))
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    return weights / np.clip(weights_sum, 1e-8, None)

