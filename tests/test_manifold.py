"""Tests for the k-NN manifold graph."""

from __future__ import annotations

import numpy as np

from geosae.manifold import build_knn_graph, normalized_kernel_weights


def test_build_knn_graph_shapes() -> None:
    rng = np.random.RandomState(0)
    features = rng.randn(20, 8).astype(np.float32)
    graph = build_knn_graph(features, k=5)
    assert graph.indices.shape == (20, 5)
    assert graph.distances.shape == (20, 5)
    assert graph.sigma > 0


def test_build_knn_graph_excludes_self() -> None:
    rng = np.random.RandomState(0)
    features = rng.randn(20, 8).astype(np.float32)
    graph = build_knn_graph(features, k=3)
    for i, neighbors in enumerate(graph.indices):
        assert i not in neighbors.tolist()


def test_normalized_kernel_weights_sum_to_one() -> None:
    distances = np.array([[0.0, 0.5, 1.0], [0.1, 0.2, 0.3]], dtype=np.float32)
    weights = normalized_kernel_weights(distances, sigma=0.5)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-5)
    assert (weights >= 0).all()
