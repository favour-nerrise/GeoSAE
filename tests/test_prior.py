"""Tests for stage 1 (geometric prior analysis)."""

from __future__ import annotations

import numpy as np
import pytest

from geosae.stage1_prior import (
    angular_radial_eta2,
    compute_report,
    gini,
    local_intrinsic_dimension,
    recommend_activation,
)


def test_gini_uniform_is_zero() -> None:
    assert gini(np.ones(10)) == pytest.approx(0.0, abs=1e-9)


def test_gini_concentrated_is_close_to_one() -> None:
    values = np.zeros(100)
    values[0] = 1.0
    assert gini(values) > 0.9


def test_angular_radial_eta2_in_unit_interval() -> None:
    rng = np.random.RandomState(0)
    features = rng.randn(50, 8).astype(np.float32)
    labels = rng.randint(0, 3, size=50)
    angular, radial = angular_radial_eta2(features, labels)
    assert 0.0 <= angular <= 1.0
    assert 0.0 <= radial <= 1.0


def test_radial_eta2_dominates_when_classes_differ_only_in_norm() -> None:
    """If classes share direction but have different magnitudes, radial > angular."""
    rng = np.random.RandomState(0)
    direction = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    n_per = 60
    features = np.concatenate([
        direction[None, :] * (1.0 + 0.05 * rng.randn(n_per, 1)),
        direction[None, :] * (3.0 + 0.05 * rng.randn(n_per, 1)),
    ]).astype(np.float32)
    labels = np.concatenate([np.zeros(n_per), np.ones(n_per)])
    angular, radial = angular_radial_eta2(features, labels)
    assert radial > angular


def test_intrinsic_dimension_returns_one_per_sample() -> None:
    rng = np.random.RandomState(0)
    features = rng.randn(40, 8).astype(np.float32)
    dims = local_intrinsic_dimension(features, n_neighbors=10)
    assert dims.shape == (40,)
    assert dims.min() >= 1
    assert dims.max() <= 8


def test_recommend_activation_only_returns_supported_choices() -> None:
    rng = np.random.RandomState(0)
    for _ in range(20):
        features = rng.randn(40, 8).astype(np.float32)
        labels = rng.randint(0, 3, size=40)
        report = compute_report(features, labels, n_neighbors=10)
        recommendation, votes = recommend_activation(report)
        assert recommendation in {"topk", "relu", "jumprelu"}
        for vote in votes.values():
            assert vote in {"topk", "relu", "jumprelu"}
