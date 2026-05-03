"""Tests for stage 3 (annotation) primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geosae.stage3_annotate import (
    DEFAULT_CATEGORY_MAP,
    annotate_features,
    partial_spearman,
)


def test_partial_spearman_returns_valid_range() -> None:
    rng = np.random.RandomState(0)
    rho, p_value = partial_spearman(rng.randn(100), rng.randn(100), rng.randn(100))
    assert -1.0 <= rho <= 1.0
    assert 0.0 <= p_value <= 1.0


def test_partial_spearman_perfect_self_correlation() -> None:
    rng = np.random.RandomState(0)
    x = rng.randn(100)
    z = rng.randn(100)
    rho, _ = partial_spearman(x, x, z)
    assert rho == pytest.approx(1.0, abs=1e-6)


def test_partial_spearman_drops_age_confound() -> None:
    """Independent feature and variable should give a smaller partial than raw correlation
    when both are inflated by a shared age-like covariate."""
    import scipy.stats
    rng = np.random.RandomState(1)
    n = 1000
    age = rng.uniform(60, 85, size=n)
    feature = age + 1.0 * rng.randn(n)
    variable = age + 1.0 * rng.randn(n)
    raw, _ = scipy.stats.spearmanr(feature, variable)
    partial, _ = partial_spearman(feature, variable, age)
    assert abs(partial) < abs(raw)


def test_annotate_features_assigns_known_categories(tmp_path) -> None:
    rng = np.random.RandomState(0)
    n_samples = 120
    n_features = 8
    age = rng.uniform(60, 85, size=n_samples)
    diagnosis_score = rng.choice([0, 1, 2], size=n_samples)
    latents = rng.rand(n_samples, n_features).astype(np.float32)
    latents[:, 0] = diagnosis_score + 0.05 * rng.randn(n_samples)

    metadata = pd.DataFrame({
        "diagnosis": pd.Series(diagnosis_score).map({0: "CN", 1: "MCI", 2: "AD"}),
        "age": age,
    })
    annotations = annotate_features(latents, metadata, age_column="age", alpha=0.05)
    assert "category" in annotations.columns
    assert annotations["category"].isin(set(DEFAULT_CATEGORY_MAP.values()) | {"non_specific"}).all()
