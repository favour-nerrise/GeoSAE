"""Shared fixtures for the GeoSAE test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture(scope="session")
def rng() -> np.random.RandomState:
    return np.random.RandomState(0)


@pytest.fixture
def synthetic_features(rng: np.random.RandomState) -> np.ndarray:
    return rng.randn(64, 16).astype(np.float32)


@pytest.fixture
def synthetic_labels(rng: np.random.RandomState) -> np.ndarray:
    return rng.randint(0, 3, size=64)


@pytest.fixture
def synthetic_metadata() -> pd.DataFrame:
    n = 64
    return pd.DataFrame({
        "RID": np.arange(n),
        "MNI_Z_Cropped": [f"scan_{i:02d}.nii.gz" for i in range(n)],
        "scan_date": ["2020-01-01"] * n,
        "diagnosis": (["CN"] * 22 + ["MCI"] * 22 + ["AD"] * 20),
        "age": np.linspace(60.0, 85.0, n),
        "sex": np.tile([1, 2], n // 2),
        "apoe4_count": np.tile([0, 1, 2], (n + 2) // 3)[:n],
    })


@pytest.fixture
def synthetic_feature_dir(tmp_path: Path, synthetic_features: np.ndarray, synthetic_metadata: pd.DataFrame) -> Path:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for i, row in synthetic_metadata.iterrows():
        stem = Path(row["MNI_Z_Cropped"]).name.replace(".nii.gz", "")
        np.save(feature_dir / f"{stem}.npy", synthetic_features[i])
    return feature_dir


@pytest.fixture
def synthetic_metadata_csv(tmp_path: Path, synthetic_metadata: pd.DataFrame) -> Path:
    csv_path = tmp_path / "metadata.csv"
    synthetic_metadata.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def trained_model_dir(
    tmp_path: Path,
    synthetic_features: np.ndarray,
) -> Path:
    """Saves a randomly-initialized SAE checkpoint matching the train.py format."""
    from geosae.models import SparseAutoencoder

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    sae = SparseAutoencoder(
        input_dim=synthetic_features.shape[1],
        expansion_factor=2,
        topk=4,
        activation="topk",
    )
    torch.save(
        {
            "config": {"expansion_factor": 2, "topk": 4, "activation": "topk"},
            "model_state_dict": sae.state_dict(),
        },
        model_dir / "best_model.pt",
    )
    return model_dir
