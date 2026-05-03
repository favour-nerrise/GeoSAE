"""Tests for stage 5 (brain-region localization)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from geosae.models import SparseAutoencoder
from geosae.stage5_localize import (
    LocalizationConfig,
    attention_rollout,
    patches_to_volume,
    run_localization,
)


def test_attention_rollout_is_row_stochastic() -> None:
    rng = np.random.RandomState(0)
    layers = [
        rng.dirichlet(np.ones(6), size=(2, 6)).astype(np.float32) for _ in range(3)
    ]
    rolled = attention_rollout(layers)
    assert rolled.shape == (6, 6)
    assert np.allclose(rolled.sum(axis=-1), 1.0, atol=1e-3)


def test_attention_rollout_rejects_empty() -> None:
    with pytest.raises(ValueError):
        attention_rollout([])


def test_patches_to_volume_shape() -> None:
    patches = np.arange(8, dtype=np.float32)
    volume = patches_to_volume(patches, patch_grid=(2, 2, 2), volume_shape=(8, 8, 8))
    assert volume.shape == (8, 8, 8)


def test_patches_to_volume_rejects_wrong_count() -> None:
    with pytest.raises(ValueError):
        patches_to_volume(np.zeros(7), patch_grid=(2, 2, 2), volume_shape=(8, 8, 8))


def test_run_localization_end_to_end(tmp_path: Path) -> None:
    rng = np.random.RandomState(0)
    n_scans = 8
    d_model = 16
    patch_grid = (2, 2, 2)
    n_tokens = int(np.prod(patch_grid))
    n_layers = 2
    volume_shape = (8, 8, 8)

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    attention_dir = tmp_path / "attention"
    attention_dir.mkdir()
    stems = [f"scan_{i:02d}" for i in range(n_scans)]
    for stem in stems:
        np.save(feature_dir / f"{stem}.npy", rng.randn(d_model).astype(np.float32))
        np.savez(
            attention_dir / f"{stem}.npz",
            **{
                f"layer_{layer + 1:02d}": rng.dirichlet(
                    np.ones(n_tokens), size=(2, n_tokens)).astype(np.float32)
                for layer in range(n_layers)
            },
        )

    metadata_csv = tmp_path / "metadata.csv"
    pd.DataFrame({
        "RID": range(n_scans),
        "MNI_Z_Cropped": [f"{s}.nii.gz" for s in stems],
        "scan_date": ["2020-01-01"] * n_scans,
        "diagnosis": ["MCI"] * n_scans,
        "age": np.linspace(60, 80, n_scans),
    }).to_csv(metadata_csv, index=False)

    atlas = np.zeros(volume_shape, dtype=np.int32)
    atlas[:, :, :4] = 1
    atlas[:, :, 4:] = 2
    atlas_path = tmp_path / "atlas.npy"
    np.save(atlas_path, atlas)
    labels_csv = tmp_path / "atlas_labels.csv"
    pd.DataFrame({"label": [1, 2], "name": ["A", "B"]}).to_csv(labels_csv, index=False)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    sae = SparseAutoencoder(input_dim=d_model, expansion_factor=2, topk=4, activation="topk")
    torch.save(
        {"config": {"expansion_factor": 2, "topk": 4, "activation": "topk"},
         "model_state_dict": sae.state_dict()},
        model_dir / "best_model.pt",
    )

    config = LocalizationConfig(
        feature_dir=feature_dir,
        metadata_csv=metadata_csv,
        attention_dir=attention_dir,
        model_dir=model_dir,
        atlas_path=atlas_path,
        atlas_labels_csv=labels_csv,
        output_json=tmp_path / "localization.json",
        n_top_features=2,
        n_max_activating=4,
        rollout_layers=n_layers,
        patch_grid=patch_grid,
        has_cls_token=False,
    )
    summary = run_localization(config)
    assert summary["n_features"] == 2
    assert (tmp_path / "localization.json").exists()
    for feature in summary["features"]:
        assert "region_importance_named" in feature
        assert set(feature["region_importance_named"]).issubset({"A", "B"})
