"""Brain-region localization via attention rollout (paper Sec. 4.5).

For each selected SAE feature, we identify the maximally activating samples,
roll out the FM's self-attention through the first ``rollout_layers`` layers
on each of those samples, project the resulting per-patch importance into
volume space via the FM's patch grid, accumulate (activation-weighted) over
samples, then read out per-region importance against an atlas.

The module is FM-agnostic. It does not load BrainIAC or any other foundation
model. It expects:

  - Pre-extracted attention weights, one ``.npz`` per scan, with keys
    ``layer_01`` ... ``layer_LL`` and values of shape
    ``(n_heads, n_tokens, n_tokens)``. The scan stems must match the
    embedding ``.npy`` filenames consumed by the rest of the pipeline.
  - An atlas NIfTI (or ``.npy``) with integer region labels, on the same
    spatial grid as the FM's input volumes (e.g. 96^3 for BrainIAC).
  - A labels CSV with columns ``label`` and ``name``.

The paper uses BrainIAC, which has no CLS token: the first token is treated
as the readout. Pass ``--has-cls-token`` if your FM prepends a CLS token,
in which case the readout is token 0 and patches are tokens [1:].
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom

from geosae import data
from geosae import io
from geosae.stage3_annotate import load_model
from geosae.selection import top_features_by_frequency


@dataclasses.dataclass(frozen=True)
class LocalizationConfig:
    """Configuration for brain-region localization."""

    feature_dir: Path
    metadata_csv: Path
    attention_dir: Path
    model_dir: Path
    atlas_path: Path
    atlas_labels_csv: Path
    output_json: Path
    n_top_features: int = 16
    n_max_activating: int = 50
    rollout_layers: int = 9
    patch_grid: tuple[int, int, int] = (6, 6, 6)
    has_cls_token: bool = False
    annotations_csv: Path | None = None


def attention_rollout(
    attention_per_layer: list[np.ndarray],
    add_residual: bool = True,
) -> np.ndarray:
    """Computes attention rollout across layers.

    Args:
      attention_per_layer: list of arrays of shape ``(n_heads, n_tokens, n_tokens)``.
      add_residual: if True, mix each layer's attention with identity (0.5/0.5)
        to model the residual stream, then row-normalize. This is the standard
        rollout (Abnar & Zuidema, 2020).

    Returns:
      The (n_tokens, n_tokens) rolled-out matrix. ``out[i, j]`` is the
      cumulative attention from token ``i`` to token ``j``.
    """
    if not attention_per_layer:
        raise ValueError("attention_per_layer must contain at least one layer.")
    rolled = None
    for layer_attn in attention_per_layer:
        attn = layer_attn.mean(axis=0)
        if add_residual:
            identity = np.eye(attn.shape[-1], dtype=attn.dtype)
            attn = 0.5 * attn + 0.5 * identity
        attn = attn / np.clip(attn.sum(axis=-1, keepdims=True), 1e-12, None)
        rolled = attn if rolled is None else attn @ rolled
    return rolled


def patches_to_volume(
    patch_importance: np.ndarray,
    patch_grid: tuple[int, int, int],
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    """Reshapes a per-patch vector to volume space via trilinear upsampling."""
    expected = int(np.prod(patch_grid))
    if patch_importance.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} patches for grid {patch_grid}, got {patch_importance.shape[0]}.")
    grid_volume = patch_importance.reshape(patch_grid)
    zoom_factors = tuple(v / g for v, g in zip(volume_shape, patch_grid))
    return zoom(grid_volume, zoom_factors, order=1)


def _load_atlas(atlas_path: Path) -> np.ndarray:
    """Loads a 3D integer atlas from a ``.npy`` or NIfTI file."""
    if atlas_path.suffix == ".npy":
        return np.load(atlas_path).astype(np.int32)
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "Loading NIfTI atlases requires `nibabel`. "
            "Install with `pip install geosae[localization]`.") from exc
    return nib.load(str(atlas_path)).get_fdata().astype(np.int32)


def _load_attention(
    attention_path: Path,
    rollout_layers: int,
) -> list[np.ndarray]:
    """Loads attention for the first ``rollout_layers`` ViT layers from .npz."""
    archive = np.load(attention_path)
    layers = []
    for i in range(1, rollout_layers + 1):
        key = f"layer_{i:02d}"
        if key not in archive:
            raise KeyError(
                f"{attention_path} is missing key '{key}' "
                f"(expected layer_01..layer_{rollout_layers:02d}).")
        layers.append(archive[key])
    return layers


def _select_features(
    latents: np.ndarray,
    config: LocalizationConfig,
) -> np.ndarray:
    """Returns the feature indices to localize."""
    alive = np.where((latents > 0).any(axis=0))[0]
    if config.annotations_csv is not None:
        annotations = pd.read_csv(config.annotations_csv)
        return top_features_by_frequency(annotations, config.n_top_features)
    activation_frequency = (latents[:, alive] > 0).mean(axis=0)
    order = np.argsort(activation_frequency)[::-1][: config.n_top_features]
    return alive[order]


def localize_feature(
    feature_index: int,
    latents: np.ndarray,
    stems: list[str],
    config: LocalizationConfig,
    atlas: np.ndarray,
) -> dict[str, object]:
    """Computes per-region importance for one SAE feature."""
    activations = latents[:, feature_index]
    candidates = np.argsort(activations)[::-1]
    accumulator = np.zeros(atlas.shape, dtype=np.float64)
    n_processed = 0

    for sample_index in candidates:
        if n_processed >= config.n_max_activating:
            break
        if activations[sample_index] <= 0:
            break
        stem = stems[sample_index]
        attention_path = config.attention_dir / f"{stem}.npz"
        if not attention_path.exists():
            continue
        per_layer = _load_attention(attention_path, config.rollout_layers)
        rolled = attention_rollout(per_layer)
        readout_row = rolled[0]
        patches = readout_row[1:] if config.has_cls_token else readout_row
        volume = patches_to_volume(patches, config.patch_grid, atlas.shape)
        accumulator += volume * float(activations[sample_index])
        n_processed += 1

    if n_processed == 0:
        return {
            "feature_index": int(feature_index),
            "n_activating_samples": 0,
            "region_importance": {},
            "top_regions": [],
        }

    accumulator /= n_processed
    importance: dict[str, float] = {}
    return _collect_region_importance(
        feature_index=feature_index,
        accumulator=accumulator,
        atlas=atlas,
        n_processed=n_processed,
        importance=importance,
    )


def _collect_region_importance(
    feature_index: int,
    accumulator: np.ndarray,
    atlas: np.ndarray,
    n_processed: int,
    importance: dict[str, float],
) -> dict[str, object]:
    region_means: dict[int, float] = {}
    for region_label in np.unique(atlas):
        if region_label == 0:
            continue
        mask = atlas == region_label
        if not mask.any():
            continue
        region_means[int(region_label)] = float(accumulator[mask].mean())
    if region_means:
        values = np.array(list(region_means.values()))
        center = float(values.mean())
        spread = float(values.std()) or 1.0
        for label, value in region_means.items():
            importance[str(label)] = (value - center) / spread
    top = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        "feature_index": int(feature_index),
        "n_activating_samples": int(n_processed),
        "region_importance": importance,
        "top_regions": [{"label": int(k), "z_score": float(v)} for k, v in top],
    }


def run_localization(config: LocalizationConfig) -> dict[str, object]:
    """Runs attention rollout and atlas mapping for the selected features."""
    table = data.load_feature_directory(config.feature_dir, config.metadata_csv)
    model = load_model(config.model_dir, table.features.shape[1])
    with torch.no_grad():
        latents, _ = model.encode(torch.from_numpy(table.features))
    latents = latents.numpy()

    atlas = _load_atlas(config.atlas_path)
    feature_indices = _select_features(latents, config)
    label_names = pd.read_csv(config.atlas_labels_csv).set_index("label")["name"].to_dict()

    results = []
    for feature_index in feature_indices:
        record = localize_feature(
            feature_index=int(feature_index),
            latents=latents,
            stems=table.stems,
            config=config,
            atlas=atlas,
        )
        record["region_importance_named"] = {
            label_names.get(int(k), f"label_{k}"): v
            for k, v in record["region_importance"].items()
        }
        record["top_regions_named"] = [
            {"name": label_names.get(item["label"], f"label_{item['label']}"),
             "z_score": item["z_score"]}
            for item in record["top_regions"]
        ]
        results.append(record)

    summary = {
        "atlas_path": str(config.atlas_path),
        "n_features": len(results),
        "rollout_layers": config.rollout_layers,
        "patch_grid": list(config.patch_grid),
        "has_cls_token": config.has_cls_token,
        "n_max_activating": config.n_max_activating,
        "features": results,
    }
    io.write_json(summary, config.output_json)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brain-region localization via attention rollout.")
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--attention-dir", type=Path, required=True,
                        help="Directory of one .npz per scan with keys layer_01..layer_NN.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--atlas-path", type=Path, required=True,
                        help="Atlas .nii(.gz) or .npy with integer labels.")
    parser.add_argument("--atlas-labels-csv", type=Path, required=True,
                        help="CSV with columns 'label' and 'name'.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--annotations-csv", type=Path, default=None,
                        help="If provided, localize the top features by activation "
                             "frequency from this annotation CSV; else top from latents.")
    parser.add_argument("--n-top-features", type=int, default=16)
    parser.add_argument("--n-max-activating", type=int, default=50)
    parser.add_argument("--rollout-layers", type=int, default=9)
    parser.add_argument("--patch-grid", type=int, nargs=3, default=[6, 6, 6])
    parser.add_argument("--has-cls-token", action="store_true",
                        help="Pass if the FM prepends a CLS token; "
                             "patches are then tokens [1:].")
    args = parser.parse_args()

    config = LocalizationConfig(
        feature_dir=args.feature_dir,
        metadata_csv=args.metadata_csv,
        attention_dir=args.attention_dir,
        model_dir=args.model_dir,
        atlas_path=args.atlas_path,
        atlas_labels_csv=args.atlas_labels_csv,
        output_json=args.output_json,
        n_top_features=args.n_top_features,
        n_max_activating=args.n_max_activating,
        rollout_layers=args.rollout_layers,
        patch_grid=tuple(args.patch_grid),
        has_cls_token=args.has_cls_token,
        annotations_csv=args.annotations_csv,
    )
    run_localization(config)


if __name__ == "__main__":
    main()
