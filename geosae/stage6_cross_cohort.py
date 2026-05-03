"""Cross-cohort replication of GeoSAE annotations (paper Sec. 4.6).

Apply a SAE trained on cohort A (e.g. ADNI) to cohort B (e.g. AIBL) without
retraining, then report:

  - Annotation agreement: Pearson r between per-feature partial-Spearman
    correlations on the two cohorts.
  - Activation consistency: Spearman r between per-feature mean activation
    magnitudes on the two cohorts.
  - Replication rate: fraction of features that are alive in both cohorts.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import torch

from geosae import data
from geosae import io
from geosae.stage3_annotate import (
    DEFAULT_CATEGORY_MAP,
    _clinical_variables,
    load_model,
    partial_spearman,
)


@dataclasses.dataclass(frozen=True)
class CrossCohortConfig:
    """Configuration for cross-cohort evaluation."""

    source_feature_dir: Path
    source_metadata_csv: Path
    target_feature_dir: Path
    target_metadata_csv: Path
    model_dir: Path
    output_json: Path
    age_column: str = "age"


def _per_feature_partial_correlations(
    latents: np.ndarray,
    metadata: pd.DataFrame,
    age_column: str,
) -> dict[str, np.ndarray]:
    """Computes per-feature partial-Spearman correlations for each clinical variable.

    Returns a mapping from variable name to a 1-D array indexed by feature.
    """
    variables = _clinical_variables(metadata)
    age = pd.to_numeric(metadata[age_column], errors="coerce").to_numpy()
    correlations: dict[str, np.ndarray] = {}
    for variable_name, variable_values in variables.items():
        out = np.zeros(latents.shape[1], dtype=np.float64)
        for feature_index in range(latents.shape[1]):
            feature_values = latents[:, feature_index]
            valid = (
                ~np.isnan(feature_values)
                & ~np.isnan(variable_values)
                & ~np.isnan(age)
            )
            if valid.sum() < 25:
                continue
            rho, _ = partial_spearman(
                feature_values[valid],
                variable_values[valid],
                age[valid],
            )
            out[feature_index] = rho
        correlations[variable_name] = out
    return correlations


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, 1.0
    rho, p_value = scipy.stats.pearsonr(x, y)
    return float(rho), float(p_value)


def _safe_spearmanr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, 1.0
    result = scipy.stats.spearmanr(x, y)
    return float(result.correlation), float(result.pvalue)


def run_cross_cohort(config: CrossCohortConfig) -> dict[str, object]:
    """Runs the cross-cohort comparison and writes a JSON summary."""
    source = data.load_feature_directory(config.source_feature_dir, config.source_metadata_csv)
    target = data.load_feature_directory(config.target_feature_dir, config.target_metadata_csv)
    if source.features.shape[1] != target.features.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: source {source.features.shape[1]} vs target {target.features.shape[1]}.")

    model = load_model(config.model_dir, source.features.shape[1])
    with torch.no_grad():
        source_latents, _ = model.encode(torch.from_numpy(source.features))
        target_latents, _ = model.encode(torch.from_numpy(target.features))
    source_latents = source_latents.numpy()
    target_latents = target_latents.numpy()

    source_alive = (source_latents > 0).any(axis=0)
    target_alive = (target_latents > 0).any(axis=0)
    shared_alive = np.where(source_alive & target_alive)[0]

    source_corr = _per_feature_partial_correlations(
        source_latents, source.metadata, config.age_column)
    target_corr = _per_feature_partial_correlations(
        target_latents, target.metadata, config.age_column)

    annotation_agreement: dict[str, dict[str, float]] = {}
    for variable_name in source_corr:
        if variable_name not in target_corr:
            continue
        source_vec = source_corr[variable_name][shared_alive]
        target_vec = target_corr[variable_name][shared_alive]
        rho, p_value = _safe_pearsonr(source_vec, target_vec)
        annotation_agreement[variable_name] = {"pearson_r": rho, "p_value": p_value}

    source_activation = source_latents[:, shared_alive].mean(axis=0)
    target_activation = target_latents[:, shared_alive].mean(axis=0)
    activation_rho, activation_p = _safe_spearmanr(source_activation, target_activation)

    replication_rate = (
        float(shared_alive.size / max(source_alive.sum(), 1))
        if source_alive.any()
        else 0.0
    )

    summary = {
        "n_source_samples": int(len(source.features)),
        "n_target_samples": int(len(target.features)),
        "n_alive_source": int(source_alive.sum()),
        "n_alive_target": int(target_alive.sum()),
        "n_shared_alive": int(shared_alive.size),
        "replication_rate_source_features_alive_in_target": replication_rate,
        "annotation_agreement": annotation_agreement,
        "activation_consistency_spearman_r": activation_rho,
        "activation_consistency_spearman_p": activation_p,
        "category_map": DEFAULT_CATEGORY_MAP,
    }
    io.write_json(summary, config.output_json)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-cohort replication of GeoSAE annotations.")
    parser.add_argument("--source-feature-dir", type=Path, required=True)
    parser.add_argument("--source-metadata-csv", type=Path, required=True)
    parser.add_argument("--target-feature-dir", type=Path, required=True)
    parser.add_argument("--target-metadata-csv", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--age-column", type=str, default="age")
    args = parser.parse_args()
    config = CrossCohortConfig(
        source_feature_dir=args.source_feature_dir,
        source_metadata_csv=args.source_metadata_csv,
        target_feature_dir=args.target_feature_dir,
        target_metadata_csv=args.target_metadata_csv,
        model_dir=args.model_dir,
        output_json=args.output_json,
        age_column=args.age_column,
    )
    run_cross_cohort(config)


if __name__ == "__main__":
    main()
