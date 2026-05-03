"""Clinical annotation for GeoSAE features."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.stats.multitest
import torch

from geosae import data
from geosae import io
from geosae.models import SparseAutoencoder


DEFAULT_CATEGORY_MAP = {
    "diagnosis": "ad_related",
    "sex": "sex_related",
    "sex_binary": "sex_related",
    "apoe4_count": "genetic",
    "hypertension": "comorbidity",
    "hyperlipidemia": "comorbidity",
    "depression": "comorbidity",
    "diabetes_type2": "comorbidity",
    "cardiovascular_disease": "comorbidity",
}


@dataclasses.dataclass(frozen=True)
class AnnotationConfig:
    """Configuration for feature annotation."""

    metadata_csv: Path
    feature_dir: Path
    model_dir: Path
    output_csv: Path
    stem_column: str = "MNI_Z_Cropped"
    age_column: str = "age"
    alpha: float = 0.05


def _rank_residuals(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """Regresses a ranked variable on a ranked covariate and returns residuals."""
    ranked_values = scipy.stats.rankdata(values)
    ranked_covariate = scipy.stats.rankdata(covariate)
    slope = np.cov(ranked_values, ranked_covariate)[0, 1] / (
        np.var(ranked_covariate) + 1e-12)
    return ranked_values - slope * ranked_covariate


def partial_spearman(x: np.ndarray, y: np.ndarray, covariate: np.ndarray) -> tuple[float, float]:
    """Computes age-deconfounded partial Spearman correlation.

    Args:
      x: Feature values.
      y: Clinical variable values.
      covariate: Age covariate values.

    Returns:
      Tuple `(rho, p_value)`.
    """
    x_residual = _rank_residuals(x, covariate)
    y_residual = _rank_residuals(y, covariate)
    if np.std(x_residual) < 1e-12 or np.std(y_residual) < 1e-12:
        return 0.0, 1.0
    return scipy.stats.pearsonr(x_residual, y_residual)


def load_model(model_dir: Path, input_dim: int) -> SparseAutoencoder:
    """Loads a trained GeoSAE checkpoint, regardless of activation."""
    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = SparseAutoencoder(
        input_dim=input_dim,
        expansion_factor=config["expansion_factor"],
        topk=config["topk"],
        activation=config.get("activation", "topk"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


_load_model = load_model  # backwards-compatible alias


def _clinical_variables(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    """Builds the set of clinical variables that will be tested."""
    variables = {}
    if "diagnosis" in metadata.columns:
        variables["diagnosis"] = metadata["diagnosis"].map(
            {"CN": 0, "MCI": 1, "AD": 2}).to_numpy()
    if "sex" in metadata.columns:
        variables["sex_binary"] = (metadata["sex"] == 2).astype(float).to_numpy()
    for name in [
        "apoe4_count",
        "hypertension",
        "hyperlipidemia",
        "depression",
        "diabetes_type2",
        "cardiovascular_disease",
    ]:
        if name in metadata.columns:
            variables[name] = pd.to_numeric(
                metadata[name], errors="coerce").to_numpy()
    return variables


def annotate_features(
    latents: np.ndarray,
    metadata: pd.DataFrame,
    age_column: str,
    alpha: float,
) -> pd.DataFrame:
    """Annotates alive features by their strongest significant clinical association.

    Args:
      latents: Sparse latent matrix.
      metadata: Metadata aligned to `latents`.
      age_column: Age covariate column name.
      alpha: FDR threshold.

    Returns:
      Annotation table for alive features only.
    """
    age = pd.to_numeric(metadata[age_column], errors="coerce").to_numpy()
    alive_indices = np.where((latents > 0).any(axis=0))[0]
    alive_latents = latents[:, alive_indices]
    variables = _clinical_variables(metadata)

    p_matrix = np.ones((len(alive_indices), len(variables)), dtype=np.float64)
    rho_matrix = np.zeros_like(p_matrix)
    variable_names = list(variables.keys())

    for feature_index in range(len(alive_indices)):
        feature_values = alive_latents[:, feature_index]
        for variable_index, variable_name in enumerate(variable_names):
            variable_values = variables[variable_name]
            valid_mask = (
                ~np.isnan(feature_values) &
                ~np.isnan(variable_values) &
                ~np.isnan(age))
            if valid_mask.sum() < 25:
                continue
            rho, p_value = partial_spearman(
                feature_values[valid_mask],
                variable_values[valid_mask],
                age[valid_mask],
            )
            rho_matrix[feature_index, variable_index] = rho
            p_matrix[feature_index, variable_index] = p_value

    rejected, corrected_p, _, _ = statsmodels.stats.multitest.multipletests(
        p_matrix.ravel(), alpha=alpha, method="fdr_bh")
    rejected = rejected.reshape(p_matrix.shape)
    corrected_p = corrected_p.reshape(p_matrix.shape)

    rows = []
    for feature_index, latent_index in enumerate(alive_indices):
        significant = np.where(rejected[feature_index])[0]
        if len(significant) == 0:
            best_variable = "non_specific"
            best_category = "non_specific"
            best_rho = 0.0
            best_p = 1.0
        else:
            best_idx = significant[np.argmax(
                np.abs(rho_matrix[feature_index, significant]))]
            best_variable = variable_names[best_idx]
            best_category = DEFAULT_CATEGORY_MAP.get(best_variable, "other")
            best_rho = float(rho_matrix[feature_index, best_idx])
            best_p = float(corrected_p[feature_index, best_idx])
        rows.append({
            "feature_index": int(latent_index),
            "activation_frequency": float((latents[:, latent_index] > 0).mean()),
            "best_variable": best_variable,
            "category": best_category,
            "partial_spearman_r": best_rho,
            "fdr_p_value": best_p,
        })

    return pd.DataFrame(rows).sort_values(
        ["activation_frequency", "feature_index"], ascending=[False, True])


def run_annotation(config: AnnotationConfig) -> pd.DataFrame:
    """Loads data and writes the annotation CSV."""
    embedding_table = data.load_feature_directory(
        feature_dir=config.feature_dir,
        metadata_csv=config.metadata_csv,
        stem_column=config.stem_column,
    )
    model = load_model(config.model_dir, embedding_table.features.shape[1])
    with torch.no_grad():
        latents, _ = model.encode(torch.from_numpy(embedding_table.features))
    annotation_df = annotate_features(
        latents.numpy(),
        embedding_table.metadata,
        config.age_column,
        config.alpha,
    )
    io.ensure_dir(config.output_csv.parent)
    annotation_df.to_csv(config.output_csv, index=False)
    return annotation_df


def main() -> None:
    """Parses arguments and runs GeoSAE feature annotation."""
    parser = argparse.ArgumentParser(description="Annotate GeoSAE features.")
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--age-column", type=str, default="age")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    config = AnnotationConfig(
        metadata_csv=args.metadata_csv,
        feature_dir=args.feature_dir,
        model_dir=args.model_dir,
        output_csv=args.output_csv,
        age_column=args.age_column,
        alpha=args.alpha,
    )
    run_annotation(config)


if __name__ == "__main__":
    main()
