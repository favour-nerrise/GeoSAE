"""Downstream evaluation utilities for GeoSAE."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import torch

from geosae import data
from geosae import io
from geosae.stage3_annotate import load_model
from geosae.selection import random_alive_features
from geosae.selection import top_features_by_frequency


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for conversion evaluation."""

    feature_dir: Path
    metadata_csv: Path
    conversion_csv: Path
    model_dir: Path
    annotations_csv: Path
    output_json: Path
    n_features: int = 16
    n_folds: int = 5
    random_draws: int = 100
    seed: int = 42


def _latest_mci_subjects(
    metadata: pd.DataFrame,
    conversion_labels: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns row indices and labels for the latest MCI scan per subject."""
    mci = metadata[metadata["diagnosis"] == "MCI"].copy()
    mci["scan_date"] = pd.to_datetime(mci["scan_date"], errors="coerce")
    latest = mci.sort_values(["RID", "scan_date"]).groupby("RID", sort=False).tail(1)
    merged = latest.merge(
        conversion_labels[["RID", "converter"]],
        on="RID",
        how="inner",
    )
    valid = ~pd.isna(merged["converter"])
    merged = merged.loc[valid].reset_index(drop=True)
    return merged["index"].to_numpy(dtype=np.int64), merged["converter"].to_numpy(dtype=np.int64)


def logistic_regression_cv(
    features: np.ndarray,
    labels: np.ndarray,
    n_folds: int,
    seed: int,
) -> dict[str, float]:
    """Runs stratified logistic-regression evaluation.

    Args:
      features: Feature matrix.
      labels: Binary labels.
      n_folds: Number of CV folds.
      seed: Random seed.

    Returns:
      Aggregate metrics across folds.
    """
    cv = sklearn.model_selection.StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )
    fold_aucs = []
    fold_sensitivities = []
    fold_specificities = []
    for train_index, test_index in cv.split(features, labels):
        model = sklearn.linear_model.LogisticRegression(max_iter=1000)
        model.fit(features[train_index], labels[train_index])
        probabilities = model.predict_proba(features[test_index])[:, 1]
        predictions = model.predict(features[test_index])
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
            labels[test_index], predictions).ravel()
        fold_aucs.append(
            sklearn.metrics.roc_auc_score(labels[test_index], probabilities))
        fold_sensitivities.append(tp / max(tp + fn, 1))
        fold_specificities.append(tn / max(tn + fp, 1))
    return {
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs)),
        "sensitivity_mean": float(np.mean(fold_sensitivities)),
        "sensitivity_std": float(np.std(fold_sensitivities)),
        "specificity_mean": float(np.mean(fold_specificities)),
        "specificity_std": float(np.std(fold_specificities)),
    }


def run_evaluation(config: EvaluationConfig) -> dict[str, float]:
    """Runs the paper's feature-selection and conversion evaluation pipeline."""
    embedding_table = data.load_feature_directory(
        feature_dir=config.feature_dir,
        metadata_csv=config.metadata_csv,
    )
    conversion_labels = pd.read_csv(config.conversion_csv)
    annotations = pd.read_csv(config.annotations_csv)
    model = load_model(config.model_dir, embedding_table.features.shape[1])

    with torch.no_grad():
        latents, _ = model.encode(torch.from_numpy(embedding_table.features))
    latents = latents.numpy()
    metadata = embedding_table.metadata.reset_index(drop=False)
    subject_indices, labels = _latest_mci_subjects(
        metadata, conversion_labels)

    top_indices = top_features_by_frequency(annotations, config.n_features)
    top_metrics = logistic_regression_cv(
        latents[subject_indices][:, top_indices],
        labels,
        config.n_folds,
        config.seed,
    )

    random_aucs = []
    for draw in range(config.random_draws):
        sampled = random_alive_features(
            annotations,
            config.n_features,
            random_state=config.seed + draw,
        )
        draw_metrics = logistic_regression_cv(
            latents[subject_indices][:, sampled],
            labels,
            config.n_folds,
            config.seed,
        )
        random_aucs.append(draw_metrics["auc_mean"])

    results = {
        "n_subjects": int(len(labels)),
        "n_converters": int(labels.sum()),
        "n_stable": int(len(labels) - labels.sum()),
        "top16_auc_mean": top_metrics["auc_mean"],
        "top16_auc_std": top_metrics["auc_std"],
        "top16_sensitivity_mean": top_metrics["sensitivity_mean"],
        "top16_specificity_mean": top_metrics["specificity_mean"],
        "random16_auc_mean": float(np.mean(random_aucs)),
        "random16_auc_std": float(np.std(random_aucs)),
    }
    io.write_json(results, config.output_json)
    return results


def main() -> None:
    """Parses arguments and runs conversion evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate GeoSAE on conversion prediction.")
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--conversion-csv", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--annotations-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = EvaluationConfig(
        feature_dir=args.feature_dir,
        metadata_csv=args.metadata_csv,
        conversion_csv=args.conversion_csv,
        model_dir=args.model_dir,
        annotations_csv=args.annotations_csv,
        output_json=args.output_json,
        n_features=args.n_features,
        n_folds=args.n_folds,
        random_draws=args.random_draws,
        seed=args.seed,
    )
    run_evaluation(config)


if __name__ == "__main__":
    main()
