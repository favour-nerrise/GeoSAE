"""Geometric prior analysis for SAE activation selection (paper Sec. 3.1).

This module reports the four geometric properties that GeoSAE uses to choose
an SAE activation function and to motivate the manifold prior:

  1. Angular vs. radial class separation (eta^2). Large angular eta^2 with
     small radial eta^2 indicates that class structure lives on directions
     rather than norms; this favors selection-based gates (TopK / JumpReLU)
     over simplex-based methods (e.g. SpaDE).
  2. Intrinsic-dimension homogeneity. Low coefficient of variation of local
     intrinsic dimension across samples indicates that the manifold has a
     consistent local geometry, which justifies a single global activation
     and a single ``k`` in TopK across all samples.
  3. Sparsity uniformity. Low Gini of per-coordinate activation magnitudes
     indicates that information is distributed across coordinates rather
     than concentrated in a few. Distributed signal disfavors simplex
     activations and favors selection-based gates.
  4. Significant negative pre-activations. Many SAE activations (ReLU,
     JumpReLU, SpaDE) zero out negative pre-activations entirely; if a
     large fraction of pre-activations is negative, those activations
     discard substantial signal. TopK acts on signed values and is
     therefore preferred when the negative fraction is large.

The paper reports four of five properties favor TopK on BrainIAC; the fifth
property (negative pre-activation magnitude statistics) is reported here as
``negative_activation_mean_abs`` for completeness.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from geosae import data
from geosae import io


_ACTIVATION_CHOICES = ("topk", "relu", "jumprelu")


@dataclasses.dataclass(frozen=True)
class GeometricReport:
    """Numerical summary of geometric properties of an embedding matrix."""

    angular_eta2: float
    radial_eta2: float
    intrinsic_dim_mean: float
    intrinsic_dim_cv: float
    sparsity_gini: float
    negative_fraction: float
    negative_activation_mean_abs: float
    n_samples: int
    d_model: int

    def to_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


def _eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way eta^2 between a continuous value and a categorical grouping."""
    valid = ~(np.isnan(values) | pd.isna(groups))
    values = values[valid]
    groups = groups[valid]
    if len(values) < 3 or len(np.unique(groups)) < 2:
        return 0.0
    grand_mean = values.mean()
    ss_total = float(((values - grand_mean) ** 2).sum())
    if ss_total < 1e-12:
        return 0.0
    ss_between = 0.0
    for group_label in np.unique(groups):
        mask = groups == group_label
        if mask.sum() == 0:
            continue
        ss_between += mask.sum() * (values[mask].mean() - grand_mean) ** 2
    return float(ss_between / ss_total)


def angular_radial_eta2(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Returns (angular eta^2, radial eta^2) for class separation.

    Angular: eta^2 between cosine of the principal angle and class.
    Radial: eta^2 between L2 norm and class.
    """
    norms = np.linalg.norm(features, axis=1)
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    directions = features / safe_norms[:, None]
    centroid = directions.mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-12)
    angular_score = directions @ centroid
    return _eta_squared(angular_score, labels), _eta_squared(norms, labels)


def local_intrinsic_dimension(
    features: np.ndarray,
    n_neighbors: int = 20,
    variance_threshold: float = 0.9,
) -> np.ndarray:
    """Per-sample intrinsic dimension via local PCA.

    Returns the number of principal components needed to capture
    ``variance_threshold`` of local variance.
    """
    n_neighbors = min(n_neighbors, len(features) - 1)
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(features)
    _, indices = knn.kneighbors(features)
    dims = np.zeros(len(features), dtype=np.int32)
    n_components = min(features.shape[1], n_neighbors)
    for i in range(len(features)):
        local = features[indices[i, 1:]]
        pca = PCA(n_components=n_components).fit(local)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        dims[i] = int(np.searchsorted(cumulative, variance_threshold) + 1)
    return dims


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector. 0 = uniform, 1 = concentrated."""
    values = np.asarray(values, dtype=np.float64)
    values = values[values >= 0]
    if values.size == 0 or values.sum() < 1e-12:
        return 0.0
    sorted_values = np.sort(values)
    n = sorted_values.size
    cumulative = np.cumsum(sorted_values)
    return float((n + 1 - 2 * cumulative.sum() / cumulative[-1]) / n)


def compute_report(
    features: np.ndarray,
    labels: np.ndarray | None,
    n_neighbors: int = 20,
) -> GeometricReport:
    """Computes the four geometric properties from a feature matrix."""
    n_samples, d_model = features.shape
    if labels is None:
        labels = np.zeros(n_samples, dtype=np.int32)
    angular, radial = angular_radial_eta2(features, labels)
    dims = local_intrinsic_dimension(features, n_neighbors=n_neighbors)
    intrinsic_mean = float(dims.mean())
    intrinsic_cv = float(dims.std() / max(intrinsic_mean, 1e-12))
    feature_norms = np.linalg.norm(features, axis=0)
    sparsity = gini(feature_norms)
    negative_fraction = float((features < 0).mean())
    negative_mean_abs = float(np.abs(features[features < 0]).mean()) if negative_fraction > 0 else 0.0
    return GeometricReport(
        angular_eta2=angular,
        radial_eta2=radial,
        intrinsic_dim_mean=intrinsic_mean,
        intrinsic_dim_cv=intrinsic_cv,
        sparsity_gini=sparsity,
        negative_fraction=negative_fraction,
        negative_activation_mean_abs=negative_mean_abs,
        n_samples=int(n_samples),
        d_model=int(d_model),
    )


def recommend_activation(report: GeometricReport) -> tuple[str, dict[str, str]]:
    """Maps a geometric report to a recommended SAE activation.

    Returns the recommended activation name and a per-property vote dict.
    Output is restricted to the activations actually implemented in
    `geosae.models.SparseAutoencoder` (``topk``, ``relu``, ``jumprelu``).
    The paper analyzed SpaDE as a fourth candidate but ruled it out on
    BrainIAC; this implementation does not provide a SpaDE gate.

    Decision rules (matching paper Sec. 3.1):
      - angular >= radial favors selection-based gates (topk, jumprelu).
      - low intrinsic-dim CV: a single global k is appropriate, favors topk.
      - low sparsity gini: distributed signal favors topk over jumprelu.
      - high negative fraction: signed gates (topk) preferred over relu.
    """
    votes: dict[str, str] = {}
    votes["angular_vs_radial"] = "topk" if report.angular_eta2 >= report.radial_eta2 else "jumprelu"
    votes["intrinsic_dim_homogeneity"] = "topk" if report.intrinsic_dim_cv < 0.25 else "jumprelu"
    votes["sparsity_uniformity"] = "topk" if report.sparsity_gini < 0.35 else "jumprelu"
    votes["negative_activations"] = "topk" if report.negative_fraction > 0.10 else "relu"
    counts: dict[str, int] = {}
    for activation in votes.values():
        counts[activation] = counts.get(activation, 0) + 1
    recommended = max(counts.items(), key=lambda kv: (kv[1], kv[0] == "topk"))[0]
    return recommended, votes


def main() -> None:
    """Runs the geometric prior analysis on a feature directory."""
    parser = argparse.ArgumentParser(
        description="Geometric prior analysis to select an SAE activation.")
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--label-column", type=str, default="diagnosis",
                        help="Categorical column used for angular/radial eta^2.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--n-neighbors", type=int, default=20)
    args = parser.parse_args()

    table = data.load_feature_directory(args.feature_dir, args.metadata_csv)
    labels = (
        table.metadata[args.label_column].to_numpy()
        if args.label_column in table.metadata.columns
        else None
    )
    report = compute_report(table.features, labels, n_neighbors=args.n_neighbors)
    activation, votes = recommend_activation(report)
    payload = {
        "report": report.to_dict(),
        "votes": votes,
        "recommended_activation": activation,
    }
    io.write_json(payload, args.output_json)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
