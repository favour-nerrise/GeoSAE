"""Feature subset selection utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def top_features_by_frequency(
    annotations: pd.DataFrame,
    n_features: int,
) -> np.ndarray:
    """Selects the top features by activation frequency.

    Args:
      annotations: Feature annotation table.
      n_features: Number of features to select.

    Returns:
      Selected latent indices.
    """
    subset = annotations.sort_values(
        ["activation_frequency", "feature_index"], ascending=[False, True])
    return subset["feature_index"].head(n_features).to_numpy(dtype=np.int64)


def random_alive_features(
    annotations: pd.DataFrame,
    n_features: int,
    random_state: int,
) -> np.ndarray:
    """Samples alive features uniformly at random.

    Args:
      annotations: Feature annotation table.
      n_features: Number of features to sample.
      random_state: Random seed.

    Returns:
      Randomly selected feature indices.
    """
    rng = np.random.RandomState(random_state)
    candidates = annotations["feature_index"].to_numpy(dtype=np.int64)
    return rng.choice(candidates, size=n_features, replace=False)

