"""Data loading and alignment utilities for GeoSAE."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class EmbeddingTable:
    """Aligned embedding matrix and metadata.

    Attributes:
      features: Array of shape `(n_samples, d_model)`.
      metadata: Metadata rows aligned to `features`.
      stems: Scan stems aligned to `features`.
    """

    features: np.ndarray
    metadata: pd.DataFrame
    stems: list[str]


def stem_from_path(value: str) -> str:
    """Extracts a scan stem from a metadata path-like value.

    Args:
      value: Path or filename stored in metadata.

    Returns:
      Filename stem with `.nii` and `.nii.gz` removed.
    """
    stem = Path(value).name
    if stem.endswith(".nii.gz"):
        return stem[:-7]
    if stem.endswith(".nii"):
        return stem[:-4]
    return Path(stem).stem


def load_metadata(metadata_csv: Path) -> pd.DataFrame:
    """Loads metadata from CSV.

    Args:
      metadata_csv: Input metadata CSV path.

    Returns:
      Loaded DataFrame.
    """
    return pd.read_csv(metadata_csv, low_memory=False)


def load_feature_directory(
    feature_dir: Path,
    metadata_csv: Path,
    stem_column: str = "MNI_Z_Cropped",
) -> EmbeddingTable:
    """Loads per-scan embeddings and aligns them to metadata.

    Args:
      feature_dir: Directory containing one `.npy` file per scan.
      metadata_csv: Metadata CSV path.
      stem_column: Metadata column used to derive scan stems.

    Returns:
      Aligned embeddings and metadata.

    Raises:
      ValueError: If required columns are missing or no aligned samples exist.
    """
    metadata = load_metadata(metadata_csv)
    if stem_column not in metadata.columns:
        raise ValueError(f"Missing required metadata column: {stem_column}")

    metadata = metadata.copy()
    metadata["_stem"] = metadata[stem_column].astype(str).map(stem_from_path)
    metadata_by_stem = {
        stem: row for stem, row in metadata.drop_duplicates("_stem").set_index("_stem").iterrows()
    }

    feature_rows = []
    meta_rows = []
    stems = []
    for path in sorted(feature_dir.glob("*.npy")):
        stem = path.stem
        if stem not in metadata_by_stem:
            continue
        feature_rows.append(np.load(path).astype(np.float32))
        meta_rows.append(metadata_by_stem[stem])
        stems.append(stem)

    if not feature_rows:
        raise ValueError(
            "No embeddings could be aligned to metadata. Check stems and paths.")

    features = np.stack(feature_rows).astype(np.float32)
    aligned_metadata = pd.DataFrame(meta_rows).reset_index(drop=True)
    return EmbeddingTable(features=features, metadata=aligned_metadata, stems=stems)


def latest_scan_per_subject(
    metadata: pd.DataFrame,
    subject_column: str = "RID",
    date_column: str = "scan_date",
) -> pd.Series:
    """Returns row indices for the latest scan per subject.

    Args:
      metadata: Scan-level metadata.
      subject_column: Subject identifier column.
      date_column: Scan date column.

    Returns:
      Integer row indices for the latest scan per subject.
    """
    sortable = metadata.copy()
    sortable[date_column] = pd.to_datetime(sortable[date_column], errors="coerce")
    sortable = sortable.sort_values([subject_column, date_column])
    return sortable.groupby(subject_column, sort=False).tail(1).index
