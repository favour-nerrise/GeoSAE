"""Atlas helpers for stage 5 (brain-region localization).

The paper uses the FSL Harvard-Oxford cortical and subcortical atlases
fetched via nilearn, merged into a single label volume on the same grid as
the FM's input volumes (96^3 for BrainIAC). This module provides the same
fetch + merge + resample step as a reusable utility so users do not have to
re-derive it.

Requires the optional ``[localization]`` extra (nilearn, nibabel).
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from geosae import io


@dataclasses.dataclass(frozen=True)
class HarvardOxfordAtlas:
    """Combined cortical + subcortical Harvard-Oxford atlas at a target shape."""

    labels: np.ndarray  # (X, Y, Z) integer label volume; 0 = background
    label_table: pd.DataFrame  # columns: label, type, name


def fetch_harvard_oxford_combined(
    target_shape: tuple[int, int, int] = (96, 96, 96),
    cortical_threshold: int = 25,
) -> HarvardOxfordAtlas:
    """Fetches the Harvard-Oxford cortical + subcortical atlas via nilearn.

    Cortical labels keep their original IDs. Subcortical labels are offset by
    the cortical maximum so a single label volume covers both. Resampling to
    ``target_shape`` is nearest-neighbor so labels remain integer-valued.

    Args:
      target_shape: Spatial shape to resample the atlas to (e.g. ``(96, 96, 96)``).
      cortical_threshold: Probability threshold (in percent) for the cortical
        atlas. The paper uses 25.

    Returns:
      A ``HarvardOxfordAtlas`` with the merged label volume and a label table.
    """
    try:
        import nibabel as nib  # noqa: F401
        from nilearn import datasets
        from scipy.ndimage import zoom
    except ImportError as exc:
        raise ImportError(
            "fetch_harvard_oxford_combined requires nilearn and nibabel. "
            "Install with `pip install -e \".[localization]\"`.") from exc

    cortical = datasets.fetch_atlas_harvard_oxford(
        f"cort-maxprob-thr{cortical_threshold}-1mm",
        symmetric_split=False,
    )
    subcortical = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")

    cortical_data = _atlas_to_array(cortical.maps)
    subcortical_data = _atlas_to_array(subcortical.maps)

    cortical_resampled = _resample_nearest(cortical_data, target_shape)
    subcortical_resampled = _resample_nearest(subcortical_data, target_shape)

    cortical_max = int(cortical_resampled.max())
    merged = cortical_resampled.copy()
    fill = (subcortical_resampled > 0) & (merged == 0)
    merged[fill] = subcortical_resampled[fill] + cortical_max

    rows: list[dict[str, object]] = []
    for index, name in enumerate(cortical.labels):
        if index == 0:
            continue
        rows.append({"label": index, "type": "cortical", "name": name})
    for index, name in enumerate(subcortical.labels):
        if index == 0:
            continue
        rows.append({"label": index + cortical_max, "type": "subcortical", "name": name})
    label_table = pd.DataFrame(rows)
    return HarvardOxfordAtlas(labels=merged.astype(np.int32), label_table=label_table)


def _atlas_to_array(maps_obj: object) -> np.ndarray:
    """Loads a nilearn atlas map into a numpy array of integer labels."""
    if hasattr(maps_obj, "get_fdata"):
        return maps_obj.get_fdata().astype(np.int32)
    import nibabel as nib
    return nib.load(str(maps_obj)).get_fdata().astype(np.int32)


def _resample_nearest(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Nearest-neighbor resampling preserves integer label values."""
    from scipy.ndimage import zoom
    factors = tuple(target / source for target, source in zip(target_shape, volume.shape))
    return zoom(volume, factors, order=0).astype(np.int32)


def save_atlas(atlas: HarvardOxfordAtlas, output_dir: Path) -> tuple[Path, Path]:
    """Writes the merged atlas volume as ``.npy`` and the label table as CSV."""
    io.ensure_dir(output_dir)
    npy_path = output_dir / "harvard_oxford_combined.npy"
    csv_path = output_dir / "harvard_oxford_combined.csv"
    np.save(npy_path, atlas.labels)
    atlas.label_table.to_csv(csv_path, index=False)
    return npy_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and save the Harvard-Oxford combined atlas at a target shape.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write harvard_oxford_combined.{npy,csv}.")
    parser.add_argument("--target-shape", type=int, nargs=3, default=[96, 96, 96],
                        help="Spatial shape to resample to. Default 96 96 96 (BrainIAC).")
    parser.add_argument("--cortical-threshold", type=int, default=25,
                        help="Probability threshold (percent) for the cortical atlas.")
    args = parser.parse_args()

    atlas = fetch_harvard_oxford_combined(
        target_shape=tuple(args.target_shape),
        cortical_threshold=args.cortical_threshold,
    )
    npy_path, csv_path = save_atlas(atlas, args.output_dir)
    print(f"Wrote atlas: {npy_path}")
    print(f"Wrote labels: {csv_path}")
    print(f"Regions: {len(atlas.label_table)} ({(atlas.label_table['type'] == 'cortical').sum()} cortical + {(atlas.label_table['type'] == 'subcortical').sum()} subcortical)")


if __name__ == "__main__":
    main()
