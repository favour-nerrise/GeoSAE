# Data Format

GeoSAE consumes three kinds of inputs.

## 1. Embeddings

One `.npy` file per scan, per layer. Each file holds a single CLS-token
embedding as a 1-D array of length `d_model` (e.g. 768 for BrainIAC).

```
features/
└── layer_NN/
    ├── <scan_stem_1>.npy
    ├── <scan_stem_2>.npy
    └── ...
```

Scan stems must match values in the metadata's `MNI_Z_Cropped` column
(after stripping `.nii` / `.nii.gz` extensions and parent directories).

## 2. Metadata CSV

Required columns:

| Column          | Type    | Description                                  |
|-----------------|---------|----------------------------------------------|
| `RID`           | int     | Subject identifier.                          |
| `MNI_Z_Cropped` | string  | Scan path/filename used to derive the stem.  |
| `scan_date`     | date    | Scan date (used for latest-scan-per-subject).|
| `diagnosis`     | string  | One of `CN`, `MCI`, `AD`.                    |
| `age`           | float   | Age in years (used as the deconfounding covariate). |

Optional clinical columns (all are tested if present):

| Column                  | Type   |
|-------------------------|--------|
| `sex`                   | int (1/2 in ADNI; mapped internally) |
| `apoe4_count`           | int    |
| `hypertension`          | 0/1    |
| `hyperlipidemia`        | 0/1    |
| `depression`            | 0/1    |
| `diabetes_type2`        | 0/1    |
| `cardiovascular_disease`| 0/1    |

## 3. Conversion CSV (for MCI-to-AD evaluation)

| Column      | Type | Description                              |
|-------------|------|------------------------------------------|
| `RID`       | int  | Subject identifier (joined to metadata). |
| `converter` | 0/1  | Did this MCI subject convert to AD?      |

Only used by `evaluate.py`. The label is taken at the subject level; the
evaluator pairs it with the latest available MCI scan per subject so that
each subject contributes one observation.
