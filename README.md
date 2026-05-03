# GeoSAE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Conference: CV4Clinic 2026](https://img.shields.io/badge/CV4Clinic-2026-7B2D8E.svg)](https://cv4clinical.github.io/cv4clinical_2026/)
[![CI](https://github.com/favour-nerrise/GeoSAE/actions/workflows/ci.yml/badge.svg)](https://github.com/favour-nerrise/GeoSAE/actions/workflows/ci.yml)

This repository is the official implementation of `GeoSAE`, a geometry-guided
sparse autoencoder for decomposing pretrained brain MRI foundation-model
embeddings into interpretable, clinically annotated features. Our associated
paper, **"GeoSAE: Geometric Prior-Guided Layer-Wise Sparse Autoencoder
Annotation of Brain MRI Foundation Models"** has been accepted to the
[CV4Clinic 2026 Workshop](https://cv4clinical.github.io/cv4clinical_2026/)
(*Computer Vision for Real-world Clinical Translation*) at CVPR 2026.

> **GeoSAE: Geometric Prior-Guided Layer-Wise Sparse Autoencoder Annotation
> of Brain MRI Foundation Models**
>
> [Favour Nerrise](mailto:fnerrise@stanford.edu),
> [Lucy Yin](mailto:lucyyin@stanford.edu),
> [Mohammad H. Abbasi](mailto:mabbasi@stanford.edu),
> [Kilian M. Pohl](mailto:kpohl@stanford.edu),
> [Ehsan Adeli](mailto:eadeli@stanford.edu)†
>
> Stanford University, Stanford, CA, USA
>
> †Corresponding author.
>
> **Abstract:** Brain MRI foundation models learn rich representations of
> anatomy, but interpreting what clinical information they encode remains an
> open problem. Standard sparse autoencoders (SAEs) suffer from severe
> feature collapse in deep transformer layers, and in Alzheimer's disease
> (AD) research, aging confounds nearly every clinical variable, making
> naïve annotation unreliable. We propose GeoSAE, a geometry-guided SAE
> framework that uses the foundation model's learned manifold structure to
> prevent feature collapse and annotates each surviving feature via
> age-deconfounded partial correlations. Applied to ~14k T1-weighted MRI
> scans from the Alzheimer's Disease Neuroimaging Initiative (ADNI) and the
> Australian Imaging Biomarkers and Lifestyle (AIBL) datasets, GeoSAE
> identifies a compact, fully interpretable feature set that predicts mild
> cognitive impairment (MCI)-to-AD conversion (AUC 0.746) using only 2% of
> the embedding dimensions, while comorbidity-annotated features achieve
> only chance-level performance. The identified features replicate across
> cohorts without retraining (r=0.97) and localize to neuroanatomically
> distinct regions consistent with Braak staging. This shows that
> geometry-guided SAEs can extract interpretable biomarkers from frozen
> brain MRI foundation models.

The pipeline has six stages, each invoked as a Python module:

| Stage | Module                       | Paper section |
|-------|------------------------------|---------------|
| 1 | `geosae.stage1_prior`            | Geometric prior analysis (Sec. 3.1) |
| 2 | `geosae.stage2_train`            | Manifold-regularized SAE training (Sec. 3.2) |
| 3 | `geosae.stage3_annotate`         | Age-deconfounded feature annotation (Sec. 3.3) |
| 4 | `geosae.stage4_evaluate`         | MCI-to-AD conversion evaluation (Sec. 4.3) |
| 5 | `geosae.stage5_localize`         | Brain-region localization via attention rollout (Sec. 4.5) |
| 6 | `geosae.stage6_cross_cohort`     | Cross-cohort replication (Sec. 4.6) |

This repository does **not** bundle BrainIAC or any other foundation model.
It expects pre-extracted CLS-token embeddings (`.npy`, one per scan) and,
for stage 5, pre-extracted attention weights (`.npz`, one per scan).

## Repository Structure

```text
GeoSAE/
├── LICENSE
├── CITATION.cff
├── README.md
├── pyproject.toml
├── requirements.txt
├── geosae/                       # Implementation package
│   ├── __init__.py
│   ├── data.py                   # embedding/metadata loading helpers
│   ├── manifold.py               # k-NN graph + Gaussian kernel weights
│   ├── models.py                 # SparseAutoencoder (topk / relu / jumprelu)
│   ├── selection.py              # top-N alive feature selection
│   ├── io.py                     # JSON I/O helpers
│   ├── stage1_prior.py
│   ├── stage2_train.py
│   ├── stage3_annotate.py
│   ├── stage4_evaluate.py
│   ├── stage5_localize.py
│   └── stage6_cross_cohort.py
├── docs/
│   ├── pipeline.md
│   └── data_format.md
├── scripts/
│   └── run_all_layers.sh         # multi-layer driver (Sec. 4.1)
└── sample_data/                  # tiny synthetic example
```

## Installation

```bash
git clone https://github.com/favour-nerrise/GeoSAE.git
cd GeoSAE
python -m venv .venv
source .venv/bin/activate
pip install -e .
# For stage 5 (brain-region localization): also install nibabel
pip install -e ".[localization]"
```

Requires Python 3.10+. Direct dependencies: `torch`, `numpy`, `pandas`,
`scipy`, `scikit-learn`, `statsmodels`. See [`requirements.txt`](requirements.txt).

## Data Expectations

GeoSAE consumes:

- **Embeddings**: a directory of `.npy` files (one per scan), each holding
  the FM's CLS-token embedding for that scan as a 1-D array.
- **Metadata CSV**: required columns `RID`, `MNI_Z_Cropped`, `scan_date`,
  `diagnosis`, `age`. Optional clinical columns extend the annotation set:
  `sex`, `apoe4_count`, `hypertension`, `hyperlipidemia`, `depression`,
  `diabetes_type2`, `cardiovascular_disease`.
- **Conversion CSV** (stage 4 only): one row per MCI subject with `RID` and
  `converter` (0/1).
- **Attention weights** (stage 5 only): one `.npz` per scan with keys
  `layer_01`, `layer_02`, ... and values of shape
  `(n_heads, n_tokens, n_tokens)`. Stems must match the embedding `.npy`
  filenames.
- **Atlas** (stage 5 only): a NIfTI or `.npy` with integer region labels on
  the same spatial grid as the FM's input volumes, plus a labels CSV with
  columns `label` and `name`.

A tiny synthetic example showing the embedding/metadata layout is in
[`sample_data/`](sample_data/). See [`docs/data_format.md`](docs/data_format.md)
for the full specification.

## Reproducing the Paper

The paper trains GeoSAE for each of the 12 BrainIAC transformer layers and
evaluates per-layer behavior. Extract CLS embeddings yourself before running
GeoSAE; see BrainIAC's repo at <https://github.com/HMSatBWH/BrainIAC>.

### Stage 1 — Geometric prior analysis (Sec. 3.1)

```bash
python -m geosae.stage1_prior \
  --feature-dir /path/to/features/layer_09 \
  --metadata-csv /path/to/metadata.csv \
  --label-column diagnosis \
  --output-json output/prior_layer09.json
```

Reports angular and radial η², intrinsic-dim mean/CV, sparsity Gini, and
negative-activation fraction, then recommends an activation function
(`topk`, `relu`, or `jumprelu`).

### Stage 2 — Manifold-regularized SAE training (Sec. 3.2, Eq. 3)

```bash
python -m geosae.stage2_train \
  --feature-dir /path/to/features/layer_09 \
  --metadata-csv /path/to/metadata.csv \
  --output-dir output/geosae_layer09 \
  --expansion-factor 2 \
  --topk 16 \
  --activation topk \
  --knn-k 15 \
  --lambda-manifold 0.1 \
  --epochs 100
```

Reproducing ablations:

- Standard SAE baseline: `--lambda-manifold 0`.
- ReLU SAE row in Table 2: `--activation relu`.
- Architecture sweeps: `--expansion-factor {2,4,8}`, `--topk {8,16,32}`.

### Stage 3 — Age-deconfounded annotation (Sec. 3.3)

```bash
python -m geosae.stage3_annotate \
  --feature-dir /path/to/features/layer_09 \
  --metadata-csv /path/to/metadata.csv \
  --model-dir output/geosae_layer09 \
  --output-csv output/geosae_layer09/feature_annotations.csv
```

Produces a CSV with one row per alive feature: activation frequency, best
clinical variable, category (`ad_related`, `sex_related`, `genetic`,
`comorbidity`, `non_specific`), partial-Spearman r, and FDR p-value.

### Stage 4 — MCI-to-AD conversion (Sec. 4.3)

```bash
python -m geosae.stage4_evaluate \
  --feature-dir /path/to/features/layer_09 \
  --metadata-csv /path/to/metadata.csv \
  --conversion-csv /path/to/mci_conversion_labels.csv \
  --model-dir output/geosae_layer09 \
  --annotations-csv output/geosae_layer09/feature_annotations.csv \
  --output-json output/geosae_layer09/conversion_results.json
```

Reports 5-fold stratified CV AUC for the top-16 features by activation
frequency, plus a random alive-feature baseline averaged over multiple
draws.

### Stage 5 — Brain-region localization (Sec. 4.5)

Requires pre-extracted attention weights and the `[localization]` extra.

#### Building the atlas

The paper uses the FSL Harvard-Oxford cortical and subcortical atlases
(probabilistic 1mm, 25% threshold), fetched via nilearn and merged into a
single label volume on the FM's input grid (96^3 for BrainIAC). The repo
provides this as a one-liner; run it once, then point stage 5 at the
generated `.npy` and `.csv`.

```bash
pip install -e ".[localization]"            # nilearn + nibabel
python -m geosae.atlases \
  --output-dir /path/to/atlas \
  --target-shape 96 96 96
# Writes: harvard_oxford_combined.npy, harvard_oxford_combined.csv
```

You can also point stage 5 directly at any user-supplied NIfTI atlas as long
as you provide a labels CSV with columns `label` and `name`.

#### Running stage 5

```bash
python -m geosae.stage5_localize \
  --feature-dir /path/to/features/layer_09 \
  --metadata-csv /path/to/metadata.csv \
  --attention-dir /path/to/attention \
  --model-dir output/geosae_layer09 \
  --atlas-path /path/to/atlas/harvard_oxford_combined.npy \
  --atlas-labels-csv /path/to/atlas/harvard_oxford_combined.csv \
  --annotations-csv output/geosae_layer09/feature_annotations.csv \
  --output-json output/geosae_layer09/localization.json \
  --rollout-layers 9 \
  --patch-grid 6 6 6
```

Pass `--has-cls-token` only if your FM prepends a CLS token (BrainIAC does
not). The output JSON contains, per feature, a z-scored region-importance
map and the top-10 regions by z-score.

### Stage 6 — Cross-cohort replication (Sec. 4.6)

```bash
python -m geosae.stage6_cross_cohort \
  --source-feature-dir /path/to/adni/features/layer_09 \
  --source-metadata-csv /path/to/adni_metadata.csv \
  --target-feature-dir /path/to/aibl/features/layer_09 \
  --target-metadata-csv /path/to/aibl_metadata.csv \
  --model-dir output/geosae_layer09 \
  --output-json output/cross_cohort_layer09.json
```

Applies the source-trained SAE to the target cohort without retraining and
reports annotation agreement (Pearson r between per-feature partial
correlations) and activation consistency (Spearman r between per-feature
mean activations).

### Cross-layer driver (Sec. 4.1)

```bash
FEATURES_ROOT=/path/to/features \
METADATA_CSV=/path/to/metadata.csv \
CONVERSION_CSV=/path/to/mci_conversion_labels.csv \
OUTPUT_ROOT=output \
bash scripts/run_all_layers.sh
```

Loops stages 2–4 over `layer_01` … `layer_12`. Override `LAYERS`,
`LAMBDA_MANIFOLD`, `TOPK`, etc. via env vars.

## Design Notes

- **Manifold regularization is on pre-activations**, not on TopK-gated
  latents. This is the core mechanism that prevents feature mortality under
  hard sparsity gates. See `geosae/stage2_train.py` and Sec. 3.2 of the
  paper.
- **Alive feature**: any feature with a positive activation on at least one
  sample. Only alive features are annotated and selected.
- **Selection**: top-N alive features by activation frequency (paper uses
  N = 16). The repo also supports a random alive-feature baseline.
- **Age deconfounding**: partial Spearman of feature × variable, ranking
  both and the age covariate, with FDR (Benjamini–Hochberg) correction
  across the full feature × variable grid.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

## Acknowledgments

This work was supported in part by National Institutes of Health grants
AG089169, AG084471, AG073356, DA057567, and AA021697, and the Stanford
Institute for Human-Centered AI (HAI) Hoffman-Yee Award. F.N. was supported
by the Stanford Graduate Fellowship, Stanford NeuroTech Graduate Training
Program Fellowship, Stanford EDGE Fellowship, and Stanford Pathways to
Neuroscience Fellowship.

### Data acknowledgments

**ADNI.** Data used in preparation of this article were obtained from the
Alzheimer's Disease Neuroimaging Initiative (ADNI) database
([adni.loni.usc.edu](https://adni.loni.usc.edu)). As such, the investigators
within the ADNI contributed to the design and implementation of ADNI and/or
provided data but did not participate in analysis or writing of this report.
A complete listing of ADNI investigators can be found at:
<https://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf>.

**AIBL.** Data used in the preparation of this article was obtained from the
Australian Imaging Biomarkers and Lifestyle flagship study of ageing (AIBL)
funded by the Commonwealth Scientific and Industrial Research Organisation
(CSIRO), which was made available at the ADNI database
([adni.loni.usc.edu](https://adni.loni.usc.edu)). The AIBL researchers
contributed data but did not participate in analysis or writing of this
report. AIBL researchers are listed at <https://aibl.csiro.au>.

## Citation

```bibtex
@inproceedings{nerrise2026geosae,
  title     = {{GeoSAE}: Geometric Prior-Guided Layer-Wise Sparse Autoencoder Annotation of Brain {MRI} Foundation Models},
  author    = {Nerrise, Favour and Yin, Lucy and Abbasi, Mohammad H. and Pohl, Kilian M. and Adeli, Ehsan},
  booktitle = {Proceedings of the CVPR Workshop on Computer Vision for Real-world Clinical Translation (CV4Clinic)},
  year      = {2026}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
