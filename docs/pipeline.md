# GeoSAE Pipeline

This document describes the data flow expected by each stage of GeoSAE.
Every stage is invoked as `python -m geosae.stageN_<name> --help`.

## Stage 0 — Foundation-model feature extraction (out of scope)

GeoSAE assumes you have already extracted CLS-token embeddings from your
foundation model. For each scan and each transformer layer you want to
analyze, save one `.npy` file containing the CLS embedding (a 1-D vector
of length `d_model`).

Suggested layout:

```
features/
├── layer_01/
│   ├── 001_S_0001_T1.npy
│   ├── 001_S_0002_T1.npy
│   └── ...
├── layer_02/
└── ...
```

The repo treats each `.npy` filename stem as the scan identifier; the
metadata CSV must include a column whose values match those stems (default
column: `MNI_Z_Cropped`).

The paper uses BrainIAC (a 12-layer ViT, `d_model = 768`); see
<https://github.com/HMSatBWH/BrainIAC>.

## Stage 1 — Geometric prior analysis (`geosae.stage1_prior`)

Computes four geometric properties from a single layer's embeddings —
angular vs. radial η², intrinsic-dim CV, sparsity Gini,
negative-activation fraction — and recommends an activation function via a
per-property vote. The report is the substantive output; the recommendation
is heuristic.

## Stage 2 — Manifold-regularized SAE training (`geosae.stage2_train`)

Builds a k-NN graph on the embedding space (default k = 15), then trains a
sparse autoencoder with manifold smoothness on encoder pre-activations.
Outputs:

- `best_model.pt` — checkpoint (config, model state, training history).
- `train_summary.json` — per-epoch MSE, manifold loss, explained variance,
  alive-feature count.

## Stage 3 — Annotation (`geosae.stage3_annotate`)

Encodes all scans, finds alive features, computes partial-Spearman
correlations of each alive feature against each clinical variable
controlling for age, applies Benjamini–Hochberg FDR over the feature ×
variable grid, and assigns each feature a category. Outputs a CSV.

## Stage 4 — Conversion evaluation (`geosae.stage4_evaluate`)

Subsets to MCI subjects, takes the latest scan per subject, joins with the
conversion labels, and runs 5-fold stratified logistic regression on the
top-16 features by activation frequency. Also runs a random alive-feature
baseline averaged over many draws. Outputs a JSON summary.

## Stage 5 — Brain-region localization (`geosae.stage5_localize`)

Identifies the most-activating samples per feature, runs attention rollout
through the first `--rollout-layers` ViT layers on each, projects the
resulting per-patch importance into volume space via the FM's patch grid,
and reads out per-region z-scores against a user-supplied atlas (NIfTI or
`.npy`). Requires pre-extracted attention weights, one `.npz` per scan
with keys `layer_01`, `layer_02`, ... and values of shape
`(n_heads, n_tokens, n_tokens)`.

The module is FM-agnostic. For BrainIAC (no CLS token, 96^3 input, 16^3
patches), use `--patch-grid 6 6 6` and omit `--has-cls-token`.

## Stage 6 — Cross-cohort replication (`geosae.stage6_cross_cohort`)

Loads a model trained on a source cohort, applies it to a target cohort
without retraining, and reports two replication metrics:

- **Annotation agreement**: Pearson r between per-feature partial-Spearman
  correlations on the two cohorts.
- **Activation consistency**: Spearman r between per-feature mean
  activation magnitudes on the two cohorts.

## Cross-layer driver (`scripts/run_all_layers.sh`)

Runs stages 2–4 for each layer directory. Useful for reproducing the
cross-layer plot (paper Fig. 2).
