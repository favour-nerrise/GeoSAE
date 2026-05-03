#!/usr/bin/env bash
# Train, annotate, and evaluate GeoSAE across all transformer layers.
#
# Expected layout:
#   ${FEATURES_ROOT}/layer_NN/<scan_stem>.npy  for NN in 01..12
#
# Usage:
#   FEATURES_ROOT=/path/to/features \
#   METADATA_CSV=/path/to/metadata.csv \
#   CONVERSION_CSV=/path/to/mci_conversion_labels.csv \
#   OUTPUT_ROOT=/path/to/output \
#   bash scripts/run_all_layers.sh

set -euo pipefail

: "${FEATURES_ROOT:?FEATURES_ROOT must be set}"
: "${METADATA_CSV:?METADATA_CSV must be set}"
: "${CONVERSION_CSV:?CONVERSION_CSV must be set}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"

EXPANSION_FACTOR="${EXPANSION_FACTOR:-2}"
TOPK="${TOPK:-16}"
KNN_K="${KNN_K:-15}"
LAMBDA_MANIFOLD="${LAMBDA_MANIFOLD:-0.1}"
EPOCHS="${EPOCHS:-100}"
ACTIVATION="${ACTIVATION:-topk}"
LAYERS="${LAYERS:-01 02 03 04 05 06 07 08 09 10 11 12}"

for layer in ${LAYERS}; do
  feature_dir="${FEATURES_ROOT}/layer_${layer}"
  output_dir="${OUTPUT_ROOT}/geosae_layer${layer}"
  if [ ! -d "${feature_dir}" ]; then
    echo "Skipping layer ${layer}: ${feature_dir} not found."
    continue
  fi

  echo "=== Layer ${layer}: stage 2 (train) ==="
  python -m geosae.stage2_train \
    --feature-dir "${feature_dir}" \
    --metadata-csv "${METADATA_CSV}" \
    --output-dir "${output_dir}" \
    --expansion-factor "${EXPANSION_FACTOR}" \
    --topk "${TOPK}" \
    --activation "${ACTIVATION}" \
    --knn-k "${KNN_K}" \
    --lambda-manifold "${LAMBDA_MANIFOLD}" \
    --epochs "${EPOCHS}"

  echo "=== Layer ${layer}: stage 3 (annotate) ==="
  python -m geosae.stage3_annotate \
    --feature-dir "${feature_dir}" \
    --metadata-csv "${METADATA_CSV}" \
    --model-dir "${output_dir}" \
    --output-csv "${output_dir}/feature_annotations.csv"

  echo "=== Layer ${layer}: stage 4 (evaluate) ==="
  python -m geosae.stage4_evaluate \
    --feature-dir "${feature_dir}" \
    --metadata-csv "${METADATA_CSV}" \
    --conversion-csv "${CONVERSION_CSV}" \
    --model-dir "${output_dir}" \
    --annotations-csv "${output_dir}/feature_annotations.csv" \
    --output-json "${output_dir}/conversion_results.json"
done

echo "All layers complete. Outputs in ${OUTPUT_ROOT}/."
