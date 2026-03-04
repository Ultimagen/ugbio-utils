#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/data/Runs/perchik/ppmseq_data/srsnv_training_workspace}"
INPUTS_DIR="${WORKDIR}/inputs"
OUTPUT_DIR="${WORKDIR}/output"
CONFIG_FILE="${INPUTS_DIR}/dataset_config.json"

# Read basename from dataset_config.json if present
if [[ -f "${CONFIG_FILE}" ]]; then
  CONFIG_BASENAME="$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['basename'])")"
  DEFAULT_BASENAME="${CONFIG_BASENAME}.deep_srsnv"
else
  DEFAULT_BASENAME="deep_srsnv"
fi

BASENAME="${BASENAME:-${DEFAULT_BASENAME}}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-3}"

echo "=== Deep SRSNV DNN Training ==="
echo "  Basename:  ${BASENAME}"
echo "  Workspace: ${WORKDIR}"
echo ""

uv run srsnv_dnn_bam_training \
  --positive-bam "${INPUTS_DIR}/positive_reads.bam" \
  --negative-bam "${INPUTS_DIR}/negative_reads.bam" \
  --positive-parquet "${INPUTS_DIR}/positive.parquet" \
  --negative-parquet "${INPUTS_DIR}/negative.parquet" \
  --training-regions "${INPUTS_DIR}/training_regions.interval_list.gz" \
  --single-model-split \
  --holdout-chromosomes "chr21,chr22" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --use-amp \
  --use-tf32 \
  --basename "${BASENAME}" \
  --output "${OUTPUT_DIR}" \
  --verbose
