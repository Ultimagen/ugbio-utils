#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/data/Runs/perchik/ppmseq_data/srsnv_training_workspace"
INPUTS_DIR="${WORKDIR}/inputs"
OUTPUT_DIR="${WORKDIR}/output"

BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-3}"

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
  --basename "23A03846_bc_90.deep_srsnv" \
  --output "${OUTPUT_DIR}" \
  --verbose
