#!/usr/bin/env bash
set -euo pipefail
#
# End-to-end Deep SRSNV pipeline:
#   CRAM → tensor caches → 3-fold split → train 3 folds
#   → shared LUT recalibration → TRT inference → report
#
# Usage:
#   bash src/srsnv/scripts/run_e2e_dnn_pipeline.sh
#
# Environment variables to skip steps (resume from failures):
#   SKIP_TENSOR_CACHE=1   - skip steps 1-2 (tensor cache build)
#   SKIP_COMBINE=1        - skip step 3 (combine_splits)
#   SKIP_TRAINING=1       - skip step 4 (fold training)
#   SKIP_RECALIBRATE=1    - skip step 5 (shared LUT recalibration)
#   SKIP_INFERENCE=1      - skip step 6 (VCF inference)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export UV_LINK_MODE=copy

# ── Paths ──────────────────────────────────────────────────────────
WORKDIR="/data/Runs/perchik/ppmseq_data/srsnv_training_workspace"
INPUTS="${WORKDIR}/inputs"
OUTPUT="${WORKDIR}/output/e2e_dnn"

SOURCE_CRAM="${INPUTS}/source.cram"
POS_PARQUET="${INPUTS}/positive.parquet"
NEG_PARQUET="${INPUTS}/negative.parquet"
TRAINING_REGIONS="${INPUTS}/training_regions.interval_list.gz"
STATS_POS="${INPUTS}/stats_positive.json"
STATS_NEG="${INPUTS}/stats_negative.json"
STATS_RAW="${INPUTS}/stats_featuremap.json"
MEAN_COVERAGE=112

PRETRAINED_CKPT="${WORKDIR}/output/dnn_chrom_val.dnn_model_fold_0_swa.ckpt"
FEATUREMAP_VCF="/data/Runs/perchik/ppmseq_data/23A03846_bc_90/featuremap_random_sample/23A03846_bc_90.random_sample.featuremap.vcf.gz"

POS_CACHE="${OUTPUT}/tensor_cache/positive_cache"
NEG_CACHE="${OUTPUT}/tensor_cache/negative_cache"
FOLDS_DIR="${OUTPUT}/tensor_cache/folds"

K_FOLDS=3
EPOCHS=1
BATCH_SIZE=1024
BASENAME="e2e"

# XGB parquet for report (must be from same sample as DNN training)
XGB_PARQUET="${WORKDIR}/output/xgb_shared_split.featuremap_df.parquet"
XGB_METADATA="${WORKDIR}/output/xgb_shared_split.srsnv_metadata.json"
DNN_REPORT_BASENAME="${BASENAME}_dnn"

# ── Skip flags ─────────────────────────────────────────────────────
SKIP_TENSOR_CACHE="${SKIP_TENSOR_CACHE:-0}"
SKIP_COMBINE="${SKIP_COMBINE:-0}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
SKIP_RECALIBRATE="${SKIP_RECALIBRATE:-0}"
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"

elapsed() {
  local t=$SECONDS
  printf '%02d:%02d:%02d' $((t/3600)) $(((t%3600)/60)) $((t%60))
}

echo "============================================"
echo "  E2E Deep SRSNV Pipeline"
echo "  Output: ${OUTPUT}"
echo "  Pretrained: $(basename ${PRETRAINED_CKPT})"
echo "  Folds: ${K_FOLDS}  Epochs: ${EPOCHS}"
echo "============================================"
echo ""

mkdir -p "${OUTPUT}"
cd "${REPO_ROOT}"
SECONDS=0

# ══════════════════════════════════════════════════════════════════
# Step 1: Build positive tensor cache
# ══════════════════════════════════════════════════════════════════
if [[ "${SKIP_TENSOR_CACHE}" == "1" ]]; then
  echo "[1/7] Skipping positive tensor cache (SKIP_TENSOR_CACHE=1)"
else
  echo "[1/7] Building positive tensor cache... ($(elapsed))"
  uv run cram_to_tensors \
    --cram "${SOURCE_CRAM}" \
    --parquet "${POS_PARQUET}" \
    --label positive \
    --output "${POS_CACHE}"
  echo "  Done. ($(elapsed))"
fi
echo ""

# ══════════════════════════════════════════════════════════════════
# Step 2: Build negative tensor cache
# ══════════════════════════════════════════════════════════════════
if [[ "${SKIP_TENSOR_CACHE}" == "1" ]]; then
  echo "[2/7] Skipping negative tensor cache (SKIP_TENSOR_CACHE=1)"
else
  echo "[2/7] Building negative tensor cache... ($(elapsed))"
  uv run cram_to_tensors \
    --cram "${SOURCE_CRAM}" \
    --parquet "${NEG_PARQUET}" \
    --label negative \
    --output "${NEG_CACHE}"
  echo "  Done. ($(elapsed))"
fi
echo ""

# ══════════════════════════════════════════════════════════════════
# Step 3: Combine + k-fold split
# ══════════════════════════════════════════════════════════════════
if [[ "${SKIP_COMBINE}" == "1" || "${SKIP_TENSOR_CACHE}" == "1" ]]; then
  echo "[3/7] Skipping combine_splits (SKIP_COMBINE=1 or SKIP_TENSOR_CACHE=1)"
  if [[ ! -d "${FOLDS_DIR}/fold_0" ]]; then
    echo "  ERROR: ${FOLDS_DIR}/fold_0 does not exist. Cannot skip this step."
    exit 1
  fi
else
  echo "[3/7] Combining caches and splitting into ${K_FOLDS} folds... ($(elapsed))"
  uv run combine_splits \
    --positive "${POS_CACHE}" \
    --negative "${NEG_CACHE}" \
    --training-regions "${TRAINING_REGIONS}" \
    --k-folds "${K_FOLDS}" \
    --holdout-chromosomes "chr21,chr22" \
    --output "${FOLDS_DIR}"
  echo "  Done. ($(elapsed))"
fi
echo ""

# ══════════════════════════════════════════════════════════════════
# Step 4: Train each fold
# ══════════════════════════════════════════════════════════════════
if [[ "${SKIP_TRAINING}" == "1" ]]; then
  echo "[4/7] Skipping training (SKIP_TRAINING=1)"
else
  for i in $(seq 0 $((K_FOLDS - 1))); do
    echo "[4/7] Training fold ${i}/${K_FOLDS}... ($(elapsed))"
    uv run srsnv_dnn_bam_training \
      --fold-dir "${FOLDS_DIR}/fold_${i}" \
      --pretrained-checkpoint "${PRETRAINED_CKPT}" \
      --training-regions "${TRAINING_REGIONS}" \
      --stats-positive "${STATS_POS}" \
      --stats-negative "${STATS_NEG}" \
      --stats-featuremap "${STATS_RAW}" \
      --mean-coverage "${MEAN_COVERAGE}" \
      --single-model-split \
      --epochs "${EPOCHS}" \
      --patience 99 \
      --min-epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --lr-scheduler cosine \
      --use-tf32 \
      --devices auto \
      --basename "${BASENAME}_fold_${i}" \
      --output "${OUTPUT}"
    echo "  Fold ${i} done. ($(elapsed))"
    echo ""
  done
fi
echo ""

# ══════════════════════════════════════════════════════════════════
# Step 5: Build shared MQUAL→SNVQ LUT across all folds
# ══════════════════════════════════════════════════════════════════
if [[ "${SKIP_RECALIBRATE}" == "1" ]]; then
  echo "[5/7] Skipping shared LUT recalibration (SKIP_RECALIBRATE=1)"
else
  echo "[5/7] Building shared MQUAL→SNVQ LUT... ($(elapsed))"

  FOLD_PARQUETS=""
  FOLD_METADATA=""
  for i in $(seq 0 $((K_FOLDS - 1))); do
    FOLD_PARQUETS="${FOLD_PARQUETS} ${OUTPUT}/${BASENAME}_fold_${i}.featuremap_df.parquet"
    FOLD_METADATA="${FOLD_METADATA} ${OUTPUT}/${BASENAME}_fold_${i}.srsnv_dnn_metadata.json"
  done

  uv run recalibrate_dnn_folds \
    --fold-parquets ${FOLD_PARQUETS} \
    --fold-metadata ${FOLD_METADATA} \
    --stats-positive "${STATS_POS}" \
    --stats-negative "${STATS_NEG}" \
    --stats-featuremap "${STATS_RAW}" \
    --training-regions "${TRAINING_REGIONS}" \
    --mean-coverage "${MEAN_COVERAGE}" \
    --output-dir "${OUTPUT}" \
    --basename "${BASENAME}"

  echo "  Done. ($(elapsed))"
fi
echo ""

# ══════════════════════════════════════════════════════════════════
# Step 6: Build ensemble manifest + TRT inference on featuremap VCF
# ══════════════════════════════════════════════════════════════════
ENSEMBLE_MANIFEST="${OUTPUT}/${BASENAME}_ensemble_manifest.json"
INFERENCE_OUTPUT="${OUTPUT}/${BASENAME}_inference_output.vcf.gz"

if [[ "${SKIP_INFERENCE}" == "1" ]]; then
  echo "[6/7] Skipping inference (SKIP_INFERENCE=1)"
else
  echo "[6/7] Building ensemble manifest and running inference... ($(elapsed))"

  # 6a: Build ensemble manifest from fold metadata + split manifest
  uv run python3 -c "
import json
from pathlib import Path

output = Path('${OUTPUT}')
basename = '${BASENAME}'
k_folds = ${K_FOLDS}
folds_dir = Path('${FOLDS_DIR}')

split_manifest_path = folds_dir / 'split_manifest.json'
split_manifest = json.loads(split_manifest_path.read_text())
chrom_to_fold = split_manifest.get('chrom_to_fold', {})

folds = []
for i in range(k_folds):
    meta_path = output / f'{basename}_fold_{i}.srsnv_dnn_metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'Missing metadata: {meta_path}')
    folds.append({'fold_idx': i, 'metadata_path': str(meta_path)})

meta_0 = json.loads(Path(folds[0]['metadata_path']).read_text())
recal_table = meta_0.get('quality_recalibration_table')

manifest = {
    'k_folds': k_folds,
    'chrom_to_fold': chrom_to_fold,
    'folds': folds,
}
if recal_table:
    manifest['quality_recalibration_table'] = recal_table

manifest_path = output / f'{basename}_ensemble_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2))
print(f'Ensemble manifest written: {manifest_path}')
"

  # 6b: Determine backend (TRT if available, else pytorch)
  BACKEND="trt"
  if ! command -v trtexec &>/dev/null; then
    echo "  trtexec not found, falling back to pytorch backend"
    BACKEND="pytorch"
  fi

  # 6c: Run inference
  echo "  Running DNN inference (backend=${BACKEND})..."
  uv run dnn_vcf_inference \
    --featuremap-vcf "${FEATUREMAP_VCF}" \
    --cram "${SOURCE_CRAM}" \
    --ensemble-manifest "${ENSEMBLE_MANIFEST}" \
    --output "${INFERENCE_OUTPUT}" \
    --backend "${BACKEND}"

  echo "  Inference done. Output: ${INFERENCE_OUTPUT} ($(elapsed))"
fi
echo ""

# ══════════════════════════════════════════════════════════════════
# Step 7: Merge with XGB parquet and generate srsnv_report
# ══════════════════════════════════════════════════════════════════
echo "[7/7] Generating DNN report via srsnv_report... ($(elapsed))"

# 7a: Merge DNN predictions into XGB parquet for srsnv_report compatibility
DNN_FOLD_META=""
for i in $(seq 0 $((K_FOLDS - 1))); do
  DNN_FOLD_META="${DNN_FOLD_META} ${OUTPUT}/${BASENAME}_fold_${i}.srsnv_dnn_metadata.json"
done

uv run prepare_dnn_report \
  --xgb-parquet "${XGB_PARQUET}" \
  --dnn-parquet "${OUTPUT}/${BASENAME}.featuremap_df.parquet" \
  --xgb-metadata "${XGB_METADATA}" \
  --dnn-metadata "${OUTPUT}/${BASENAME}_fold_0.srsnv_dnn_metadata.json" \
  --dnn-fold-metadata ${DNN_FOLD_META} \
  --output-dir "${OUTPUT}" \
  --basename "${DNN_REPORT_BASENAME}"

# 7b: Generate the standard srsnv_report HTML
uv run srsnv_report \
  --featuremap-df "${OUTPUT}/${DNN_REPORT_BASENAME}.featuremap_df.parquet" \
  --srsnv-metadata "${OUTPUT}/${DNN_REPORT_BASENAME}.srsnv_metadata.json" \
  --report-path "${OUTPUT}" \
  --basename "${DNN_REPORT_BASENAME}"

echo "  Report: ${OUTPUT}/${DNN_REPORT_BASENAME}.srsnv_report.html"
echo ""

echo "============================================"
echo "  E2E Pipeline Complete! ($(elapsed))"
echo ""
echo "  Outputs:"
echo "    Folds:       ${FOLDS_DIR}/"
echo "    Models:      ${OUTPUT}/${BASENAME}_fold_*.ckpt"
echo "    Shared LUT:  ${OUTPUT}/${BASENAME}.featuremap_df.parquet"
echo "    Manifest:    ${ENSEMBLE_MANIFEST}"
echo "    Inference:   ${INFERENCE_OUTPUT}"
echo "    Report:      ${OUTPUT}/${DNN_REPORT_BASENAME}.srsnv_report.html"
echo "============================================"
