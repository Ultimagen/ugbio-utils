#!/usr/bin/env bash
set -euo pipefail
#
# End-to-end Deep SRSNV pipeline:
#   CRAM → tensor caches → 3-fold split → train 3 folds → TRT inference → report
#
# Usage:
#   bash src/srsnv/scripts/run_e2e_dnn_pipeline.sh
#
# Environment variables to skip steps (resume from failures):
#   SKIP_TENSOR_CACHE=1   - skip steps 1-2 (tensor cache build)
#   SKIP_COMBINE=1        - skip step 3 (combine_splits)
#   SKIP_TRAINING=1       - skip step 4 (fold training)
#   SKIP_INFERENCE=1      - skip step 5 (VCF inference)
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
EPOCHS=2
BATCH_SIZE=1024
BASENAME="e2e"

# ── Skip flags ─────────────────────────────────────────────────────
SKIP_TENSOR_CACHE="${SKIP_TENSOR_CACHE:-0}"
SKIP_COMBINE="${SKIP_COMBINE:-0}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
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
  echo "[1/6] Skipping positive tensor cache (SKIP_TENSOR_CACHE=1)"
else
  echo "[1/6] Building positive tensor cache... ($(elapsed))"
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
  echo "[2/6] Skipping negative tensor cache (SKIP_TENSOR_CACHE=1)"
else
  echo "[2/6] Building negative tensor cache... ($(elapsed))"
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
  echo "[3/6] Skipping combine_splits (SKIP_COMBINE=1 or SKIP_TENSOR_CACHE=1)"
  if [[ ! -d "${FOLDS_DIR}/fold_0" ]]; then
    echo "  ERROR: ${FOLDS_DIR}/fold_0 does not exist. Cannot skip this step."
    exit 1
  fi
else
  echo "[3/6] Combining caches and splitting into ${K_FOLDS} folds... ($(elapsed))"
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
# Step 4: Train each fold (2 epochs, pretrained)
# ══════════════════════════════════════════════════════════════════
if [[ "${SKIP_TRAINING}" == "1" ]]; then
  echo "[4/6] Skipping training (SKIP_TRAINING=1)"
else
  for i in $(seq 0 $((K_FOLDS - 1))); do
    echo "[4/6] Training fold ${i}/${K_FOLDS}... ($(elapsed))"
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
# Step 5: Build ensemble manifest + TRT inference on featuremap VCF
# ══════════════════════════════════════════════════════════════════
ENSEMBLE_MANIFEST="${OUTPUT}/${BASENAME}_ensemble_manifest.json"
INFERENCE_OUTPUT="${OUTPUT}/${BASENAME}_inference_output.vcf.gz"

if [[ "${SKIP_INFERENCE}" == "1" ]]; then
  echo "[5/6] Skipping inference (SKIP_INFERENCE=1)"
else
  echo "[5/6] Building ensemble manifest and running inference... ($(elapsed))"

  # 5a: Build ensemble manifest from fold metadata + split manifest
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

  # 5b: Determine backend (TRT if available, else pytorch)
  BACKEND="trt"
  if ! command -v trtexec &>/dev/null; then
    echo "  trtexec not found, falling back to pytorch backend"
    BACKEND="pytorch"
  fi

  # 5c: Run inference
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
# Step 6: Combine fold parquets + generate comparison report
# ══════════════════════════════════════════════════════════════════
COMBINED_PARQUET="${OUTPUT}/${BASENAME}_combined.featuremap_df.parquet"
REPORT_HTML="${OUTPUT}/${BASENAME}_dnn_report.html"

echo "[6/6] Generating comparison report... ($(elapsed))"

# 6a: Combine fold parquets (each fold has val+test predictions)
uv run python3 -c "
import polars as pl
from pathlib import Path

output = Path('${OUTPUT}')
basename = '${BASENAME}'
k_folds = ${K_FOLDS}

frames = []
for i in range(k_folds):
    p = output / f'{basename}_fold_{i}.featuremap_df.parquet'
    if p.exists():
        df = pl.read_parquet(p)
        print(f'  fold_{i}: {len(df)} rows')
        frames.append(df)
    else:
        print(f'  WARNING: {p} not found, skipping')

if not frames:
    raise RuntimeError('No fold parquets found')

combined = pl.concat(frames)

test_mask = combined['fold_id'] == -1
test_rows = combined.filter(test_mask)
non_test_rows = combined.filter(~test_mask)

test_deduped = test_rows.unique(subset=['CHROM', 'POS', 'RN'], keep='first')

final = pl.concat([non_test_rows, test_deduped])
print(f'  Combined: {len(final)} rows ({len(non_test_rows)} val + {len(test_deduped)} test)')

combined_path = output / f'{basename}_combined.featuremap_df.parquet'
final.write_parquet(combined_path)
print(f'  Written: {combined_path}')
"

# 6b: Generate HTML report
uv run compare_models_report \
  --model "name=DNN_e2e,parquet=${COMBINED_PARQUET},metadata=${OUTPUT}/${BASENAME}_fold_0.srsnv_dnn_metadata.json" \
  --output "${REPORT_HTML}"

echo "  Report: ${REPORT_HTML}"
echo ""

echo "============================================"
echo "  E2E Pipeline Complete! ($(elapsed))"
echo ""
echo "  Outputs:"
echo "    Folds:     ${FOLDS_DIR}/"
echo "    Models:    ${OUTPUT}/${BASENAME}_fold_*.ckpt"
echo "    Manifest:  ${ENSEMBLE_MANIFEST}"
echo "    Inference: ${INFERENCE_OUTPUT}"
echo "    Report:    ${REPORT_HTML}"
echo "============================================"
