#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORKDIR="${WORKDIR:-/data/Runs/perchik/ppmseq_data/srsnv_training_workspace}"
INPUTS_DIR="${WORKDIR}/inputs"
OUTPUT_DIR="${WORKDIR}/output"
CONFIG_FILE="${INPUTS_DIR}/dataset_config.json"

# Read mean coverage and basename from dataset_config.json if present
if [[ -f "${CONFIG_FILE}" ]]; then
  MEAN_COVERAGE="${MEAN_COVERAGE:-$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['mean_coverage'])")}"
  CONFIG_BASENAME="$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['basename'])")"
else
  MEAN_COVERAGE="${MEAN_COVERAGE:-112}"
  CONFIG_BASENAME="unknown"
fi

SPLIT_MANIFEST="${OUTPUT_DIR}/chr20_val_split_manifest.json"
XGB_PARAMS="${SCRIPT_DIR}/xgb_tuned_params.json"

POS_PARQUET="${INPUTS_DIR}/positive.parquet"
NEG_PARQUET="${INPUTS_DIR}/negative.parquet"
SOURCE_CRAM="${INPUTS_DIR}/source.cram"
TRAINING_REGIONS="${INPUTS_DIR}/training_regions.interval_list.gz"
STATS_POS="${INPUTS_DIR}/stats_positive.json"
STATS_NEG="${INPUTS_DIR}/stats_negative.json"
STATS_RAW="${INPUTS_DIR}/stats_featuremap.json"

# Tensor cache paths for new pipeline
POS_CACHE="${OUTPUT_DIR}/tensor_cache/positive_cache"
NEG_CACHE="${OUTPUT_DIR}/tensor_cache/negative_cache"
FOLDS_DIR="${OUTPUT_DIR}/tensor_cache/folds"

XGB_BASENAME="${XGB_BASENAME:-xgb_chrom_val}"
DNN_BASENAME="${DNN_BASENAME:-dnn_chrom_val}"

BATCH_SIZE="${BATCH_SIZE:-1024}"
EPOCHS="${EPOCHS:-12}"
PATIENCE="${PATIENCE:-3}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"

echo "=== SRSNV XGBoost vs DNN comparison ==="
echo "  Dataset:       ${CONFIG_BASENAME}"
echo "  Mean coverage: ${MEAN_COVERAGE}"
echo "  Workspace:     ${WORKDIR}"
echo ""

echo "[1/4] Train XGBoost (tuned params) with chromosome-based val split..."
echo "  Using params: ${XGB_PARAMS}"
uv run srsnv_training \
  --positive "${POS_PARQUET}" \
  --negative "${NEG_PARQUET}" \
  --training-regions "${TRAINING_REGIONS}" \
  --stats-positive "${STATS_POS}" \
  --stats-negative "${STATS_NEG}" \
  --stats-featuremap "${STATS_RAW}" \
  --mean-coverage "${MEAN_COVERAGE}" \
  --single-model-split \
  --val-chromosomes "chr20" \
  --split-manifest-out "${SPLIT_MANIFEST}" \
  --holdout-chromosomes "chr21,chr22" \
  --model-params "${XGB_PARAMS}" \
  --features 'REF:ALT:X_HMER_REF:X_HMER_ALT:X_PREV1:X_NEXT1:X_PREV2:X_NEXT2:X_PREV3:X_NEXT3:BCSQ:BCSQCSS:RL:INDEX:DUP:REV:SCST:SCED:MAPQ:EDIST:SMQ_BEFORE:SMQ_AFTER:tm:rq:st:et' \
  --basename "${XGB_BASENAME}" \
  --output "${OUTPUT_DIR}"

echo "[2/4] Build tensor caches from CRAM and train deep_srsnv CNN..."
echo "  [2a] Building positive tensor cache..."
uv run cram_to_tensors \
  --cram "${SOURCE_CRAM}" \
  --parquet "${POS_PARQUET}" \
  --label positive \
  --output "${POS_CACHE}"

echo "  [2b] Building negative tensor cache..."
uv run cram_to_tensors \
  --cram "${SOURCE_CRAM}" \
  --parquet "${NEG_PARQUET}" \
  --label negative \
  --output "${NEG_CACHE}"

echo "  [2c] Combining and splitting into folds..."
uv run combine_splits \
  --positive "${POS_CACHE}" \
  --negative "${NEG_CACHE}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --single-model-split \
  --output "${FOLDS_DIR}"

echo "  [2d] Training DNN from fold directory..."
uv run srsnv_dnn_bam_training \
  --fold-dir "${FOLDS_DIR}/fold_0" \
  --stats-positive "${STATS_POS}" \
  --stats-negative "${STATS_NEG}" \
  --stats-featuremap "${STATS_RAW}" \
  --mean-coverage "${MEAN_COVERAGE}" \
  --single-model-split \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --lr-scheduler "${LR_SCHEDULER}" \
  --use-tf32 \
  --swa \
  --swa-epoch-start 6 \
  --devices auto \
  --basename "${DNN_BASENAME}" \
  --output "${OUTPUT_DIR}" \
  --verbose

echo "[3/4] Compare metrics on val + holdout (test) sets..."
uv run python - <<PY
import json
import polars as pl
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score

out = Path("${OUTPUT_DIR}")
xgb_df = pl.read_parquet(out / "${XGB_BASENAME}.featuremap_df.parquet")
dnn_df = pl.read_parquet(out / "${DNN_BASENAME}.featuremap_df.parquet")

splits = {
    "val (chr20)": {
        "xgb_mask": xgb_df["fold_id"] == 1,
        "dnn_mask": dnn_df["fold_id"] == 1,
    },
    "test (chr21,chr22)": {
        "xgb_mask": xgb_df["fold_id"].is_null(),
        "dnn_mask": dnn_df["fold_id"] == -1,
    },
}

def calc_metrics(y, p):
    return {
        "auc": roc_auc_score(y, p),
        "aupr": average_precision_score(y, p),
        "logloss": log_loss(y, p, labels=[0, 1]),
    }

header = f"{'Split':<18} {'Metric':<12} {'XGBoost':>12} {'DNN':>12} {'Delta':>12} {'XGB rows':>10} {'DNN rows':>10}"
print(header)
print("-" * len(header))

for split_name, masks in splits.items():
    x_y = xgb_df.filter(masks["xgb_mask"])["label"].cast(pl.Int64).to_numpy()
    x_p = xgb_df.filter(masks["xgb_mask"])["prob_orig"].to_numpy()
    d_y = dnn_df.filter(masks["dnn_mask"])["label"].cast(pl.Int64).to_numpy()
    d_p = dnn_df.filter(masks["dnn_mask"])["prob_orig"].to_numpy()

    xm = calc_metrics(x_y, x_p)
    dm = calc_metrics(d_y, d_p)

    for metric in ["auc", "aupr", "logloss"]:
        print(f"{split_name:<18} {metric:<12} {xm[metric]:>12.6f} {dm[metric]:>12.6f} {dm[metric] - xm[metric]:>+12.6f} {len(x_y):>10} {len(d_y):>10}")
    print()

xgb_meta = json.loads((out / "${XGB_BASENAME}.srsnv_metadata.json").read_text())
dnn_meta = json.loads((out / "${DNN_BASENAME}.srsnv_dnn_metadata.json").read_text())

print("=== XGBoost training summary ===")
xgb_params = xgb_meta.get("model_params", {})
print(f"  n_estimators={xgb_params.get('n_estimators')}, early_stopping_rounds={xgb_params.get('early_stopping_rounds')}")
print(f"  eta={xgb_params.get('eta')}, max_depth={xgb_params.get('max_depth')}")

print("\n=== DNN (Lightning) training summary ===")
dnn_tr = dnn_meta.get("training_results", [])
if isinstance(dnn_tr, list):
    for fold in dnn_tr:
        print(f"  fold={fold.get('fold')} best_epoch={fold.get('best_epoch')}/{fold.get('total_epochs')} "
              f"stopped_early={fold.get('stopped_early')} "
              f"best_val_auc={fold.get('best_val_auc')}")
dnn_params = dnn_meta.get("training_parameters", {})
print(f"  epochs={dnn_params.get('epochs')}, patience={dnn_params.get('patience')}, "
      f"batch_size={dnn_params.get('batch_size')}, lr_scheduler={dnn_params.get('lr_scheduler')}")
PY

echo "[4/4] Generate HTML comparison report..."
uv run python -m ugbio_srsnv.compare_models_report \
  --xgb-parquet "${OUTPUT_DIR}/${XGB_BASENAME}.featuremap_df.parquet" \
  --dnn-parquet "${OUTPUT_DIR}/${DNN_BASENAME}.featuremap_df.parquet" \
  --xgb-metadata "${OUTPUT_DIR}/${XGB_BASENAME}.srsnv_metadata.json" \
  --dnn-metadata "${OUTPUT_DIR}/${DNN_BASENAME}.srsnv_dnn_metadata.json" \
  --output "${OUTPUT_DIR}/xgb_vs_dnn_chrom_val_comparison.html"

echo "Done. Report: ${OUTPUT_DIR}/xgb_vs_dnn_chrom_val_comparison.html"
