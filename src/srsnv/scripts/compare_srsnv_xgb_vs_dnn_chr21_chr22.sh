#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORKDIR="/data/Runs/perchik/ppmseq_data/srsnv_training_workspace"
INPUTS_DIR="${WORKDIR}/inputs"
OUTPUT_DIR="${WORKDIR}/output"

SPLIT_MANIFEST="${OUTPUT_DIR}/chr21_chr22_split_manifest.json"
XGB_PARAMS="${SCRIPT_DIR}/xgb_tuned_params.json"

POS_PARQUET="${INPUTS_DIR}/positive.parquet"
NEG_PARQUET="${INPUTS_DIR}/negative.parquet"
POS_BAM="${INPUTS_DIR}/positive_reads.bam"
NEG_BAM="${INPUTS_DIR}/negative_reads.bam"
TRAINING_REGIONS="${INPUTS_DIR}/training_regions.interval_list.gz"
STATS_POS="${INPUTS_DIR}/stats_positive.json"
STATS_NEG="${INPUTS_DIR}/stats_negative.json"
STATS_RAW="${INPUTS_DIR}/stats_featuremap.json"

BATCH_SIZE="${BATCH_SIZE:-1024}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-3}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"

echo "[1/4] Train XGBoost (tuned params) with shared split manifest..."
echo "  Using params: ${XGB_PARAMS}"
uv run srsnv_training \
  --positive "${POS_PARQUET}" \
  --negative "${NEG_PARQUET}" \
  --training-regions "${TRAINING_REGIONS}" \
  --stats-positive "${STATS_POS}" \
  --stats-negative "${STATS_NEG}" \
  --stats-featuremap "${STATS_RAW}" \
  --mean-coverage 112 \
  --single-model-split \
  --val-fraction 0.1 \
  --split-hash-key RN \
  --split-manifest-out "${SPLIT_MANIFEST}" \
  --holdout-chromosomes "chr21,chr22" \
  --model-params "${XGB_PARAMS}" \
  --features 'REF:ALT:X_HMER_REF:X_HMER_ALT:X_PREV1:X_NEXT1:X_PREV2:X_NEXT2:X_PREV3:X_NEXT3:BCSQ:BCSQCSS:RL:INDEX:DUP:REV:SCST:SCED:MAPQ:EDIST:SMQ_BEFORE:SMQ_AFTER:tm:rq:st:et' \
  --basename xgb_shared_split \
  --output "${OUTPUT_DIR}"

echo "[2/4] Train deep_srsnv CNN (Lightning) with same split manifest..."
uv run srsnv_dnn_bam_training \
  --positive-bam "${POS_BAM}" \
  --negative-bam "${NEG_BAM}" \
  --positive-parquet "${POS_PARQUET}" \
  --negative-parquet "${NEG_PARQUET}" \
  --training-regions "${TRAINING_REGIONS}" \
  --stats-positive "${STATS_POS}" \
  --stats-negative "${STATS_NEG}" \
  --stats-featuremap "${STATS_RAW}" \
  --mean-coverage 112 \
  --split-manifest-in "${SPLIT_MANIFEST}" \
  --single-model-split \
  --val-fraction 0.1 \
  --split-hash-key RN \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --lr-scheduler "${LR_SCHEDULER}" \
  --use-tf32 \
  --basename dnn_shared_split \
  --output "${OUTPUT_DIR}" \
  --verbose

echo "[3/4] Compare metrics on val + holdout (test) sets..."
uv run python - <<'PY'
import json
import polars as pl
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score

out = Path("/data/Runs/perchik/ppmseq_data/srsnv_training_workspace/output")
xgb_df = pl.read_parquet(out / "xgb_shared_split.featuremap_df.parquet")
dnn_df = pl.read_parquet(out / "dnn_shared_split.featuremap_df.parquet")

splits = {
    "val": {
        "xgb_mask": xgb_df["fold_id"] == 1,
        "dnn_mask": dnn_df["fold_id"] == 1,
    },
    "test (holdout)": {
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

xgb_meta = json.loads((out / "xgb_shared_split.srsnv_metadata.json").read_text())
dnn_meta = json.loads((out / "dnn_shared_split.srsnv_dnn_metadata.json").read_text())

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
  --xgb-parquet "${OUTPUT_DIR}/xgb_shared_split.featuremap_df.parquet" \
  --dnn-parquet "${OUTPUT_DIR}/dnn_shared_split.featuremap_df.parquet" \
  --xgb-metadata "${OUTPUT_DIR}/xgb_shared_split.srsnv_metadata.json" \
  --dnn-metadata "${OUTPUT_DIR}/dnn_shared_split.srsnv_dnn_metadata.json" \
  --output "${OUTPUT_DIR}/xgb_vs_dnn_lightning_comparison.html"

echo "Done. Report: ${OUTPUT_DIR}/xgb_vs_dnn_lightning_comparison.html"
