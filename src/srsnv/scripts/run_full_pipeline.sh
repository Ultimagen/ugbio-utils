#!/usr/bin/env bash
set -euo pipefail
#
# Full SRSNV pipeline: setup workspace, extract BAMs, train XGB + DNN, compare.
#
# Usage:
#   bash run_full_pipeline.sh /data/Runs/perchik/ppmseq_data/nanoseq_cord_blood
#
#   # Skip BAM extraction if already done:
#   SKIP_BAM_EXTRACT=1 bash run_full_pipeline.sh /data/Runs/perchik/ppmseq_data/nanoseq_cord_blood
#
#   # Skip everything up to comparison (reuse existing training outputs):
#   SKIP_BAM_EXTRACT=1 SKIP_TRAINING=1 bash run_full_pipeline.sh /data/Runs/perchik/ppmseq_data/nanoseq_cord_blood
#
# Environment variables:
#   WORKDIR           - workspace dir (default: /data/Runs/perchik/ppmseq_data/srsnv_training_workspace)
#   SKIP_BAM_EXTRACT  - set to 1 to skip BAM extraction step
#   SKIP_TRAINING     - set to 1 to skip training steps (XGB + DNN), jump to comparison
#   BATCH_SIZE, EPOCHS, PATIENCE, LR_SCHEDULER - DNN training hyperparameters
#

DATASET_DIR="${1:?Usage: $0 <dataset_dir>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

WORKDIR="${WORKDIR:-/data/Runs/perchik/ppmseq_data/srsnv_training_workspace}"
WORKSPACE_RUN_DIR="${WORKDIR}/run"

SKIP_BAM_EXTRACT="${SKIP_BAM_EXTRACT:-0}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"

echo "============================================"
echo "  SRSNV Full Pipeline"
echo "  Dataset: ${DATASET_DIR}"
echo "  Workspace: ${WORKDIR}"
echo "============================================"
echo ""

# ------------------------------------------------------------------
# Step 1: Setup workspace (symlinks + config)
# ------------------------------------------------------------------
echo ">>> Step 1/4: Setting up workspace for dataset..."
(
  cd "${REPO_ROOT}"
  uv run python "${SCRIPT_DIR}/setup_dataset.py" "${DATASET_DIR}" --workspace "${WORKDIR}"
)
echo ""

# Read config values for downstream steps
CONFIG_FILE="${WORKDIR}/inputs/dataset_config.json"
MEAN_COVERAGE="$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['mean_coverage'])")"
CONFIG_BASENAME="$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['basename'])")"

echo "  Basename:     ${CONFIG_BASENAME}"
echo "  Mean coverage: ${MEAN_COVERAGE}"
echo ""

# ------------------------------------------------------------------
# Step 2: Extract pos/neg BAMs from CRAM
# ------------------------------------------------------------------
if [[ "${SKIP_BAM_EXTRACT}" == "1" ]]; then
  echo ">>> Step 2/4: Skipping BAM extraction (SKIP_BAM_EXTRACT=1)"
else
  echo ">>> Step 2/4: Extracting positive/negative BAMs from CRAM..."
  echo "  This may take several hours for large CRAMs."
  (
    cd "${REPO_ROOT}"
    uv run python "${WORKSPACE_RUN_DIR}/extract_pos_neg_bams_from_cram.py" --workspace "${WORKDIR}"
  )
fi
echo ""

# ------------------------------------------------------------------
# Step 3 & 4: Train XGB + DNN + Compare (via comparison script)
# ------------------------------------------------------------------
if [[ "${SKIP_TRAINING}" == "1" ]]; then
  echo ">>> Step 3/4: Skipping training (SKIP_TRAINING=1)"
  echo ">>> Step 4/4: Skipping comparison (SKIP_TRAINING=1)"
else
  echo ">>> Step 3/4: Training XGBoost + DNN and comparing..."
  export WORKDIR MEAN_COVERAGE
  (
    cd "${REPO_ROOT}"
    bash "${SCRIPT_DIR}/compare_srsnv_xgb_vs_dnn_chr21_chr22.sh"
  )
fi

echo ""
echo "============================================"
echo "  Pipeline complete for: ${CONFIG_BASENAME}"
echo "  Outputs in: ${WORKDIR}/output/"
echo "============================================"
