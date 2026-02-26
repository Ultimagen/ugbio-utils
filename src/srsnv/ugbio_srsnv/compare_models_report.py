"""Generate an HTML report comparing XGBoost and DNN SRSNV models.

Reads prediction parquet files and metadata JSONs produced by both models
(trained on the same data splits) and generates a self-contained HTML report
with overlaid metrics, curves, and feature-stratified error analysis.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Environment, FileSystemLoader
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from ugbio_core.logger import logger

matplotlib.use("Agg")

REPORT_TEMPLATE = "compare_models_report.html.j2"
TEMPLATE_DIR = Path(__file__).parent / "reports"

COLOR_XGB = "#1f77b4"
COLOR_DNN = "#ff7f0e"
COLOR_BETTER = "#2ca02c"
COLOR_WORSE = "#d62728"

MIN_UNIQUE_LABELS = 2
MIN_SAMPLE_SIZE = 10
MIN_SLICE_SIZE = 50
TRINUC_LEN = 3
MAX_TICK_LABELS_HORIZONTAL = 6
SNVQ_THRESHOLDS = [0, 30, 40, 50, 60, 70]
_LUT_PAIR_LEN = 2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(
    xgb_parquet: Path,
    dnn_parquet: Path,
    xgb_metadata: Path,
    dnn_metadata: Path,
) -> dict:
    """Load parquet predictions and JSON metadata for both models."""
    logger.info("Loading XGBoost parquet: %s", xgb_parquet)
    xgb_df = pd.read_parquet(xgb_parquet)
    logger.info("Loading DNN parquet: %s", dnn_parquet)
    dnn_df = pd.read_parquet(dnn_parquet)

    logger.info("Loading metadata JSONs")
    with open(xgb_metadata) as f:
        xgb_meta = json.load(f)
    with open(dnn_metadata) as f:
        dnn_meta = json.load(f)

    merged = xgb_df.merge(
        dnn_df[["CHROM", "POS", "RN", "prob_orig", "SNVQ", "MQUAL"]].rename(
            columns={
                "prob_orig": "prob_dnn",
                "SNVQ": "snvq_dnn",
                "MQUAL": "mqual_dnn",
            }
        ),
        on=["CHROM", "POS", "RN"],
    )
    merged = merged.rename(
        columns={
            "prob_orig": "prob_xgb",
            "SNVQ": "snvq_xgb",
            "MQUAL": "mqual_xgb",
        }
    )
    logger.info("Merged dataframe shape: %s", merged.shape)

    return {
        "xgb_df": xgb_df,
        "dnn_df": dnn_df,
        "merged": merged,
        "xgb_meta": xgb_meta,
        "dnn_meta": dnn_meta,
    }


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < MIN_UNIQUE_LABELS or len(y) < MIN_SAMPLE_SIZE:
        return None
    return float(roc_auc_score(y, p))


def _safe_aupr(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < MIN_UNIQUE_LABELS or len(y) < MIN_SAMPLE_SIZE:
        return None
    return float(average_precision_score(y, p))


def _safe_logloss(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < MIN_UNIQUE_LABELS or len(y) < MIN_SAMPLE_SIZE:
        return None
    return float(log_loss(y, p, labels=[0, 1]))


def _safe_brier(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < MIN_UNIQUE_LABELS or len(y) < MIN_SAMPLE_SIZE:
        return None
    return float(brier_score_loss(y, p))


def _calc_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "auc": _safe_auc(y, p),
        "aupr": _safe_aupr(y, p),
        "logloss": _safe_logloss(y, p),
        "brier": _safe_brier(y, p),
    }


def _fmt(v, digits=4):
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def _relative_improvement(xgb_val: float | None, dnn_val: float | None, metric_name: str) -> float | None:
    """Compute relative improvement as % of gap closed.

    For higher-is-better metrics (AUC, AUPR): (dnn - xgb) / (1 - xgb) * 100
    For lower-is-better metrics (logloss, brier): (xgb - dnn) / xgb * 100
    """
    if xgb_val is None or dnn_val is None:
        return None
    if metric_name in ("logloss", "brier"):
        if xgb_val == 0:
            return None
        return (xgb_val - dnn_val) / xgb_val * 100
    gap = 1.0 - xgb_val
    if gap == 0:
        return None
    return (dnn_val - xgb_val) / gap * 100


def _improvement_class(rel_imp: float | None) -> str:
    """Return CSS class for relative improvement value (positive = better)."""
    if rel_imp is None:
        return ""
    return "better" if rel_imp > 0 else "worse"


def _delta_class(delta, metric_name):
    """Return CSS class for delta value."""
    if delta is None:
        return ""
    if metric_name in ("logloss", "brier"):
        return "better" if delta < 0 else "worse"
    return "better" if delta > 0 else "worse"


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _fig_to_b64(fig: plt.Figure, dpi: int = 150) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _load_dnn_epoch_metrics(dnn_meta: dict) -> pd.DataFrame | None:
    """Load per-epoch training metrics from Lightning CSVLogger output.

    Infers the ``metrics.csv`` path from the first checkpoint path in metadata.
    Lightning logs train and val metrics on separate rows, so we group by epoch
    and take the first non-NaN value for each metric column.
    """
    ckpt_paths = dnn_meta.get("best_checkpoint_paths", [])
    if not ckpt_paths:
        logger.warning("No best_checkpoint_paths in DNN metadata — cannot locate CSVLogger output")
        return None

    ckpt_path = Path(ckpt_paths[0])
    logs_dir = ckpt_path.parent / (ckpt_path.name.split(".dnn_model_fold_")[0] + ".lightning_logs")
    if not logs_dir.is_dir():
        parent = ckpt_path.parent
        candidates = list(parent.glob("*.lightning_logs"))
        if candidates:
            logs_dir = candidates[0]
        else:
            logger.warning("Cannot find lightning_logs directory near %s", ckpt_path)
            return None

    csv_files = sorted(logs_dir.glob("fold_*/metrics.csv"))
    if not csv_files:
        logger.warning("No metrics.csv found in %s", logs_dir)
        return None

    frames = []
    for csv_file in csv_files:
        try:
            metrics_df = pd.read_csv(csv_file)
            if "epoch" not in metrics_df.columns:
                continue
            per_epoch = metrics_df.groupby("epoch").first().reset_index()
            frames.append(per_epoch)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to read %s", csv_file)
            continue

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True).groupby("epoch").mean(numeric_only=True).reset_index()
    logger.info("Loaded DNN epoch metrics: %d epochs from %d fold(s)", len(combined), len(frames))
    return combined


# ---------------------------------------------------------------------------
# Compute all report data
# ---------------------------------------------------------------------------


def compute_report_data(data: dict) -> dict:  # noqa: C901, PLR0912, PLR0915
    """Compute all metrics, tables, and plots for the report."""
    merged = data["merged"]
    xgb_df = data["xgb_df"]
    dnn_df = data["dnn_df"]
    xgb_meta = data["xgb_meta"]
    dnn_meta = data["dnn_meta"]

    m_test = merged[merged["fold_id"].isna()]
    m_val = merged[merged["fold_id"] == 1]

    y_test = m_test["label"].astype(int).to_numpy()
    y_val = m_val["label"].astype(int).to_numpy()

    p_xgb_test = m_test["prob_xgb"].to_numpy()
    p_dnn_test = m_test["prob_dnn"].to_numpy()
    p_xgb_val = m_val["prob_xgb"].to_numpy()
    p_dnn_val = m_val["prob_dnn"].to_numpy()

    snvq_xgb_test = m_test["snvq_xgb"].to_numpy()
    snvq_dnn_test = m_test["snvq_dnn"].to_numpy()
    mqual_xgb_test = m_test["mqual_xgb"].to_numpy()
    mqual_dnn_test = m_test["mqual_dnn"].to_numpy()

    report = {"plots": {}}

    # ---- 1. Executive summary ----
    logger.info("Computing executive summary metrics")
    summary_rows = []
    for split_name, y, p_xgb, p_dnn in [
        ("Validation", y_val, p_xgb_val, p_dnn_val),
        ("Test (holdout)", y_test, p_xgb_test, p_dnn_test),
    ]:
        xm = _calc_metrics(y, p_xgb)
        dm = _calc_metrics(y, p_dnn)
        for metric in ["auc", "aupr", "logloss", "brier"]:
            rel_imp = _relative_improvement(xm[metric], dm[metric], metric)
            summary_rows.append(
                {
                    "split": split_name,
                    "metric": metric.upper(),
                    "xgb": _fmt(xm[metric]),
                    "dnn": _fmt(dm[metric]),
                    "rel_imp": _fmt(rel_imp, 1) if rel_imp is not None else "N/A",
                    "rel_imp_class": _improvement_class(rel_imp),
                }
            )
    report["summary"] = summary_rows

    # ---- 2. Dataset overview ----
    logger.info("Computing dataset overview")
    split_info = dnn_meta.get("split_prevalence", {})
    dataset_rows = []
    for role in ["train", "val", "test"]:
        info = split_info.get(role, {})
        dataset_rows.append(
            {
                "split": role.capitalize(),
                "rows": f"{info.get('rows', 'N/A'):,}" if isinstance(info.get("rows"), int) else "N/A",
                "positives": f"{info.get('positives', 'N/A'):,}" if isinstance(info.get("positives"), int) else "N/A",
                "negatives": f"{info.get('negatives', 'N/A'):,}" if isinstance(info.get("negatives"), int) else "N/A",
                "prevalence": _fmt(info.get("prevalence"), 4),
            }
        )
    report["dataset"] = dataset_rows

    manifest = dnn_meta.get("split_manifest", {})
    report["split_config"] = {
        "holdout_chromosomes": ", ".join(manifest.get("holdout_chromosomes", [])),
        "val_fraction": manifest.get("val_fraction"),
        "hash_key": manifest.get("hash_key"),
        "split_mode": manifest.get("split_mode"),
        "xgb_total": len(xgb_df),
        "dnn_total": len(dnn_df),
        "merged_total": len(merged),
    }

    # ---- 3. ROC curves ----
    logger.info("Generating ROC curves")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, split_name, y, px, pd_ in [
        (axes[0], "Validation", y_val, p_xgb_val, p_dnn_val),
        (axes[1], "Test (holdout)", y_test, p_xgb_test, p_dnn_test),
    ]:
        fpr_x, tpr_x, _ = roc_curve(y, px)
        fpr_d, tpr_d, _ = roc_curve(y, pd_)
        auc_x = roc_auc_score(y, px)
        auc_d = roc_auc_score(y, pd_)
        ax.plot(fpr_x, tpr_x, color=COLOR_XGB, lw=2, label=f"XGBoost (AUC={auc_x:.4f})")
        ax.plot(fpr_d, tpr_d, color=COLOR_DNN, lw=2, label=f"DNN (AUC={auc_d:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {split_name}")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    report["plots"]["roc"] = _fig_to_b64(fig)

    # ---- 4. Precision-Recall curves ----
    logger.info("Generating PR curves")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, split_name, y, px, pd_ in [
        (axes[0], "Validation", y_val, p_xgb_val, p_dnn_val),
        (axes[1], "Test (holdout)", y_test, p_xgb_test, p_dnn_test),
    ]:
        prec_x, rec_x, _ = precision_recall_curve(y, px)
        prec_d, rec_d, _ = precision_recall_curve(y, pd_)
        aupr_x = average_precision_score(y, px)
        aupr_d = average_precision_score(y, pd_)
        ax.plot(rec_x, prec_x, color=COLOR_XGB, lw=2, label=f"XGBoost (AUPR={aupr_x:.4f})")
        ax.plot(rec_d, prec_d, color=COLOR_DNN, lw=2, label=f"DNN (AUPR={aupr_d:.4f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall — {split_name}")
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    report["plots"]["pr"] = _fig_to_b64(fig)

    # ---- 5. Score distribution (SNVQ / Phred-scaled) ----
    logger.info("Generating score distributions (SNVQ from parquet)")
    tp_mask_global = y_test == 1
    all_snvq = np.concatenate([snvq_xgb_test, snvq_dnn_test])
    all_snvq_finite = all_snvq[np.isfinite(all_snvq)]
    snvq_xlim = float(np.percentile(all_snvq_finite, 99.5)) if len(all_snvq_finite) > 0 else 100.0
    snvq_bins = np.linspace(0, snvq_xlim, 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    for snvq, model, ls in [(snvq_xgb_test, "XGB", "-"), (snvq_dnn_test, "DNN", "--")]:
        ax.hist(
            snvq[tp_mask_global],
            bins=snvq_bins,
            density=True,
            histtype="step",
            color=COLOR_BETTER,
            linestyle=ls,
            lw=1.5,
            label=f"{model} TP",
        )
        ax.hist(
            snvq[~tp_mask_global],
            bins=snvq_bins,
            density=True,
            histtype="step",
            color=COLOR_WORSE,
            linestyle=ls,
            lw=1.5,
            label=f"{model} FP",
        )
    ax.set_xlabel("SNVQ (Phred-scaled)")
    ax.set_ylabel("Density")
    ax.set_title("SNVQ Distribution — Test Set (solid=XGB, dashed=DNN)")
    ax.set_yscale("log")
    ax.set_xlim(0, snvq_xlim)
    ax.legend(fontsize=10)
    ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    report["plots"]["score_dist"] = _fig_to_b64(fig)

    # ---- 6. Calibration + SNVQ-MQUAL mapping ----
    logger.info("Generating calibration & SNVQ-MQUAL mapping")
    fig, (ax_cal, ax_lut, ax_hist) = plt.subplots(
        3,
        1,
        figsize=(9, 14),
        gridspec_kw={"height_ratios": [3, 2, 1]},
    )

    for model_name, probs, color in [("XGBoost", p_xgb_test, COLOR_XGB), ("DNN", p_dnn_test, COLOR_DNN)]:
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=20, strategy="uniform")
        ece = float(np.mean(np.abs(prob_true - prob_pred)))
        ax_cal.plot(prob_pred, prob_true, "o-", color=color, lw=2, label=f"{model_name} (ECE={ece:.4f})")
    ax_cal.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax_cal.set_xlabel("Mean Predicted Probability")
    ax_cal.set_ylabel("Fraction of Positives")
    ax_cal.set_title("Calibration Curve — Test Set")
    ax_cal.legend(fontsize=11)
    ax_cal.grid(visible=True, alpha=0.3)

    for meta, model, color, ls in [
        (xgb_meta, "XGB", COLOR_XGB, "-"),
        (dnn_meta, "DNN", COLOR_DNN, "--"),
    ]:
        lut = meta.get("quality_recalibration_table")
        if lut is not None and len(lut) == _LUT_PAIR_LEN:
            x_lut, y_lut = np.array(lut[0]), np.array(lut[1])
            ax_lut.plot(x_lut, y_lut, color=color, linestyle=ls, lw=2, label=f"{model}: MQUAL → SNVQ")
    ax_lut_twin = ax_lut.twinx()
    all_mq = np.concatenate([mqual_xgb_test, mqual_dnn_test])
    all_mq_fin = all_mq[np.isfinite(all_mq)]
    mq_max = float(np.percentile(all_mq_fin, 99.5)) if len(all_mq_fin) > 0 else 100.0
    lut_bins = np.linspace(0, mq_max, 80)
    for mqual_arr, model, ls in [(mqual_xgb_test, "XGB", "-"), (mqual_dnn_test, "DNN", "--")]:
        mq_fin = mqual_arr[np.isfinite(mqual_arr)]
        ax_lut_twin.hist(
            mq_fin,
            bins=lut_bins,
            density=True,
            histtype="step",
            color="gray",
            linestyle=ls,
            lw=0.8,
            alpha=0.5,
            label=f"{model} MQUAL dist",
        )
    ax_lut.set_xlabel("MQUAL")
    ax_lut.set_ylabel("SNVQ")
    ax_lut.set_xlim(0, mq_max)
    ax_lut.set_title("MQUAL → SNVQ Mapping (solid=XGB, dashed=DNN)")
    ax_lut.legend(loc="upper left", fontsize=9)
    ax_lut_twin.set_ylabel("Density")
    ax_lut_twin.legend(loc="upper right", fontsize=8)
    ax_lut.grid(visible=True, alpha=0.3)

    prob_bins = np.linspace(0, 1, 80)
    ax_hist.hist(
        p_xgb_test,
        bins=prob_bins,
        density=True,
        histtype="step",
        color=COLOR_XGB,
        linestyle="-",
        lw=1.5,
        label="XGBoost",
    )
    ax_hist.hist(
        p_dnn_test, bins=prob_bins, density=True, histtype="step", color=COLOR_DNN, linestyle="--", lw=1.5, label="DNN"
    )
    ax_hist.set_xlabel("Predicted Probability")
    ax_hist.set_ylabel("Density")
    ax_hist.set_yscale("log")
    ax_hist.legend(fontsize=9)
    ax_hist.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    report["plots"]["calibration"] = _fig_to_b64(fig)

    # ---- 7. Quality score distribution (MQUAL from parquet) ----
    logger.info("Computing quality score distributions")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    qual_rows = []
    for model_name, mqual_arr in [("XGBoost", mqual_xgb_test), ("DNN", mqual_dnn_test)]:
        for lbl_name, mask in [("TP", y_test == 1), ("FP", y_test == 0)]:
            vals = mqual_arr[mask]
            vals = vals[np.isfinite(vals)]
            row = {"model": model_name, "label": lbl_name}
            for p in percentiles:
                row[f"p{p}"] = _fmt(float(np.percentile(vals, p)), 1) if len(vals) > 0 else "N/A"
            qual_rows.append(row)
    report["quality_percentiles"] = qual_rows

    all_mqual = np.concatenate([mqual_xgb_test, mqual_dnn_test])
    all_mqual_finite = all_mqual[np.isfinite(all_mqual)]
    mqual_xlim = float(np.percentile(all_mqual_finite, 99.5)) if len(all_mqual_finite) > 0 else 100.0
    mqual_bins = np.linspace(0, mqual_xlim, 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    for mqual_arr, model, ls in [(mqual_xgb_test, "XGB", "-"), (mqual_dnn_test, "DNN", "--")]:
        fin = np.isfinite(mqual_arr)
        tp_q = mqual_arr[fin & (y_test == 1)]
        fp_q = mqual_arr[fin & (y_test == 0)]
        ax.hist(
            tp_q,
            bins=mqual_bins,
            density=True,
            histtype="step",
            color=COLOR_BETTER,
            linestyle=ls,
            lw=1.5,
            label=f"{model} TP",
        )
        ax.hist(
            fp_q,
            bins=mqual_bins,
            density=True,
            histtype="step",
            color=COLOR_WORSE,
            linestyle=ls,
            lw=1.5,
            label=f"{model} FP",
        )
    ax.set_xlabel("MQUAL (Phred-scaled)")
    ax.set_ylabel("Density")
    ax.set_title("Quality Score Distribution — Test Set (solid=XGB, dashed=DNN)")
    ax.set_yscale("log")
    ax.set_xlim(0, mqual_xlim)
    ax.legend(fontsize=10)
    ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    report["plots"]["quality_dist"] = _fig_to_b64(fig)

    # ---- 8. Training progress ----
    logger.info("Generating training progress curves")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    xgb_results = xgb_meta.get("training_results", [{}])
    if isinstance(xgb_results, list) and len(xgb_results) > 0:
        xgb_res = xgb_results[0]
        v0 = xgb_res.get("validation_0", {})
        v1 = xgb_res.get("validation_1", {})

        if "auc" in v0:
            rounds = np.arange(1, len(v0["auc"]) + 1)
            axes[0, 0].plot(rounds, v0["auc"], color=COLOR_XGB, alpha=0.4, lw=1, label="Train")
            axes[0, 0].plot(rounds, v1.get("auc", []), color=COLOR_XGB, lw=2, label="Val")
            axes[0, 0].set_xlabel("Boosting Round")
            axes[0, 0].set_ylabel("AUC")
            axes[0, 0].set_title("XGBoost — AUC")
            axes[0, 0].legend()
            axes[0, 0].grid(visible=True, alpha=0.3)

        if "logloss" in v0:
            axes[1, 0].plot(rounds, v0["logloss"], color=COLOR_XGB, alpha=0.4, lw=1, label="Train")
            axes[1, 0].plot(rounds, v1.get("logloss", []), color=COLOR_XGB, lw=2, label="Val")
            axes[1, 0].set_xlabel("Boosting Round")
            axes[1, 0].set_ylabel("Logloss")
            axes[1, 0].set_title("XGBoost — Logloss")
            axes[1, 0].legend()
            axes[1, 0].grid(visible=True, alpha=0.3)

    dnn_epoch_metrics = _load_dnn_epoch_metrics(dnn_meta)
    if dnn_epoch_metrics is not None and not dnn_epoch_metrics.empty:
        epochs = dnn_epoch_metrics["epoch"].to_numpy()
        if "val_auc" in dnn_epoch_metrics.columns:
            if "train_auc" in dnn_epoch_metrics.columns:
                axes[0, 1].plot(epochs, dnn_epoch_metrics["train_auc"], color=COLOR_DNN, alpha=0.4, lw=1, label="Train")
            axes[0, 1].plot(epochs, dnn_epoch_metrics["val_auc"], color=COLOR_DNN, lw=2, label="Val")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("AUC")
            axes[0, 1].set_title("DNN — AUC")
            axes[0, 1].legend()
            axes[0, 1].grid(visible=True, alpha=0.3)

        if "val_loss" in dnn_epoch_metrics.columns:
            if "train_loss" in dnn_epoch_metrics.columns:
                axes[1, 1].plot(
                    epochs, dnn_epoch_metrics["train_loss"], color=COLOR_DNN, alpha=0.4, lw=1, label="Train"
                )
            axes[1, 1].plot(epochs, dnn_epoch_metrics["val_loss"], color=COLOR_DNN, lw=2, label="Val")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].set_title("DNN — Loss")
            axes[1, 1].legend()
            axes[1, 1].grid(visible=True, alpha=0.3)

    # Share y-axis limits across XGBoost (col 0) and DNN (col 1)
    for row in range(2):
        all_ylims = []
        for col in range(2):
            yl = axes[row, col].get_ylim()
            if yl != (0.0, 1.0) or axes[row, col].has_data():
                all_ylims.append(yl)
        if all_ylims:
            ymin = min(y[0] for y in all_ylims)
            ymax = max(y[1] for y in all_ylims)
            margin = 0.02 * (ymax - ymin) if ymax > ymin else 0.01
            for col in range(2):
                axes[row, col].set_ylim(ymin - margin, ymax + margin)

    fig.suptitle("Training Progress", fontsize=14, y=1.01)
    fig.tight_layout()
    report["plots"]["training_progress"] = _fig_to_b64(fig)

    # ---- 9. Confusion matrices ----
    logger.info("Generating confusion matrices")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    threshold = 0.5
    for ax, model_name, probs in [
        (axes[0], "XGBoost", p_xgb_test),
        (axes[1], "DNN", p_dnn_test),
    ]:
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(y_test, preds)
        ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["FP pred", "TP pred"])
        ax.set_yticklabels(["Actual FP", "Actual TP"])
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=13,
                )
        f1 = f1_score(y_test, preds)
        ax.set_title(f"{model_name} (threshold={threshold}, F1={f1:.4f})")
    fig.suptitle("Confusion Matrix — Test Set", fontsize=13)
    fig.tight_layout()
    report["plots"]["confusion"] = _fig_to_b64(fig)

    # ---- 10. Per-chromosome performance ----
    logger.info("Computing per-chromosome performance")
    chroms_in_test = sorted(m_test["CHROM"].unique())
    chrom_rows = []
    for chrom in chroms_in_test:
        mask_c = m_test["CHROM"] == chrom
        yc = m_test.loc[mask_c, "label"].astype(int).to_numpy()
        px_c = m_test.loc[mask_c, "prob_xgb"].to_numpy()
        pd_c = m_test.loc[mask_c, "prob_dnn"].to_numpy()
        auc_x = _safe_auc(yc, px_c)
        auc_d = _safe_auc(yc, pd_c)
        aupr_x = _safe_aupr(yc, px_c)
        aupr_d = _safe_aupr(yc, pd_c)
        ri_auc = _relative_improvement(auc_x, auc_d, "auc")
        ri_aupr = _relative_improvement(aupr_x, aupr_d, "aupr")
        chrom_rows.append(
            {
                "chrom": chrom,
                "n": int(mask_c.sum()),
                "auc_xgb": _fmt(auc_x),
                "auc_dnn": _fmt(auc_d),
                "ri_auc": _fmt(ri_auc, 1) if ri_auc is not None else "N/A",
                "ri_auc_class": _improvement_class(ri_auc),
                "aupr_xgb": _fmt(aupr_x),
                "aupr_dnn": _fmt(aupr_d),
                "ri_aupr": _fmt(ri_aupr, 1) if ri_aupr is not None else "N/A",
                "ri_aupr_class": _improvement_class(ri_aupr),
            }
        )
    report["per_chrom"] = chrom_rows

    if len(chroms_in_test) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        x_pos = np.arange(len(chroms_in_test))
        w = 0.35
        for ax, metric in [(axes[0], "auc"), (axes[1], "aupr")]:
            xgb_vals = [float(r[f"{metric}_xgb"]) if r[f"{metric}_xgb"] != "N/A" else 0 for r in chrom_rows]
            dnn_vals = [float(r[f"{metric}_dnn"]) if r[f"{metric}_dnn"] != "N/A" else 0 for r in chrom_rows]
            ax.bar(x_pos - w / 2, xgb_vals, w, label="XGBoost", color=COLOR_XGB)
            ax.bar(x_pos + w / 2, dnn_vals, w, label="DNN", color=COLOR_DNN)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(chroms_in_test)
            ax.set_ylabel(metric.upper())
            ax.set_title(f"Per-Chromosome {metric.upper()} — Test Set")
            ax.legend()
            ax.grid(visible=True, alpha=0.3, axis="y")
        fig.tight_layout()
        report["plots"]["per_chrom"] = _fig_to_b64(fig)

    # ---- 11. Model architecture summary ----
    logger.info("Building model architecture summary")
    xgb_model_params = xgb_meta.get("model_params", {})
    xgb_features = xgb_meta.get("features", [])
    xgb_split = xgb_meta.get("split_summary", {})

    dnn_arch = dnn_meta.get("model_architecture", {})
    dnn_params = dnn_meta.get("training_parameters", {})
    dnn_results = dnn_meta.get("training_results", [{}])
    dnn_fold0 = dnn_results[0] if dnn_results else {}

    report["arch"] = {
        "xgb": {
            "n_estimators": xgb_model_params.get("n_estimators"),
            "early_stopping_rounds": xgb_model_params.get("early_stopping_rounds"),
            "eval_metric": ", ".join(xgb_model_params.get("eval_metric", [])),
            "n_features": len(xgb_features),
            "feature_names": ", ".join(f.get("name", str(f)) if isinstance(f, dict) else str(f) for f in xgb_features),
            "best_iteration": xgb_split.get("best_iteration"),
        },
        "dnn": {
            "class_name": dnn_arch.get("class_name"),
            "trainable_parameters": f"{dnn_arch.get('trainable_parameters', 0):,}",
            "epochs": dnn_params.get("epochs"),
            "patience": dnn_params.get("patience"),
            "batch_size": dnn_params.get("batch_size"),
            "learning_rate": dnn_params.get("learning_rate"),
            "best_epoch": dnn_fold0.get("best_epoch"),
            "stopped_early": dnn_fold0.get("stopped_early"),
            "use_amp": dnn_params.get("use_amp"),
        },
    }

    # ---- 12. SNVQ threshold recall ----
    logger.info("Computing SNVQ threshold recall")
    n_tp_test = int(y_test.sum())
    tp_mask_test = y_test == 1

    median_snvq_xgb = float(np.median(snvq_xgb_test[tp_mask_test]))
    median_snvq_dnn = float(np.median(snvq_dnn_test[tp_mask_test]))

    snvq_recall_rows = []
    for thr in SNVQ_THRESHOLDS:
        recall_xgb = float(np.sum((snvq_xgb_test >= thr) & tp_mask_test)) / n_tp_test if n_tp_test > 0 else 0
        recall_dnn = float(np.sum((snvq_dnn_test >= thr) & tp_mask_test)) / n_tp_test if n_tp_test > 0 else 0
        ri = _relative_improvement(recall_xgb, recall_dnn, "auc")
        n_xgb = int(np.sum((snvq_xgb_test >= thr) & tp_mask_test))
        n_dnn = int(np.sum((snvq_dnn_test >= thr) & tp_mask_test))
        snvq_recall_rows.append(
            {
                "threshold": thr,
                "recall_xgb": _fmt(recall_xgb),
                "recall_dnn": _fmt(recall_dnn),
                "n_xgb": f"{n_xgb:,}",
                "n_dnn": f"{n_dnn:,}",
                "ri": _fmt(ri, 1) if ri is not None else "N/A",
                "ri_class": _improvement_class(ri),
            }
        )
    report["snvq_recall"] = {
        "rows": snvq_recall_rows,
        "median_snvq_xgb": _fmt(median_snvq_xgb, 1),
        "median_snvq_dnn": _fmt(median_snvq_dnn, 1),
        "n_tp": f"{n_tp_test:,}",
    }

    # ---- 12b. SNVQ vs ppmSeq tags ----
    logger.info("Computing SNVQ vs ppmSeq tag heatmaps")
    report["snvq_tags"] = _compute_snvq_vs_ppmseq_tags(m_test, tp_mask_test, snvq_xgb_test, snvq_dnn_test)

    # Section 14 (SNVQ-MQUAL mapping) is now embedded in Section 6 above

    # ---- 15. Logit histogram ----
    logger.info("Generating logit histogram")
    report["plots"]["logit_hist"] = _compute_logit_histogram(m_test, y_test)

    # ---- 16. SNVQ histogram by mixed status ----
    logger.info("Generating SNVQ histogram by mixed status")
    report["plots"]["snvq_mixed_hist"] = _compute_snvq_mixed_histogram(m_test, y_test, snvq_xgb_test, snvq_dnn_test)

    # ---- 17. SNVQ by trinucleotide context ----
    logger.info("Generating SNVQ by trinucleotide context")
    report["plots"]["trinuc_snvq"] = _compute_trinuc_snvq_comparison(m_test, y_test, snvq_xgb_test, snvq_dnn_test)

    # ---- 13. Error analysis ----
    report["error_analysis"] = _compute_error_analysis(m_test, y_test)

    return report


# ---------------------------------------------------------------------------
# Error analysis (Section 13)
# ---------------------------------------------------------------------------


def _sliced_auc(df: pd.DataFrame, y: np.ndarray, col: str, values: list | None = None) -> list[dict]:
    """Compute AUC/AUPR for both models sliced by a categorical column."""
    if values is None:
        values = sorted(df[col].dropna().unique())
    rows = []
    for val in values:
        mask = df[col] == val
        n = int(mask.sum())
        if n < MIN_SLICE_SIZE:
            continue
        yi = y[mask.to_numpy()]
        if len(np.unique(yi)) < MIN_UNIQUE_LABELS:
            continue
        auc_x = _safe_auc(yi, df.loc[mask, "prob_xgb"].to_numpy())
        auc_d = _safe_auc(yi, df.loc[mask, "prob_dnn"].to_numpy())
        aupr_x = _safe_aupr(yi, df.loc[mask, "prob_xgb"].to_numpy())
        aupr_d = _safe_aupr(yi, df.loc[mask, "prob_dnn"].to_numpy())
        ri_auc = _relative_improvement(auc_x, auc_d, "auc")
        rows.append(
            {
                "value": str(val) if val != "" else "(empty)",
                "n": n,
                "auc_xgb": auc_x,
                "auc_dnn": auc_d,
                "aupr_xgb": aupr_x,
                "aupr_dnn": aupr_d,
                "ri_auc": ri_auc,
                "ri_class": _improvement_class(ri_auc),
            }
        )
    return rows


def _sliced_auc_numeric_bins(df: pd.DataFrame, y: np.ndarray, col: str, n_bins: int = 10) -> list[dict]:
    """Compute AUC for both models sliced by binned numeric column."""
    valid = df[col].notna()
    vals = df.loc[valid, col]
    try:
        bins = pd.qcut(vals, n_bins, duplicates="drop")
    except ValueError:
        bins = pd.cut(vals, n_bins, duplicates="drop")
    rows = []
    for interval in sorted(bins.unique()):
        mask = valid & (bins == interval)
        n = int(mask.sum())
        if n < MIN_SLICE_SIZE:
            continue
        yi = y[mask.to_numpy()]
        if len(np.unique(yi)) < MIN_UNIQUE_LABELS:
            continue
        auc_x = _safe_auc(yi, df.loc[mask, "prob_xgb"].to_numpy())
        auc_d = _safe_auc(yi, df.loc[mask, "prob_dnn"].to_numpy())
        ri_auc = _relative_improvement(auc_x, auc_d, "auc")
        rows.append(
            {
                "value": str(interval),
                "n": n,
                "auc_xgb": auc_x,
                "auc_dnn": auc_d,
                "ri_auc": ri_auc,
                "ri_class": _improvement_class(ri_auc),
            }
        )
    return rows


def _make_grouped_bar(rows: list[dict], title: str, xlabel: str = "") -> str:
    """Create a grouped bar chart from sliced AUC rows, return base64 PNG."""
    if not rows:
        return ""
    labels = [r["value"] for r in rows]
    xgb_vals = [r["auc_xgb"] if r["auc_xgb"] is not None else 0 for r in rows]
    dnn_vals = [r["auc_dnn"] if r["auc_dnn"] is not None else 0 for r in rows]

    x_pos = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 5))
    ax.bar(x_pos - w / 2, xgb_vals, w, label="XGBoost", color=COLOR_XGB)
    ax.bar(x_pos + w / 2, dnn_vals, w, label="DNN", color=COLOR_DNN)
    ax.set_xticks(x_pos)
    needs_rotation = len(labels) > MAX_TICK_LABELS_HORIZONTAL
    ax.set_xticklabels(labels, rotation=45 if needs_rotation else 0, ha="right" if needs_rotation else "center")
    ax.set_ylabel("AUC")
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.legend()
    ax.grid(visible=True, alpha=0.3, axis="y")

    all_vals = [v for v in xgb_vals + dnn_vals if v > 0]
    if all_vals:
        ymin = min(all_vals)
        margin = (1 - ymin) * 0.1
        ax.set_ylim(max(0, ymin - margin), 1.0)

    fig.tight_layout()
    return _fig_to_b64(fig)


def _make_trinuc_heatmap(test_df: pd.DataFrame, y: np.ndarray) -> tuple[str, list[dict]]:
    """Create a trinuc context heatmap of delta AUC and return (b64_png, rows)."""
    test_df = test_df.copy()
    test_df["trinuc"] = test_df["X_PREV1"].astype(str) + test_df["REF"].astype(str) + test_df["X_NEXT1"].astype(str)
    rows = _sliced_auc(test_df, y, "trinuc")
    if not rows:
        return "", []

    bases = ["A", "C", "G", "T"]
    trinucs = sorted({r["value"] for r in rows})
    prev_bases = sorted({t[0] for t in trinucs if len(t) == TRINUC_LEN})
    next_bases = sorted({t[2] for t in trinucs if len(t) == TRINUC_LEN})

    delta_map = {r["value"]: r["ri_auc"] for r in rows if r["ri_auc"] is not None}
    matrix = np.full((len(prev_bases) * len(bases), len(next_bases)), np.nan)
    y_labels = []
    for i, pb in enumerate(prev_bases):
        for j, rb in enumerate(bases):
            y_labels.append(f"{pb}{rb}")
            for k, nb in enumerate(next_bases):
                key = f"{pb}{rb}{nb}"
                if key in delta_map:
                    matrix[i * len(bases) + j, k] = delta_map[key]

    fig, ax = plt.subplots(figsize=(8, 12))
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix))) if not np.all(np.isnan(matrix)) else 0.01
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(next_bases)))
    ax.set_xticklabels(next_bases)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Next Base")
    ax.set_ylabel("Prev + Ref Base")
    ax.set_title("Relative Improvement % (DNN vs XGB) by Trinucleotide Context")
    plt.colorbar(im, ax=ax, label="% gap closed (green = DNN better)", shrink=0.6)
    fig.tight_layout()
    return _fig_to_b64(fig), rows


def _compute_error_agreement(df: pd.DataFrame, y: np.ndarray, threshold: float = 0.5) -> dict:
    """Classify each test read by correctness of each model at given threshold."""
    xgb_correct = (df["prob_xgb"].to_numpy() >= threshold).astype(int) == y
    dnn_correct = (df["prob_dnn"].to_numpy() >= threshold).astype(int) == y

    both_correct = int(np.sum(xgb_correct & dnn_correct))
    both_wrong = int(np.sum(~xgb_correct & ~dnn_correct))
    only_xgb = int(np.sum(xgb_correct & ~dnn_correct))
    only_dnn = int(np.sum(~xgb_correct & dnn_correct))
    total = len(y)

    categories = [
        {"name": "Both correct", "count": f"{both_correct:,}", "pct": f"{100 * both_correct / total:.2f}%"},
        {"name": "Both wrong", "count": f"{both_wrong:,}", "pct": f"{100 * both_wrong / total:.2f}%"},
        {"name": "Only XGBoost correct", "count": f"{only_xgb:,}", "pct": f"{100 * only_xgb / total:.2f}%"},
        {"name": "Only DNN correct", "count": f"{only_dnn:,}", "pct": f"{100 * only_dnn / total:.2f}%"},
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [c["name"] for c in categories]
    counts = [both_correct, both_wrong, only_xgb, only_dnn]
    colors = [COLOR_BETTER, COLOR_WORSE, COLOR_XGB, COLOR_DNN]
    bars = ax.barh(names, counts, color=colors)
    for bar, cnt in zip(bars, counts, strict=True):
        ax.text(
            bar.get_width() + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{cnt:,} ({100 * cnt / total:.1f}%)",
            va="center",
            fontsize=10,
        )
    ax.set_xlabel("Number of Reads")
    ax.set_title(f"Error Agreement (threshold={threshold}) — Test Set")
    ax.grid(visible=True, alpha=0.3, axis="x")
    fig.tight_layout()

    return {
        "categories": categories,
        "plot": _fig_to_b64(fig),
    }


def _compute_snvq_vs_ppmseq_tags(  # noqa: PLR0915
    m_test: pd.DataFrame,
    tp_mask: np.ndarray,
    snvq_xgb: np.ndarray,
    snvq_dnn: np.ndarray,
) -> dict:
    """Compute median SNVQ per ppmSeq start/end tag combination for both models."""
    result: dict = {}
    df_tp = m_test.loc[tp_mask].copy()

    st_col = "st" if "st" in df_tp.columns else None
    et_col = "et" if "et" in df_tp.columns else None
    if st_col is None or et_col is None:
        logger.warning("ppmSeq tag columns (st, et) not found — skipping SNVQ vs tags section")
        return result

    df_tp["snvq_xgb"] = snvq_xgb[tp_mask]
    df_tp["snvq_dnn"] = snvq_dnn[tp_mask]
    df_tp[st_col] = df_tp[st_col].astype(str).replace({"": "(empty)", "nan": "(empty)"})
    df_tp[et_col] = df_tp[et_col].astype(str).replace({"": "(empty)", "nan": "(empty)"})

    xgb_pivot = df_tp.pivot_table(index=st_col, columns=et_col, values="snvq_xgb", aggfunc="median", dropna=False)
    dnn_pivot = df_tp.pivot_table(index=st_col, columns=et_col, values="snvq_dnn", aggfunc="median", dropna=False)
    count_pivot = df_tp.pivot_table(index=st_col, columns=et_col, values="snvq_xgb", aggfunc="count", dropna=False)

    preferred_order = ["MIXED", "MINUS", "PLUS", "(empty)"]
    all_st_raw = set(xgb_pivot.index) | set(dnn_pivot.index)
    all_et_raw = set(xgb_pivot.columns) | set(dnn_pivot.columns)
    all_st = [t for t in preferred_order if t in all_st_raw] + sorted(all_st_raw - set(preferred_order))
    all_et = [t for t in preferred_order if t in all_et_raw] + sorted(all_et_raw - set(preferred_order))
    xgb_pivot = xgb_pivot.reindex(index=all_st, columns=all_et)
    dnn_pivot = dnn_pivot.reindex(index=all_st, columns=all_et)
    count_pivot = count_pivot.reindex(index=all_st, columns=all_et).fillna(0)

    delta_pivot = dnn_pivot - xgb_pivot

    pct_pivot = (count_pivot / count_pivot.to_numpy().sum()) * 100

    def _annot_table(val_df, pct_df):
        """Build combined annotation: value [pct%]."""
        annot = val_df.copy().astype(object)
        for r in annot.index:
            for c in annot.columns:
                v = val_df.loc[r, c]
                p = pct_df.loc[r, c]
                if pd.isna(v):
                    annot.loc[r, c] = ""
                else:
                    annot.loc[r, c] = f"{v:.1f}\n[{p:.1f}%]"
        return annot

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    for ax, pivot, annot_piv, title, cmap in [
        (axes[0], xgb_pivot, _annot_table(xgb_pivot, pct_pivot), "XGBoost — Median SNVQ", "inferno"),
        (axes[1], dnn_pivot, _annot_table(dnn_pivot, pct_pivot), "DNN — Median SNVQ", "inferno"),
    ]:
        sns.heatmap(
            pivot,
            annot=annot_piv,
            fmt="",
            cmap=cmap,
            cbar=True,
            linewidths=1,
            linecolor="black",
            annot_kws={"size": 11},
            square=True,
            ax=ax,
        )
        ax.set_xlabel("ppmSeq End Tag (et)")
        ax.set_ylabel("ppmSeq Start Tag (st)")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    vmax = np.nanmax(np.abs(delta_pivot.to_numpy()))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1.0
    sns.heatmap(
        delta_pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        cbar=True,
        linewidths=1,
        linecolor="black",
        annot_kws={"size": 10},
        square=True,
        ax=axes[2],
    )
    axes[2].set_xlabel("ppmSeq End Tag (et)")
    axes[2].set_ylabel("ppmSeq Start Tag (st)")
    axes[2].set_title("Delta Median SNVQ (DNN - XGBoost)")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].tick_params(axis="y", rotation=0)

    fig.suptitle("SNVQ vs ppmSeq Start/End Tags (TP reads only, Test Set)", fontsize=14, y=1.02)
    fig.tight_layout()
    result["plot"] = _fig_to_b64(fig)

    return result


def _compute_logit_histogram(m_test: pd.DataFrame, y_test: np.ndarray) -> str:
    """Plot logit histograms split by FP / TP mixed / TP non-mixed for both models."""
    from ugbio_srsnv.srsnv_utils import prob_to_logit  # noqa: PLC0415

    tp_mask = y_test == 1
    st_vals = m_test["st"].astype(str).to_numpy() if "st" in m_test.columns else np.full(len(y_test), "")
    et_vals = m_test["et"].astype(str).to_numpy() if "et" in m_test.columns else np.full(len(y_test), "")
    is_mixed = (st_vals == "MIXED") & (et_vals == "MIXED")

    fp_mask = ~tp_mask
    tp_mixed_mask = tp_mask & is_mixed
    tp_nonmixed_mask = tp_mask & ~is_mixed

    logits_xgb = prob_to_logit(m_test["prob_xgb"].to_numpy(), phred=True)
    logits_dnn = prob_to_logit(m_test["prob_dnn"].to_numpy(), phred=True)

    all_finite = np.concatenate([logits_xgb, logits_dnn])
    all_finite = all_finite[np.isfinite(all_finite)]
    xlim_lo = float(np.percentile(all_finite, 0.5)) if len(all_finite) > 0 else -10
    xlim_hi = float(np.percentile(all_finite, 99.5)) if len(all_finite) > 0 else 50
    shared_bins = np.linspace(xlim_lo, xlim_hi, 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    categories = [
        (fp_mask, "FP", "tab:blue"),
        (tp_mixed_mask, "TP mixed", "green"),
        (tp_nonmixed_mask, "TP non-mixed", "red"),
    ]
    for logits, model, ls in [(logits_xgb, "XGB", "-"), (logits_dnn, "DNN", "--")]:
        for mask, cat_label, clr in categories:
            vals = logits[mask]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                ax.hist(
                    vals,
                    bins=shared_bins,
                    density=True,
                    histtype="step",
                    color=clr,
                    linestyle=ls,
                    lw=1.5,
                    label=f"{model}: {cat_label}",
                )

    ax.set_xlabel("ML Logit (Phred scale)")
    ax.set_ylabel("Density")
    ax.set_title("Logit Distribution — Test Set (solid=XGB, dashed=DNN)")
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _compute_snvq_mixed_histogram(
    m_test: pd.DataFrame,
    y_test: np.ndarray,
    snvq_xgb: np.ndarray,
    snvq_dnn: np.ndarray,
) -> str:
    """Plot SNVQ distribution for TP reads split by mixed status for both models."""
    tp_mask = y_test == 1
    st_vals = m_test["st"].astype(str).to_numpy() if "st" in m_test.columns else np.full(len(y_test), "")
    et_vals = m_test["et"].astype(str).to_numpy() if "et" in m_test.columns else np.full(len(y_test), "")

    is_mixed_start = st_vals == "MIXED"
    is_mixed_end = et_vals == "MIXED"

    nonmixed = tp_mask & ~is_mixed_start & ~is_mixed_end
    one_end = tp_mask & (is_mixed_start ^ is_mixed_end)
    both_ends = tp_mask & is_mixed_start & is_mixed_end

    all_snvq_tp = np.concatenate([snvq_xgb[tp_mask], snvq_dnn[tp_mask]])
    all_finite = all_snvq_tp[np.isfinite(all_snvq_tp)]
    xlim_max = float(np.percentile(all_finite, 99.5)) if len(all_finite) > 0 else 100.0
    shared_bins = np.linspace(0, xlim_max, 100)

    categories = [
        (nonmixed, "non-mixed", "red"),
        (one_end, "mixed, one end", "blue"),
        (both_ends, "mixed, both ends", "green"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for snvq, model, ls in [(snvq_xgb, "XGB", "-"), (snvq_dnn, "DNN", "--")]:
        for mask, cat_label, clr in categories:
            vals = snvq[mask]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                ax.hist(
                    vals,
                    bins=shared_bins,
                    density=True,
                    histtype="step",
                    color=clr,
                    linestyle=ls,
                    lw=1.5,
                    label=f"{model}: {cat_label}",
                )

    ax.set_xlabel("SNVQ")
    ax.set_ylabel("Density")
    ax.set_title("SNVQ by Mixed Status — TP reads, Test Set (solid=XGB, dashed=DNN)")
    ax.set_xlim(0, xlim_max)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _compute_trinuc_snvq_comparison(  # noqa: C901, PLR0912, PLR0915
    m_test: pd.DataFrame,
    y_test: np.ndarray,
    snvq_xgb: np.ndarray,
    snvq_dnn: np.ndarray,
    q1: float = 0.1,
    q2: float = 0.9,
) -> str:
    """Plot median SNVQ by trinucleotide context for XGBoost vs DNN.

    Two-row layout matching the original monitor report: each row has a SNVQ
    panel (median + q1-q2 shading) on top and a density histogram (TP/FP) below.
    Row 1 = forward trinucs (A>C .. G>T), Row 2 = complement (C>A .. T>G).
    """
    from matplotlib import gridspec  # noqa: PLC0415

    from ugbio_srsnv.srsnv_utils import FLOW_ORDER, is_cycle_skip  # noqa: PLC0415

    tp_mask = y_test == 1
    if tp_mask.sum() == 0:
        logger.warning("No TP reads for trinuc SNVQ comparison")
        return ""

    all_df = m_test.copy()
    all_df["tcwa"] = (
        all_df["X_PREV1"].astype(str)
        + all_df["REF"].astype(str)
        + all_df["X_NEXT1"].astype(str)
        + all_df["ALT"].astype(str)
    )
    all_df["snvq_xgb"] = snvq_xgb
    all_df["snvq_dnn"] = snvq_dnn
    all_df["label"] = y_test

    trinuc_ref_alt_fwd = [
        c1 + r + c2 + a
        for r, a in (("A", "C"), ("A", "G"), ("A", "T"), ("C", "G"), ("C", "T"), ("G", "T"))
        for c1 in ("A", "C", "G", "T")
        for c2 in ("A", "C", "G", "T")
        if r != a
    ]
    trinuc_ref_alt_rev = [
        c1 + r + c2 + a
        for a, r in (("A", "C"), ("A", "G"), ("A", "T"), ("C", "G"), ("C", "T"), ("G", "T"))
        for c1 in ("A", "C", "G", "T")
        for c2 in ("A", "C", "G", "T")
        if r != a
    ]
    trinuc_ref_alt = trinuc_ref_alt_fwd + trinuc_ref_alt_rev
    trinuc_index = np.array([f"{t[0]}[{t[1]}>{t[3]}]{t[2]}" for t in trinuc_ref_alt])
    trinuc_is_cycle_skip = np.array([is_cycle_skip(tcwa, flow_order=FLOW_ORDER) for tcwa in trinuc_ref_alt])
    snv_labels = [" ".join(trinuc_index[i][2:5]) for i in range(0, 16 * 12, 16)]

    tcwa_to_idx = {tcwa: i for i, tcwa in enumerate(trinuc_ref_alt)}
    n_total = 192
    n_panel = 96

    all_df["tcwa_idx"] = all_df["tcwa"].map(tcwa_to_idx)
    all_df = all_df.dropna(subset=["tcwa_idx"])
    all_df["tcwa_idx"] = all_df["tcwa_idx"].astype(int)

    tp_df = all_df[all_df["label"] == 1]

    # -- per-context SNVQ stats (TP only) --
    med_xgb = np.full(n_total, np.nan)
    med_dnn = np.full(n_total, np.nan)
    lo_xgb = np.full(n_total, np.nan)
    hi_xgb = np.full(n_total, np.nan)
    lo_dnn = np.full(n_total, np.nan)
    hi_dnn = np.full(n_total, np.nan)

    for idx, grp in tp_df.groupby("tcwa_idx"):
        med_xgb[idx] = grp["snvq_xgb"].median()
        med_dnn[idx] = grp["snvq_dnn"].median()
        lo_xgb[idx] = grp["snvq_xgb"].quantile(q1)
        hi_xgb[idx] = grp["snvq_xgb"].quantile(q2)
        lo_dnn[idx] = grp["snvq_dnn"].quantile(q1)
        hi_dnn[idx] = grp["snvq_dnn"].quantile(q2)

    # -- per-context histogram (TP + FP) --
    n_tp = int(tp_mask.sum())
    n_fp = int((~tp_mask).sum())
    hist_tp = np.zeros(n_total)
    hist_fp = np.zeros(n_total)
    for idx, grp in all_df.groupby("tcwa_idx"):
        label_counts = grp["label"].value_counts()
        hist_tp[idx] = label_counts.get(1, 0)
        hist_fp[idx] = label_counts.get(0, 0)
    if hist_tp.sum() > 0:
        hist_tp = hist_tp / hist_tp.sum()
    if hist_fp.sum() > 0:
        hist_fp = hist_fp / hist_fp.sum()

    x_vals = list(range(n_panel))
    x_ext = [x_vals[0] - 0.5] + x_vals + [x_vals[-1] + 0.5]

    def _extend(arr):
        return np.concatenate([[arr[0]], arr, [arr[-1]]])

    # -- figure layout: 2 rows × (SNVQ + Histogram) --
    fig = plt.figure(figsize=(18, 14))
    hspace = 0.12
    height_ratios = [1, 2]
    gs_top = gridspec.GridSpec(2, 1, height_ratios=height_ratios, hspace=0.0, top=0.92, bottom=0.55 + hspace / 2)
    gs_bot = gridspec.GridSpec(2, 1, height_ratios=height_ratios, hspace=0.0, top=0.55 - hspace / 2, bottom=0.18)

    snv_positions = [8, 24, 40, 56, 72, 88]

    for panel_idx, gs in enumerate([gs_top, gs_bot]):
        sl = slice(panel_idx * n_panel, (panel_idx + 1) * n_panel)
        inds = np.arange(panel_idx * n_panel, (panel_idx + 1) * n_panel)

        qual_ax = fig.add_subplot(gs[0])
        hist_ax = fig.add_subplot(gs[1], sharex=qual_ax)

        # -- SNVQ panel --
        qual_ax.fill_between(x_ext, _extend(lo_xgb[sl]), _extend(hi_xgb[sl]), step="mid", color=COLOR_XGB, alpha=0.2)
        qual_ax.step(x_ext, _extend(med_xgb[sl]), where="mid", color=COLOR_XGB, lw=1.5, label="XGBoost")
        qual_ax.fill_between(x_ext, _extend(lo_dnn[sl]), _extend(hi_dnn[sl]), step="mid", color=COLOR_DNN, alpha=0.2)
        qual_ax.step(
            x_ext,
            _extend(med_dnn[sl]),
            where="mid",
            color=COLOR_DNN,
            lw=1.5,
            linestyle="--",
            label="DNN",
        )
        qual_ax.set_ylabel("SNVQ", fontsize=14)
        qual_ax.set_xlim(-0.5, n_panel - 0.5)
        qual_ax.grid(visible=True, axis="both", alpha=0.75, linestyle=":")

        ylim_min, ylim_max = qual_ax.get_ylim()
        for j in range(5):
            qual_ax.plot([(j + 1) * 16 - 0.5] * 2, [ylim_min, ylim_max], "k--", lw=0.8)
        for lbl, pos in zip(snv_labels[6 * panel_idx : 6 * panel_idx + 6], snv_positions, strict=False):
            qual_ax.annotate(
                lbl,
                xy=(pos, qual_ax.get_ylim()[1]),
                xytext=(-2, 6),
                textcoords="offset points",
                ha="center",
                fontsize=12,
                fontweight="bold",
            )
        qual_ax.set_xticklabels([])

        # -- Histogram panel --
        hist_ax.bar(x_vals, hist_tp[sl], width=1.0, alpha=0.5, color="tab:orange", label=f"TP ({n_tp})")
        hist_ax.bar(x_vals, hist_fp[sl], width=1.0, alpha=0.5, color="tab:blue", label=f"FP ({n_fp})")
        hist_ylim = 1.05 * max(hist_tp[sl].max(), hist_fp[sl].max()) if hist_tp[sl].max() > 0 else 1.0
        for j in range(5):
            hist_ax.plot([(j + 1) * 16 - 0.5] * 2, [0, hist_ylim], "k--", lw=0.8)
        hist_ax.set_ylim(0, hist_ylim)
        hist_ax.set_ylabel("Density", fontsize=14)
        hist_ax.set_xlim(-0.5, n_panel - 0.5)
        hist_ax.grid(visible=True, axis="both", alpha=0.75, linestyle=":")

        hist_ax.set_xticks(x_vals)
        hist_ax.set_xticklabels(trinuc_index[inds], rotation=90, fontsize=10)
        tick_colors = ["green" if trinuc_is_cycle_skip[idx] else "red" for idx in inds]
        for j in range(n_panel):
            hist_ax.get_xticklabels()[j].set_color(tick_colors[j])
        hist_ax.tick_params(axis="x", pad=-2)

    # -- Bottom legend area --
    fig.legend(
        [
            plt.Rectangle((0, 0), 1, 1, fc="tab:orange", alpha=0.5),
            plt.Rectangle((0, 0), 1, 1, fc="tab:blue", alpha=0.5),
        ],
        [f"TP ({n_tp})", f"FP ({n_fp})"],
        title="Histogram: Labels (# SNVs total)",
        fontsize=14,
        title_fontsize=14,
        loc="lower center",
        bbox_to_anchor=(0.24, 0.03),
        ncol=1,
        frameon=False,
    )
    fig.legend(
        [plt.Line2D([0], [0], color=COLOR_XGB, lw=2), plt.Line2D([0], [0], color=COLOR_DNN, lw=2, linestyle="--")],
        ["XGBoost", "DNN"],
        title=f"SNVQ on TP (median + {int(q1 * 100)}%\u2013{int(q2 * 100)}% range)",
        fontsize=14,
        title_fontsize=14,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=1,
        frameon=False,
    )
    fig.text(0.7, 0.07, "Trinuc-SNV:", ha="left", fontsize=14, color="black")
    fig.text(0.73, 0.04, "Green: Cycle skip", ha="left", fontsize=14, color="green")
    fig.text(0.73, 0.01, "Red: No cycle skip", ha="left", fontsize=14, color="red")

    fig.suptitle("Quality as function of trinuc context and alt", fontsize=20, y=0.96)

    return _fig_to_b64(fig)


def _compute_error_analysis(m_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    """Run all error analysis sub-sections."""
    ea = {}
    m_test = m_test.copy()

    # 13a. Substitution type
    logger.info("Error analysis: substitution type")
    m_test["sub_type"] = m_test["REF"].astype(str) + ">" + m_test["ALT"].astype(str)
    ea["sub_type"] = _sliced_auc(m_test, y_test, "sub_type")
    ea["sub_type_plot"] = _make_grouped_bar(ea["sub_type"], "AUC by Substitution Type", "Substitution")

    # 13b. Mixed status (st)
    logger.info("Error analysis: mixed status (st)")
    m_test["st_filled"] = m_test["st"].astype(str).replace("", "(empty)").replace("nan", "(empty)")
    ea["mixed_st"] = _sliced_auc(m_test, y_test, "st_filled")
    ea["mixed_st_plot"] = _make_grouped_bar(ea["mixed_st"], "AUC by ppmSeq Start Tag (st)", "Start Tag")

    # 13c. End tag (et)
    logger.info("Error analysis: end tag (et)")
    m_test["et_filled"] = m_test["et"].astype(str).replace("", "(empty)").replace("nan", "(empty)")
    ea["end_tag"] = _sliced_auc(m_test, y_test, "et_filled")
    ea["end_tag_plot"] = _make_grouped_bar(ea["end_tag"], "AUC by ppmSeq End Tag (et)", "End Tag")

    # 13d. Combined tag (tm)
    logger.info("Error analysis: combined tag (tm)")
    m_test["tm_filled"] = m_test["tm"].astype(str).replace("", "(empty)").replace("nan", "(empty)")
    ea["tm_tag"] = _sliced_auc(m_test, y_test, "tm_filled")
    ea["tm_tag_plot"] = _make_grouped_bar(ea["tm_tag"], "AUC by ppmSeq Combined Tag (tm)", "Tag")

    # 13e. Homopolymer length
    logger.info("Error analysis: homopolymer length")
    ea["hmer"] = _sliced_auc(m_test, y_test, "X_HMER_REF")
    ea["hmer_plot"] = _make_grouped_bar(ea["hmer"], "AUC by Homopolymer Length (X_HMER_REF)", "Homopolymer Length")

    # 13f. Edit distance
    logger.info("Error analysis: edit distance")
    ea["edist"] = _sliced_auc(m_test, y_test, "EDIST")
    ea["edist_plot"] = _make_grouped_bar(ea["edist"], "AUC by Edit Distance", "Edit Distance")

    # 13g. Read position (INDEX binned)
    logger.info("Error analysis: read position (INDEX)")
    ea["index_bin"] = _sliced_auc_numeric_bins(m_test, y_test, "INDEX", n_bins=10)
    ea["index_bin_plot"] = _make_grouped_bar(ea["index_bin"], "AUC by Read Position (INDEX)", "Position Bin")

    # 13h. Strand
    logger.info("Error analysis: strand (REV)")
    ea["strand"] = _sliced_auc(m_test, y_test, "REV")
    for r in ea["strand"]:
        r["value"] = "Reverse" if r["value"] == "1" else "Forward"
    ea["strand_plot"] = _make_grouped_bar(ea["strand"], "AUC by Strand", "Strand")

    # 13i. Read quality (rq binned)
    logger.info("Error analysis: read quality (rq)")
    ea["rq_bin"] = _sliced_auc_numeric_bins(m_test, y_test, "rq", n_bins=10)
    ea["rq_bin_plot"] = _make_grouped_bar(ea["rq_bin"], "AUC by Read Quality (rq)", "rq Bin")

    # 13j. Trinucleotide context
    logger.info("Error analysis: trinucleotide context")
    ea["trinuc_plot"], ea["trinuc"] = _make_trinuc_heatmap(m_test, y_test)

    # 13k. Error agreement
    logger.info("Error analysis: error agreement")
    ea["agreement"] = _compute_error_agreement(m_test, y_test)

    # Format numeric values for table display
    for key in [
        "sub_type",
        "mixed_st",
        "end_tag",
        "tm_tag",
        "hmer",
        "edist",
        "index_bin",
        "strand",
        "rq_bin",
        "trinuc",
    ]:
        if key in ea and isinstance(ea[key], list):
            for r in ea[key]:
                for f_key in ["auc_xgb", "auc_dnn", "aupr_xgb", "aupr_dnn"]:
                    if f_key in r and r[f_key] is not None and not isinstance(r[f_key], str):
                        r[f_key] = _fmt(r[f_key])
                    elif f_key in r and r[f_key] is None:
                        r[f_key] = "N/A"
                if "ri_auc" in r and r["ri_auc"] is not None and not isinstance(r["ri_auc"], str):
                    r["ri_auc"] = _fmt(r["ri_auc"], 1)
                elif "ri_auc" in r and r["ri_auc"] is None:
                    r["ri_auc"] = "N/A"

    return ea


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def render_html(report: dict, output_path: Path) -> None:
    """Render the HTML report from the Jinja2 template."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,  # noqa: S701 - HTML template with trusted inputs only
    )
    template = env.get_template(REPORT_TEMPLATE)
    html = template.render(**report)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate XGBoost vs DNN SRSNV comparison report")
    ap.add_argument("--xgb-parquet", type=Path, required=True, help="XGBoost featuremap_df parquet")
    ap.add_argument("--dnn-parquet", type=Path, required=True, help="DNN featuremap_df parquet")
    ap.add_argument("--xgb-metadata", type=Path, required=True, help="XGBoost srsnv_metadata JSON")
    ap.add_argument("--dnn-metadata", type=Path, required=True, help="DNN srsnv_dnn_metadata JSON")
    ap.add_argument("--output", type=Path, required=True, help="Output HTML path")
    return ap.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data = load_data(args.xgb_parquet, args.dnn_parquet, args.xgb_metadata, args.dnn_metadata)
    report = compute_report_data(data)
    render_html(report, args.output)


def main():
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
