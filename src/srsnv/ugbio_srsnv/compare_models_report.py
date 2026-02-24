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
from jinja2 import Environment, FileSystemLoader
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
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
        dnn_df[["CHROM", "POS", "RN", "prob_orig"]].rename(columns={"prob_orig": "prob_dnn"}),
        on=["CHROM", "POS", "RN"],
    )
    merged = merged.rename(columns={"prob_orig": "prob_xgb"})
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


def _calc_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "auc": _safe_auc(y, p),
        "aupr": _safe_aupr(y, p),
        "logloss": _safe_logloss(y, p),
    }


def _fmt(v, digits=4):
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def _delta_class(delta, metric_name):
    """Return CSS class for delta value."""
    if delta is None:
        return ""
    if metric_name == "logloss":
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
        for metric in ["auc", "aupr", "logloss"]:
            delta = (dm[metric] - xm[metric]) if (dm[metric] is not None and xm[metric] is not None) else None
            summary_rows.append(
                {
                    "split": split_name,
                    "metric": metric.upper(),
                    "xgb": _fmt(xm[metric]),
                    "dnn": _fmt(dm[metric]),
                    "delta": _fmt(delta) if delta is not None else "N/A",
                    "delta_class": _delta_class(delta, metric),
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

    # ---- 5. Score distribution ----
    logger.info("Generating score distributions")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, model_name, probs in [
        (axes[0], "XGBoost", p_xgb_test),
        (axes[1], "DNN", p_dnn_test),
    ]:
        tp_mask = y_test == 1
        ax.hist(probs[tp_mask], bins=100, alpha=0.6, label="TP", color=COLOR_BETTER, density=True)
        ax.hist(probs[~tp_mask], bins=100, alpha=0.6, label="FP", color=COLOR_WORSE, density=True)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.set_title(f"Score Distribution — {model_name} (Test)")
        ax.legend(fontsize=10)
        ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    report["plots"]["score_dist"] = _fig_to_b64(fig)

    # ---- 6. Calibration ----
    logger.info("Generating calibration curves")
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    for model_name, probs, color in [("XGBoost", p_xgb_test, COLOR_XGB), ("DNN", p_dnn_test, COLOR_DNN)]:
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=20, strategy="uniform")
        ece = float(np.mean(np.abs(prob_true - prob_pred)))
        ax.plot(prob_pred, prob_true, "o-", color=color, lw=2, label=f"{model_name} (ECE={ece:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve — Test Set")
    ax.legend(fontsize=11)
    ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    report["plots"]["calibration"] = _fig_to_b64(fig)

    # ---- 7. Quality score distribution ----
    logger.info("Computing quality score distributions")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    qual_rows = []
    for model_name, probs in [("XGBoost", p_xgb_test), ("DNN", p_dnn_test)]:
        mqual = -10 * np.log10(np.clip(1 - probs, 1e-10, 1))
        for lbl_name, mask in [("TP", y_test == 1), ("FP", y_test == 0)]:
            vals = mqual[mask]
            row = {"model": model_name, "label": lbl_name}
            for p in percentiles:
                row[f"p{p}"] = _fmt(float(np.percentile(vals, p)), 1)
            qual_rows.append(row)
    report["quality_percentiles"] = qual_rows

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for model_name, probs, color in [("XGBoost", p_xgb_test, COLOR_XGB), ("DNN", p_dnn_test, COLOR_DNN)]:
        mqual = -10 * np.log10(np.clip(1 - probs, 1e-10, 1))
        tp_q = mqual[y_test == 1]
        fp_q = mqual[y_test == 0]
        ax.hist(tp_q, bins=100, alpha=0.4, label=f"{model_name} TP", color=color, density=True)
        ax.hist(fp_q, bins=100, alpha=0.2, label=f"{model_name} FP", color=color, density=True, linestyle="--")
    ax.set_xlabel("MQUAL (Phred-scaled)")
    ax.set_ylabel("Density")
    ax.set_title("Quality Score Distribution — Test Set")
    ax.set_xlim(0, 40)
    ax.legend(fontsize=9)
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

    dnn_runtime = dnn_meta.get("training_runtime_metrics", [])
    if dnn_runtime:
        epochs = [r["epoch"] for r in dnn_runtime]
        axes[0, 1].plot(epochs, [r["train_auc"] for r in dnn_runtime], color=COLOR_DNN, alpha=0.4, lw=1, label="Train")
        axes[0, 1].plot(epochs, [r["val_auc"] for r in dnn_runtime], color=COLOR_DNN, lw=2, label="Val")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("AUC")
        axes[0, 1].set_title("DNN — AUC")
        axes[0, 1].legend()
        axes[0, 1].grid(visible=True, alpha=0.3)

        axes[1, 1].plot(
            epochs, [r["train_logloss"] for r in dnn_runtime], color=COLOR_DNN, alpha=0.4, lw=1, label="Train"
        )
        axes[1, 1].plot(epochs, [r["val_logloss"] for r in dnn_runtime], color=COLOR_DNN, lw=2, label="Val")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Logloss")
        axes[1, 1].set_title("DNN — Logloss")
        axes[1, 1].legend()
        axes[1, 1].grid(visible=True, alpha=0.3)

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
        delta_auc = (auc_d - auc_x) if (auc_d is not None and auc_x is not None) else None
        delta_aupr = (aupr_d - aupr_x) if (aupr_d is not None and aupr_x is not None) else None
        chrom_rows.append(
            {
                "chrom": chrom,
                "n": int(mask_c.sum()),
                "auc_xgb": _fmt(auc_x),
                "auc_dnn": _fmt(auc_d),
                "delta_auc": _fmt(delta_auc),
                "delta_auc_class": _delta_class(delta_auc, "auc"),
                "aupr_xgb": _fmt(aupr_x),
                "aupr_dnn": _fmt(aupr_d),
                "delta_aupr": _fmt(delta_aupr),
                "delta_aupr_class": _delta_class(delta_aupr, "aupr"),
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
        delta = (auc_d - auc_x) if (auc_d is not None and auc_x is not None) else None
        rows.append(
            {
                "value": str(val) if val != "" else "(empty)",
                "n": n,
                "auc_xgb": auc_x,
                "auc_dnn": auc_d,
                "aupr_xgb": aupr_x,
                "aupr_dnn": aupr_d,
                "delta_auc": delta,
                "delta_class": _delta_class(delta, "auc"),
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
        delta = (auc_d - auc_x) if (auc_d is not None and auc_x is not None) else None
        rows.append(
            {
                "value": str(interval),
                "n": n,
                "auc_xgb": auc_x,
                "auc_dnn": auc_d,
                "delta_auc": delta,
                "delta_class": _delta_class(delta, "auc"),
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

    delta_map = {r["value"]: r["delta_auc"] for r in rows if r["delta_auc"] is not None}
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
    ax.set_title("Delta AUC (DNN - XGB) by Trinucleotide Context")
    plt.colorbar(im, ax=ax, label="Delta AUC (green = DNN better)", shrink=0.6)
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
                for f_key in ["auc_xgb", "auc_dnn", "aupr_xgb", "aupr_dnn", "delta_auc"]:
                    if f_key in r and r[f_key] is not None and not isinstance(r[f_key], str):
                        r[f_key] = _fmt(r[f_key])
                    elif f_key in r and r[f_key] is None:
                        r[f_key] = "N/A"

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
