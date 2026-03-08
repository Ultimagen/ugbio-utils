"""Generate an HTML report comparing N SRSNV models.

Reads prediction parquet files and metadata JSONs produced by multiple models
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

MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#7f7f7f"]
LINE_STYLES = ["-", "--", "-.", ":"]
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


def load_data(models: list[dict]) -> dict:
    """Load parquet predictions and JSON metadata for all models.

    Parameters
    ----------
    models : list[dict]
        Each dict must have ``name`` (str), ``parquet`` (Path), ``metadata`` (Path).
        Optionally ``color`` (str) to override the default palette.

    Returns
    -------
    dict
        models, model_names, merged DataFrame, model_dfs, model_metas.
    """
    model_descs: list[dict] = []
    model_dfs: dict[str, pd.DataFrame] = {}
    model_metas: dict[str, dict] = {}

    for i, m in enumerate(models):
        name = m["name"]
        color = m.get("color", MODEL_COLORS[i % len(MODEL_COLORS)])
        ls = LINE_STYLES[i % len(LINE_STYLES)]
        model_descs.append({"name": name, "color": color, "linestyle": ls})

        logger.info("Loading %s parquet: %s", name, m["parquet"])
        model_dfs[name] = pd.read_parquet(m["parquet"])

        logger.info("Loading %s metadata: %s", name, m["metadata"])
        with open(m["metadata"]) as f:
            model_metas[name] = json.load(f)

    model_names = [md["name"] for md in model_descs]
    base_name = model_names[0]

    merged = model_dfs[base_name].copy()
    merged = merged.rename(
        columns={
            "prob_orig": f"prob_{base_name}",
            "SNVQ": f"snvq_{base_name}",
            "MQUAL": f"mqual_{base_name}",
        }
    )

    for name in model_names[1:]:
        other = model_dfs[name][["CHROM", "POS", "RN", "prob_orig", "SNVQ", "MQUAL"]].rename(
            columns={
                "prob_orig": f"prob_{name}",
                "SNVQ": f"snvq_{name}",
                "MQUAL": f"mqual_{name}",
            }
        )
        merged = merged.merge(other, on=["CHROM", "POS", "RN"])

    logger.info("Merged dataframe shape: %s", merged.shape)
    return {
        "models": model_descs,
        "model_names": model_names,
        "merged": merged,
        "model_dfs": model_dfs,
        "model_metas": model_metas,
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


def _finite_hist_range(values: np.ndarray, *, min_floor: float = 0.0) -> tuple[float, float]:
    """Return an x-range that includes the full finite tail with a small right pad."""
    finite = np.asarray(values)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return min_floor, max(min_floor + 1.0, 1.0)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    lo = max(min_floor, lo)
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    pad = max(span * 0.02, 0.5)
    return lo, hi + pad


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


def _relative_improvement(base_val: float | None, model_val: float | None, metric_name: str) -> float | None:
    """Compute relative improvement of *model_val* over *base_val*.

    Higher-is-better (AUC, AUPR): (model - base) / (1 - base) * 100
    Lower-is-better (logloss, brier): (base - model) / base * 100
    """
    if base_val is None or model_val is None:
        return None
    if metric_name in ("logloss", "brier"):
        if base_val == 0:
            return None
        return (base_val - model_val) / base_val * 100
    gap = 1.0 - base_val
    if gap == 0:
        return None
    return (model_val - base_val) / gap * 100


def _improvement_class(rel_imp: float | None) -> str:
    if rel_imp is None:
        return ""
    return "better" if rel_imp > 0 else "worse"


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _fig_to_b64(fig: plt.Figure, dpi: int = 150) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _load_dnn_epoch_metrics(meta: dict) -> pd.DataFrame | None:
    """Load per-epoch training metrics from Lightning CSVLogger output."""
    ckpt_paths = meta.get("best_checkpoint_paths", [])
    if not ckpt_paths:
        return None

    ckpt_path = Path(ckpt_paths[0])
    logs_dir = ckpt_path.parent / (ckpt_path.name.split(".dnn_model_fold_")[0] + ".lightning_logs")
    if not logs_dir.is_dir():
        candidates = list(ckpt_path.parent.glob("*.lightning_logs"))
        logs_dir = candidates[0] if candidates else None
    if logs_dir is None or not logs_dir.is_dir():
        return None

    csv_files = sorted(logs_dir.glob("fold_*/metrics.csv"))
    if not csv_files:
        return None

    frames = []
    for csv_file in csv_files:
        try:
            metrics_df = pd.read_csv(csv_file)
            if "epoch" in metrics_df.columns:
                frames.append(metrics_df.groupby("epoch").first().reset_index())
        except Exception:  # noqa: BLE001, S112
            continue

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True).groupby("epoch").mean(numeric_only=True).reset_index()
    logger.info("Loaded DNN epoch metrics: %d epochs from %d fold(s)", len(combined), len(frames))
    return combined


def _get_training_curves(meta: dict) -> dict | None:
    """Extract training curves from model metadata, supporting XGB and DNN formats."""
    xgb_results = meta.get("training_results", [])
    if isinstance(xgb_results, list) and xgb_results:
        res = xgb_results[0]
        v0 = res.get("validation_0", {})
        v1 = res.get("validation_1", {})
        if "auc" in v0:
            return {
                "x": np.arange(1, len(v0["auc"]) + 1),
                "train_auc": v0.get("auc"),
                "val_auc": v1.get("auc"),
                "train_loss": v0.get("logloss"),
                "val_loss": v1.get("logloss"),
                "x_label": "Boosting Round",
            }

    epoch_df = _load_dnn_epoch_metrics(meta)
    if epoch_df is not None and not epoch_df.empty:
        result: dict = {"x": epoch_df["epoch"].to_numpy(), "x_label": "Epoch"}
        for key in ("train_auc", "val_auc", "train_loss", "val_loss"):
            result[key] = epoch_df[key].to_numpy() if key in epoch_df.columns else None
        return result
    return None


# ---------------------------------------------------------------------------
# Compute all report data
# ---------------------------------------------------------------------------


def compute_report_data(data: dict) -> dict:  # noqa: C901, PLR0912, PLR0915
    """Compute all metrics, tables, and plots for the N-model comparison report."""
    merged = data["merged"]
    model_names = data["model_names"]
    model_descs = data["models"]
    model_metas = data["model_metas"]
    model_dfs = data["model_dfs"]
    baseline = model_names[0]
    non_baseline = model_names[1:]
    n_models = len(model_names)

    m_test = merged[merged["fold_id"].isna()]
    m_val = merged[merged["fold_id"] == 1]
    y_test = m_test["label"].astype(int).to_numpy()
    y_val = m_val["label"].astype(int).to_numpy()

    probs_test = {n: m_test[f"prob_{n}"].to_numpy() for n in model_names}
    probs_val = {n: m_val[f"prob_{n}"].to_numpy() for n in model_names}
    snvqs_test = {n: m_test[f"snvq_{n}"].to_numpy() for n in model_names}
    mquals_test = {n: m_test[f"mqual_{n}"].to_numpy() for n in model_names}

    report: dict = {
        "plots": {},
        "models": model_descs,
        "model_names": model_names,
        "baseline": baseline,
        "non_baseline": non_baseline,
    }

    # ---- 1. Executive summary ----
    logger.info("Computing executive summary metrics")
    summary_rows = []
    for split_name, y, split_probs in [
        ("Validation", y_val, probs_val),
        ("Test (holdout)", y_test, probs_test),
    ]:
        all_m = {n: _calc_metrics(y, split_probs[n]) for n in model_names}
        base_m = all_m[baseline]
        for metric in ["auc", "aupr", "logloss", "brier"]:
            values = {n: _fmt(all_m[n][metric]) for n in model_names}
            rel_imps: dict = {}
            for n in non_baseline:
                ri = _relative_improvement(base_m[metric], all_m[n][metric], metric)
                rel_imps[n] = {"val": _fmt(ri, 1) if ri is not None else "N/A", "cls": _improvement_class(ri)}
            summary_rows.append({"split": split_name, "metric": metric.upper(), "values": values, "rel_imps": rel_imps})
    report["summary"] = summary_rows

    # ---- 2. Dataset overview ----
    logger.info("Computing dataset overview")
    first_meta = model_metas[model_names[0]]
    split_info = first_meta.get("split_prevalence", {})
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

    manifest = first_meta.get("split_manifest", {})
    model_totals = {n: len(model_dfs[n]) for n in model_names}
    report["split_config"] = {
        "holdout_chromosomes": ", ".join(manifest.get("holdout_chromosomes", [])),
        "val_fraction": manifest.get("val_fraction"),
        "hash_key": manifest.get("hash_key"),
        "split_mode": manifest.get("split_mode"),
        "model_totals": model_totals,
        "merged_total": len(merged),
    }

    # ---- 3. ROC curves ----
    logger.info("Generating ROC curves")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, split_name, y, split_probs in [
        (axes[0], "Validation", y_val, probs_val),
        (axes[1], "Test (holdout)", y_test, probs_test),
    ]:
        for m in model_descs:
            p = split_probs[m["name"]]
            fpr, tpr, _ = roc_curve(y, p)
            auc_val = roc_auc_score(y, p)
            ax.plot(
                fpr, tpr, color=m["color"], lw=2, linestyle=m["linestyle"], label=f"{m['name']} (AUC={auc_val:.4f})"
            )
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {split_name}")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    report["plots"]["roc"] = _fig_to_b64(fig)

    # ---- 4. Precision-Recall curves ----
    logger.info("Generating PR curves")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, split_name, y, split_probs in [
        (axes[0], "Validation", y_val, probs_val),
        (axes[1], "Test (holdout)", y_test, probs_test),
    ]:
        for m in model_descs:
            p = split_probs[m["name"]]
            prec, rec, _ = precision_recall_curve(y, p)
            aupr_val = average_precision_score(y, p)
            ax.plot(
                rec, prec, color=m["color"], lw=2, linestyle=m["linestyle"], label=f"{m['name']} (AUPR={aupr_val:.4f})"
            )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall — {split_name}")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    report["plots"]["pr"] = _fig_to_b64(fig)

    # ---- 5. Score distribution (SNVQ / Phred-scaled) ----
    logger.info("Generating score distributions")
    tp_mask_global = y_test == 1
    all_snvq = np.concatenate([snvqs_test[n] for n in model_names])
    snvq_xmin, snvq_xmax = _finite_hist_range(all_snvq, min_floor=0.0)
    snvq_bins = np.linspace(snvq_xmin, snvq_xmax, 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    for m in model_descs:
        snvq = snvqs_test[m["name"]]
        ax.hist(
            snvq[tp_mask_global],
            bins=snvq_bins,
            density=True,
            histtype="step",
            color=COLOR_BETTER,
            linestyle=m["linestyle"],
            lw=1.5,
            label=f"{m['name']} TP",
        )
        ax.hist(
            snvq[~tp_mask_global],
            bins=snvq_bins,
            density=True,
            histtype="step",
            color=COLOR_WORSE,
            linestyle=m["linestyle"],
            lw=1.5,
            label=f"{m['name']} FP",
        )
    ax.set_xlabel("SNVQ (Phred-scaled)")
    ax.set_ylabel("Density")
    ax.set_title("SNVQ Distribution — Test Set")
    ax.set_yscale("log")
    ax.set_xlim(snvq_xmin, snvq_xmax)
    ax.legend(fontsize=9, ncol=max(1, n_models // 2))
    ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    report["plots"]["score_dist"] = _fig_to_b64(fig)

    # ---- 6. Calibration + SNVQ-MQUAL mapping ----
    logger.info("Generating calibration & SNVQ-MQUAL mapping")
    fig, (ax_cal, ax_lut, ax_hist) = plt.subplots(3, 1, figsize=(9, 14), gridspec_kw={"height_ratios": [3, 2, 1]})
    for m in model_descs:
        p = probs_test[m["name"]]
        prob_true, prob_pred = calibration_curve(y_test, p, n_bins=20, strategy="uniform")
        ece = float(np.mean(np.abs(prob_true - prob_pred)))
        ax_cal.plot(prob_pred, prob_true, "o-", color=m["color"], lw=2, label=f"{m['name']} (ECE={ece:.4f})")
    ax_cal.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax_cal.set_xlabel("Mean Predicted Probability")
    ax_cal.set_ylabel("Fraction of Positives")
    ax_cal.set_title("Calibration Curve — Test Set")
    ax_cal.legend(fontsize=9)
    ax_cal.grid(visible=True, alpha=0.3)

    for m in model_descs:
        meta = model_metas[m["name"]]
        lut = meta.get("quality_recalibration_table")
        if lut is not None and len(lut) == _LUT_PAIR_LEN:
            x_lut, y_lut = np.array(lut[0]), np.array(lut[1])
            ax_lut.plot(
                x_lut, y_lut, color=m["color"], linestyle=m["linestyle"], lw=2, label=f"{m['name']}: MQUAL→SNVQ"
            )

    ax_lut_twin = ax_lut.twinx()
    all_mq = np.concatenate([mquals_test[n] for n in model_names])
    all_mq_fin = all_mq[np.isfinite(all_mq)]
    mq_max = float(np.percentile(all_mq_fin, 99.5)) if len(all_mq_fin) > 0 else 100.0
    lut_bins = np.linspace(0, mq_max, 80)
    for m in model_descs:
        mq_fin = mquals_test[m["name"]]
        mq_fin = mq_fin[np.isfinite(mq_fin)]
        ax_lut_twin.hist(
            mq_fin,
            bins=lut_bins,
            density=True,
            histtype="step",
            color="gray",
            linestyle=m["linestyle"],
            lw=0.8,
            alpha=0.5,
            label=f"{m['name']} MQUAL dist",
        )
    ax_lut.set_xlabel("MQUAL")
    ax_lut.set_ylabel("SNVQ")
    ax_lut.set_xlim(0, mq_max)
    ax_lut.set_title("MQUAL → SNVQ Mapping")
    ax_lut.legend(loc="upper left", fontsize=8)
    ax_lut_twin.set_ylabel("Density")
    ax_lut_twin.legend(loc="upper right", fontsize=7)
    ax_lut.grid(visible=True, alpha=0.3)

    prob_bins = np.linspace(0, 1, 80)
    for m in model_descs:
        ax_hist.hist(
            probs_test[m["name"]],
            bins=prob_bins,
            density=True,
            histtype="step",
            color=m["color"],
            linestyle=m["linestyle"],
            lw=1.5,
            label=m["name"],
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
    for name in model_names:
        mqual_arr = mquals_test[name]
        for lbl_name, mask in [("TP", y_test == 1), ("FP", y_test == 0)]:
            vals = mqual_arr[mask]
            vals = vals[np.isfinite(vals)]
            row: dict = {"model": name, "label": lbl_name}
            for p in percentiles:
                row[f"p{p}"] = _fmt(float(np.percentile(vals, p)), 1) if len(vals) > 0 else "N/A"
            qual_rows.append(row)
    report["quality_percentiles"] = qual_rows

    all_mqual = np.concatenate([mquals_test[n] for n in model_names])
    mqual_xmin, mqual_xmax = _finite_hist_range(all_mqual, min_floor=0.0)
    mqual_bins = np.linspace(mqual_xmin, mqual_xmax, 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    for m in model_descs:
        mq = mquals_test[m["name"]]
        fin = np.isfinite(mq)
        ax.hist(
            mq[fin & (y_test == 1)],
            bins=mqual_bins,
            density=True,
            histtype="step",
            color=COLOR_BETTER,
            linestyle=m["linestyle"],
            lw=1.5,
            label=f"{m['name']} TP",
        )
        ax.hist(
            mq[fin & (y_test == 0)],
            bins=mqual_bins,
            density=True,
            histtype="step",
            color=COLOR_WORSE,
            linestyle=m["linestyle"],
            lw=1.5,
            label=f"{m['name']} FP",
        )
    ax.set_xlabel("MQUAL (Phred-scaled)")
    ax.set_ylabel("Density")
    ax.set_title("Quality Score Distribution — Test Set")
    ax.set_yscale("log")
    ax.set_xlim(mqual_xmin, mqual_xmax)
    ax.legend(fontsize=9, ncol=max(1, n_models // 2))
    ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    report["plots"]["quality_dist"] = _fig_to_b64(fig)

    # ---- 8. Training progress ----
    logger.info("Generating training progress curves")
    fig, axes = plt.subplots(2, n_models, figsize=(8 * n_models, 10), squeeze=False)
    for col, m in enumerate(model_descs):
        curves = _get_training_curves(model_metas[m["name"]])
        if curves is None:
            axes[0, col].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0, col].transAxes)
            axes[0, col].set_title(f"{m['name']} — AUC")
            axes[1, col].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1, col].transAxes)
            axes[1, col].set_title(f"{m['name']} — Loss")
            continue
        x = curves["x"]
        if curves.get("val_auc") is not None:
            if curves.get("train_auc") is not None:
                axes[0, col].plot(x, curves["train_auc"], color=m["color"], alpha=0.4, lw=1, label="Train")
            axes[0, col].plot(x, curves["val_auc"], color=m["color"], lw=2, label="Val")
        axes[0, col].set_xlabel(curves["x_label"])
        axes[0, col].set_ylabel("AUC")
        axes[0, col].set_title(f"{m['name']} — AUC")
        axes[0, col].legend()
        axes[0, col].grid(visible=True, alpha=0.3)
        if curves.get("val_loss") is not None:
            if curves.get("train_loss") is not None:
                axes[1, col].plot(x, curves["train_loss"], color=m["color"], alpha=0.4, lw=1, label="Train")
            axes[1, col].plot(x, curves["val_loss"], color=m["color"], lw=2, label="Val")
        axes[1, col].set_xlabel(curves["x_label"])
        axes[1, col].set_ylabel("Loss")
        axes[1, col].set_title(f"{m['name']} — Loss")
        axes[1, col].legend()
        axes[1, col].grid(visible=True, alpha=0.3)

    for row in range(2):
        all_ylims = [axes[row, c].get_ylim() for c in range(n_models) if axes[row, c].has_data()]
        if all_ylims:
            ymin = min(yl[0] for yl in all_ylims)
            ymax = max(yl[1] for yl in all_ylims)
            margin = 0.02 * (ymax - ymin) if ymax > ymin else 0.01
            for c in range(n_models):
                axes[row, c].set_ylim(ymin - margin, ymax + margin)
    fig.suptitle("Training Progress", fontsize=14, y=1.01)
    fig.tight_layout()
    report["plots"]["training_progress"] = _fig_to_b64(fig)

    # ---- 9. Confusion matrices ----
    logger.info("Generating confusion matrices")
    fig, axes_cm = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
    threshold = 0.5
    for idx, m in enumerate(model_descs):
        ax = axes_cm[0, idx]
        preds = (probs_test[m["name"]] >= threshold).astype(int)
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
        ax.set_title(f"{m['name']} (thr={threshold}, F1={f1:.4f})")
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
        crow: dict = {"chrom": chrom, "n": int(mask_c.sum()), "auc": {}, "aupr": {}, "ri_auc": {}, "ri_aupr": {}}
        base_auc = base_aupr = None
        for name in model_names:
            pc = m_test.loc[mask_c, f"prob_{name}"].to_numpy()
            a = _safe_auc(yc, pc)
            ap = _safe_aupr(yc, pc)
            crow["auc"][name] = _fmt(a)
            crow["aupr"][name] = _fmt(ap)
            if name == baseline:
                base_auc, base_aupr = a, ap
            else:
                ri_a = _relative_improvement(base_auc, a, "auc")
                ri_ap = _relative_improvement(base_aupr, ap, "aupr")
                crow["ri_auc"][name] = {
                    "val": _fmt(ri_a, 1) if ri_a is not None else "N/A",
                    "cls": _improvement_class(ri_a),
                }
                crow["ri_aupr"][name] = {
                    "val": _fmt(ri_ap, 1) if ri_ap is not None else "N/A",
                    "cls": _improvement_class(ri_ap),
                }
        chrom_rows.append(crow)
    report["per_chrom"] = chrom_rows

    if len(chroms_in_test) > 1:
        fig, axes_pc = plt.subplots(1, 2, figsize=(12, 5))
        x_pos = np.arange(len(chroms_in_test))
        bar_w = 0.8 / n_models
        for ax, metric in [(axes_pc[0], "auc"), (axes_pc[1], "aupr")]:
            for i, m in enumerate(model_descs):
                vals = [float(r[metric][m["name"]]) if r[metric][m["name"]] != "N/A" else 0 for r in chrom_rows]
                offset = (i - (n_models - 1) / 2) * bar_w
                ax.bar(x_pos + offset, vals, bar_w, label=m["name"], color=m["color"])
            ax.set_xticks(x_pos)
            ax.set_xticklabels(chroms_in_test)
            ax.set_ylabel(metric.upper())
            ax.set_title(f"Per-Chromosome {metric.upper()} — Test Set")
            ax.legend(fontsize=9)
            ax.grid(visible=True, alpha=0.3, axis="y")
        fig.tight_layout()
        report["plots"]["per_chrom"] = _fig_to_b64(fig)

    # ---- 11. Model architecture summary ----
    logger.info("Building model architecture summary")
    arch_list = []
    for name in model_names:
        meta = model_metas[name]
        info: dict = {"name": name, "params": [], "feature_names": None, "n_features": None}

        mp = meta.get("model_params", {})
        if mp.get("n_estimators"):
            info["params"].append(("n_estimators", mp["n_estimators"]))
        if mp.get("early_stopping_rounds"):
            info["params"].append(("early_stopping_rounds", mp["early_stopping_rounds"]))
        em = mp.get("eval_metric", [])
        if em:
            info["params"].append(("eval_metric", ", ".join(em)))

        features = meta.get("features", [])
        if features:
            info["n_features"] = len(features)
            info["params"].append(("Number of features", len(features)))
            info["feature_names"] = ", ".join(
                f.get("name", str(f)) if isinstance(f, dict) else str(f) for f in features
            )

        ss = meta.get("split_summary", {})
        if ss.get("best_iteration") is not None:
            info["params"].append(("Best iteration", ss["best_iteration"]))

        dnn_arch = meta.get("model_architecture", {})
        if dnn_arch.get("class_name"):
            info["params"].append(("Architecture", dnn_arch["class_name"]))
        if dnn_arch.get("trainable_parameters") is not None:
            info["params"].append(("Trainable parameters", f"{dnn_arch['trainable_parameters']:,}"))

        tp = meta.get("training_parameters", {})
        param_map = [
            ("epochs", "Epochs (max)"),
            ("patience", "Patience"),
            ("batch_size", "Batch size"),
            ("learning_rate", "Learning rate"),
            ("use_amp", "Mixed precision (AMP)"),
        ]
        for key, label in param_map:
            if tp.get(key) is not None:
                info["params"].append((label, tp[key]))

        tr = meta.get("training_results", [])
        if isinstance(tr, list) and tr:
            fold0 = tr[0]
            if fold0.get("best_epoch") is not None:
                info["params"].append(("Best epoch", fold0["best_epoch"]))
            if fold0.get("stopped_early") is not None:
                info["params"].append(("Early stopped", fold0["stopped_early"]))

        arch_list.append(info)
    report["arch"] = arch_list

    # ---- 12. SNVQ threshold recall ----
    logger.info("Computing SNVQ threshold recall")
    n_tp_test = int(y_test.sum())
    tp_mask_test = y_test == 1
    median_snvqs = {n: float(np.median(snvqs_test[n][tp_mask_test])) for n in model_names}

    snvq_recall_rows = []
    for thr in SNVQ_THRESHOLDS:
        row_r: dict = {"threshold": thr, "recalls": {}, "counts": {}, "ri": {}}
        base_recall = None
        for name in model_names:
            recall = float(np.sum((snvqs_test[name] >= thr) & tp_mask_test)) / n_tp_test if n_tp_test > 0 else 0
            count = int(np.sum((snvqs_test[name] >= thr) & tp_mask_test))
            row_r["recalls"][name] = _fmt(recall)
            row_r["counts"][name] = f"{count:,}"
            if name == baseline:
                base_recall = recall
            else:
                ri = _relative_improvement(base_recall, recall, "auc")
                row_r["ri"][name] = {"val": _fmt(ri, 1) if ri is not None else "N/A", "cls": _improvement_class(ri)}
        snvq_recall_rows.append(row_r)

    report["snvq_recall"] = {
        "rows": snvq_recall_rows,
        "median_snvqs": {n: _fmt(median_snvqs[n], 1) for n in model_names},
        "n_tp": f"{n_tp_test:,}",
    }

    # ---- 12b. SNVQ vs ppmSeq tags ----
    logger.info("Computing SNVQ vs ppmSeq tag heatmaps")
    report["snvq_tags"] = _compute_snvq_vs_ppmseq_tags(m_test, tp_mask_test, snvqs_test, model_names, model_descs)

    # ---- 15. Logit histogram ----
    logger.info("Generating logit histogram")
    report["plots"]["logit_hist"] = _compute_logit_histogram(m_test, y_test, model_names, model_descs)

    # ---- 16. SNVQ histogram by mixed status ----
    logger.info("Generating SNVQ histogram by mixed status")
    report["plots"]["snvq_mixed_hist"] = _compute_snvq_mixed_histogram(
        m_test,
        y_test,
        snvqs_test,
        model_names,
        model_descs,
    )

    # ---- 17. SNVQ by trinucleotide context ----
    logger.info("Generating SNVQ by trinucleotide context")
    report["plots"]["trinuc_snvq"] = _compute_trinuc_snvq_comparison(
        m_test,
        y_test,
        snvqs_test,
        model_names,
        model_descs,
    )

    # ---- 13. Error analysis ----
    report["error_analysis"] = _compute_error_analysis(m_test, y_test, model_names, baseline, model_descs)

    return report


# ---------------------------------------------------------------------------
# Error analysis helpers
# ---------------------------------------------------------------------------


def _sliced_auc(
    df: pd.DataFrame,
    y: np.ndarray,
    col: str,
    model_names: list[str],
    baseline: str,
    values: list | None = None,
) -> list[dict]:
    """Compute AUC/AUPR for all models sliced by a categorical column."""
    if values is None:
        values = sorted(df[col].dropna().unique())
    rows = []
    non_baseline = [n for n in model_names if n != baseline]
    for val in values:
        mask = df[col] == val
        n = int(mask.sum())
        if n < MIN_SLICE_SIZE:
            continue
        yi = y[mask.to_numpy()]
        if len(np.unique(yi)) < MIN_UNIQUE_LABELS:
            continue
        row: dict = {"value": str(val) if val != "" else "(empty)", "n": n, "auc": {}, "aupr": {}, "ri_auc": {}}
        base_auc = None
        for name in model_names:
            pi = df.loc[mask, f"prob_{name}"].to_numpy()
            row["auc"][name] = _safe_auc(yi, pi)
            row["aupr"][name] = _safe_aupr(yi, pi)
            if name == baseline:
                base_auc = row["auc"][name]
        for name in non_baseline:
            row["ri_auc"][name] = _relative_improvement(base_auc, row["auc"][name], "auc")
        first_ri = row["ri_auc"].get(non_baseline[0]) if non_baseline else None
        row["ri_class"] = _improvement_class(first_ri)
        rows.append(row)
    return rows


def _sliced_auc_numeric_bins(
    df: pd.DataFrame,
    y: np.ndarray,
    col: str,
    model_names: list[str],
    baseline: str,
    n_bins: int = 10,
) -> list[dict]:
    """Compute AUC for all models sliced by binned numeric column."""
    valid = df[col].notna()
    vals = df.loc[valid, col]
    try:
        bins = pd.qcut(vals, n_bins, duplicates="drop")
    except ValueError:
        bins = pd.cut(vals, n_bins, duplicates="drop")
    non_baseline = [n for n in model_names if n != baseline]
    rows = []
    for interval in sorted(bins.unique()):
        mask = valid & (bins == interval)
        n = int(mask.sum())
        if n < MIN_SLICE_SIZE:
            continue
        yi = y[mask.to_numpy()]
        if len(np.unique(yi)) < MIN_UNIQUE_LABELS:
            continue
        row: dict = {"value": str(interval), "n": n, "auc": {}, "aupr": {}, "ri_auc": {}}
        base_auc = None
        for name in model_names:
            pi = df.loc[mask, f"prob_{name}"].to_numpy()
            row["auc"][name] = _safe_auc(yi, pi)
            if name == baseline:
                base_auc = row["auc"][name]
        for name in non_baseline:
            row["ri_auc"][name] = _relative_improvement(base_auc, row["auc"][name], "auc")
        first_ri = row["ri_auc"].get(non_baseline[0]) if non_baseline else None
        row["ri_class"] = _improvement_class(first_ri)
        rows.append(row)
    return rows


def _make_grouped_bar(
    rows: list[dict],
    title: str,
    model_names: list[str],
    model_descs: list[dict],
    xlabel: str = "",
) -> str:
    """Create a grouped bar chart from sliced AUC rows, return base64 PNG."""
    if not rows:
        return ""
    labels = [r["value"] for r in rows]
    n_m = len(model_names)
    x_pos = np.arange(len(labels))
    bar_width = 0.8 / max(n_m, 1)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 5))
    for i, m in enumerate(model_descs):
        vals = []
        for r in rows:
            v = r["auc"].get(m["name"])
            vals.append(v if v is not None and isinstance(v, int | float) else 0)
        offset = (i - (n_m - 1) / 2) * bar_width
        ax.bar(x_pos + offset, vals, bar_width, label=m["name"], color=m["color"])

    ax.set_xticks(x_pos)
    needs_rotation = len(labels) > MAX_TICK_LABELS_HORIZONTAL
    ax.set_xticklabels(labels, rotation=45 if needs_rotation else 0, ha="right" if needs_rotation else "center")
    ax.set_ylabel("AUC")
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.legend(fontsize=9)
    ax.grid(visible=True, alpha=0.3, axis="y")

    all_vals = [
        r["auc"].get(n, 0)
        for r in rows
        for n in model_names
        if isinstance(r["auc"].get(n), int | float) and r["auc"].get(n, 0) > 0
    ]
    if all_vals:
        ymin = min(all_vals)
        margin = (1 - ymin) * 0.1
        ax.set_ylim(max(0, ymin - margin), 1.0)

    fig.tight_layout()
    return _fig_to_b64(fig)


def _make_trinuc_heatmap(
    test_df: pd.DataFrame,
    y: np.ndarray,
    model_names: list[str],
    baseline: str,
    model_descs: list[dict],
) -> tuple[str, list[dict]]:
    """Create trinuc context heatmap(s) of relative improvement and return (b64_png, rows)."""
    test_df = test_df.copy()
    test_df["trinuc"] = test_df["X_PREV1"].astype(str) + test_df["REF"].astype(str) + test_df["X_NEXT1"].astype(str)
    rows = _sliced_auc(test_df, y, "trinuc", model_names, baseline)
    if not rows:
        return "", []

    non_baseline = [n for n in model_names if n != baseline]
    if not non_baseline:
        return "", rows

    bases = ["A", "C", "G", "T"]
    trinucs = sorted({r["value"] for r in rows})
    prev_bases = sorted({t[0] for t in trinucs if len(t) == TRINUC_LEN})
    next_bases = sorted({t[2] for t in trinucs if len(t) == TRINUC_LEN})

    n_panels = len(non_baseline)
    fig, axes_h = plt.subplots(1, max(n_panels, 1), figsize=(8 * max(n_panels, 1), 12), squeeze=False)

    for panel_idx, model_name in enumerate(non_baseline):
        delta_map = {}
        for r in rows:
            ri = r["ri_auc"].get(model_name)
            if ri is not None:
                delta_map[r["value"]] = ri
        matrix = np.full((len(prev_bases) * len(bases), len(next_bases)), np.nan)
        y_labels = []
        for pi, pb in enumerate(prev_bases):
            for bi, rb in enumerate(bases):
                y_labels.append(f"{pb}{rb}")
                for ki, nb in enumerate(next_bases):
                    key = f"{pb}{rb}{nb}"
                    if key in delta_map:
                        matrix[pi * len(bases) + bi, ki] = delta_map[key]

        ax = axes_h[0, panel_idx]
        vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix))) if not np.all(np.isnan(matrix)) else 0.01
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(next_bases)))
        ax.set_xticklabels(next_bases)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("Next Base")
        ax.set_ylabel("Prev + Ref Base")
        ax.set_title(f"Rel. Imp. % ({model_name} vs {baseline})")
        plt.colorbar(im, ax=ax, label="% gap closed (green=better)", shrink=0.6)
    fig.tight_layout()
    return _fig_to_b64(fig), rows


def _compute_error_agreement(
    df: pd.DataFrame,
    y: np.ndarray,
    model_names: list[str],
    model_descs: list[dict],
    threshold: float = 0.5,
) -> dict:
    """Classify each test read by correctness of each model at given threshold."""
    n_total = len(y)
    correct = {name: ((df[f"prob_{name}"].to_numpy() >= threshold).astype(int) == y) for name in model_names}

    all_correct = np.ones(n_total, dtype=bool)
    all_wrong = np.ones(n_total, dtype=bool)
    for name in model_names:
        all_correct &= correct[name]
        all_wrong &= ~correct[name]

    categories = [
        {"name": "All correct", "count_raw": int(all_correct.sum()), "color": COLOR_BETTER},
        {"name": "All wrong", "count_raw": int(all_wrong.sum()), "color": COLOR_WORSE},
    ]

    accounted = all_correct | all_wrong
    for m in model_descs:
        only_wrong = ~correct[m["name"]]
        for other in model_names:
            if other != m["name"]:
                only_wrong = only_wrong & correct[other]
        count = int(only_wrong.sum())
        if count > 0:
            categories.append({"name": f"Only {m['name']} wrong", "count_raw": count, "color": m["color"]})
            accounted = accounted | only_wrong

    other_count = int((~accounted).sum())
    if other_count > 0:
        categories.append({"name": "Other (mixed)", "count_raw": other_count, "color": "#888888"})

    for cat in categories:
        cat["count"] = f"{cat['count_raw']:,}"
        cat["pct"] = f"{100 * cat['count_raw'] / n_total:.2f}%"

    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.2), 5))
    names = [c["name"] for c in categories]
    counts = [c["count_raw"] for c in categories]
    colors = [c["color"] for c in categories]
    bars = ax.barh(names, counts, color=colors)
    for bar, cnt in zip(bars, counts, strict=True):
        ax.text(
            bar.get_width() + n_total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{cnt:,} ({100 * cnt / n_total:.1f}%)",
            va="center",
            fontsize=10,
        )
    ax.set_xlabel("Number of Reads")
    ax.set_title(f"Error Agreement (threshold={threshold}) — Test Set")
    ax.grid(visible=True, alpha=0.3, axis="x")
    fig.tight_layout()

    return {"categories": categories, "plot": _fig_to_b64(fig)}


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------


def _compute_snvq_vs_ppmseq_tags(  # noqa: PLR0915
    m_test: pd.DataFrame,
    tp_mask: np.ndarray,
    snvqs: dict[str, np.ndarray],
    model_names: list[str],
    model_descs: list[dict],
) -> dict:
    """Compute median SNVQ per ppmSeq start/end tag combination for all models."""
    result: dict = {}
    df_tp = m_test.loc[tp_mask].copy()

    st_col = "st" if "st" in df_tp.columns else None
    et_col = "et" if "et" in df_tp.columns else None
    if st_col is None or et_col is None:
        logger.warning("ppmSeq tag columns (st, et) not found — skipping SNVQ vs tags section")
        return result

    for name in model_names:
        df_tp[f"snvq_{name}"] = snvqs[name][tp_mask]
    df_tp[st_col] = df_tp[st_col].astype(str).replace({"": "(empty)", "nan": "(empty)"})
    df_tp[et_col] = df_tp[et_col].astype(str).replace({"": "(empty)", "nan": "(empty)"})

    pivots: dict[str, pd.DataFrame] = {}
    for name in model_names:
        pivots[name] = df_tp.pivot_table(
            index=st_col, columns=et_col, values=f"snvq_{name}", aggfunc="median", dropna=False
        )
    count_pivot = df_tp.pivot_table(
        index=st_col, columns=et_col, values=f"snvq_{model_names[0]}", aggfunc="count", dropna=False
    )

    preferred_order = ["MIXED", "MINUS", "PLUS", "(empty)"]
    all_st_raw: set = set()
    all_et_raw: set = set()
    for p in pivots.values():
        all_st_raw |= set(p.index)
        all_et_raw |= set(p.columns)
    all_st = [t for t in preferred_order if t in all_st_raw] + sorted(all_st_raw - set(preferred_order))
    all_et = [t for t in preferred_order if t in all_et_raw] + sorted(all_et_raw - set(preferred_order))

    for name in model_names:
        pivots[name] = pivots[name].reindex(index=all_st, columns=all_et)
    count_pivot = count_pivot.reindex(index=all_st, columns=all_et).fillna(0)
    pct_pivot = (count_pivot / count_pivot.to_numpy().sum()) * 100

    def _annot_table(val_df, pct_df):
        annot = val_df.copy().astype(object)
        for r in annot.index:
            for c in annot.columns:
                v = val_df.loc[r, c]
                p = pct_df.loc[r, c]
                annot.loc[r, c] = "" if pd.isna(v) else f"{v:.1f}\n[{p:.1f}%]"
        return annot

    n_panels = len(model_names)
    fig, axes_h = plt.subplots(1, n_panels, figsize=(10 * n_panels, 10), squeeze=False)

    for idx, m in enumerate(model_descs):
        ax = axes_h[0, idx]
        pivot = pivots[m["name"]]
        annot_piv = _annot_table(pivot, pct_pivot)
        sns.heatmap(
            pivot,
            annot=annot_piv,
            fmt="",
            cmap="inferno",
            cbar=True,
            linewidths=1,
            linecolor="black",
            annot_kws={"size": 11},
            square=True,
            ax=ax,
        )
        ax.set_xlabel("ppmSeq End Tag (et)")
        ax.set_ylabel("ppmSeq Start Tag (st)")
        ax.set_title(f"{m['name']} — Median SNVQ")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("SNVQ vs ppmSeq Start/End Tags (TP reads only, Test Set)", fontsize=14, y=1.02)
    fig.tight_layout()
    result["plot"] = _fig_to_b64(fig)
    return result


def _compute_logit_histogram(
    m_test: pd.DataFrame,
    y_test: np.ndarray,
    model_names: list[str],
    model_descs: list[dict],
) -> str:
    """Plot logit histograms split by FP / TP mixed / TP non-mixed for all models."""
    from ugbio_srsnv.srsnv_utils import prob_to_logit  # noqa: PLC0415

    tp_mask = y_test == 1
    st_vals = m_test["st"].astype(str).to_numpy() if "st" in m_test.columns else np.full(len(y_test), "")
    et_vals = m_test["et"].astype(str).to_numpy() if "et" in m_test.columns else np.full(len(y_test), "")
    is_mixed = (st_vals == "MIXED") & (et_vals == "MIXED")

    fp_mask = ~tp_mask
    tp_mixed_mask = tp_mask & is_mixed
    tp_nonmixed_mask = tp_mask & ~is_mixed

    logits = {n: prob_to_logit(m_test[f"prob_{n}"].to_numpy(), phred=True) for n in model_names}
    all_finite = np.concatenate([logits[n] for n in model_names])
    xlim_lo, xlim_hi = _finite_hist_range(all_finite, min_floor=-1e9)
    shared_bins = np.linspace(xlim_lo, xlim_hi, 100)

    categories = [
        (fp_mask, "FP", "tab:blue"),
        (tp_mixed_mask, "TP mixed", "green"),
        (tp_nonmixed_mask, "TP non-mixed", "red"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for m in model_descs:
        for mask, cat_label, clr in categories:
            vals = logits[m["name"]][mask]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                ax.hist(
                    vals,
                    bins=shared_bins,
                    density=True,
                    histtype="step",
                    color=clr,
                    linestyle=m["linestyle"],
                    lw=1.5,
                    label=f"{m['name']}: {cat_label}",
                )
    ax.set_xlabel("ML Logit (Phred scale)")
    ax.set_ylabel("Density")
    ax.set_title("Logit Distribution — Test Set")
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _compute_snvq_mixed_histogram(
    m_test: pd.DataFrame,
    y_test: np.ndarray,
    snvqs: dict[str, np.ndarray],
    model_names: list[str],
    model_descs: list[dict],
) -> str:
    """Plot SNVQ distribution for TP reads split by mixed status for all models."""
    tp_mask = y_test == 1
    st_vals = m_test["st"].astype(str).to_numpy() if "st" in m_test.columns else np.full(len(y_test), "")
    et_vals = m_test["et"].astype(str).to_numpy() if "et" in m_test.columns else np.full(len(y_test), "")

    is_mixed_start = st_vals == "MIXED"
    is_mixed_end = et_vals == "MIXED"
    nonmixed = tp_mask & ~is_mixed_start & ~is_mixed_end
    one_end = tp_mask & (is_mixed_start ^ is_mixed_end)
    both_ends = tp_mask & is_mixed_start & is_mixed_end

    all_snvq_tp = np.concatenate([snvqs[n][tp_mask] for n in model_names])
    xlim_min, xlim_max = _finite_hist_range(all_snvq_tp, min_floor=0.0)
    shared_bins = np.linspace(xlim_min, xlim_max, 100)

    categories = [
        (nonmixed, "non-mixed", "red"),
        (one_end, "mixed, one end", "blue"),
        (both_ends, "mixed, both ends", "green"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for m in model_descs:
        for mask, cat_label, clr in categories:
            vals = snvqs[m["name"]][mask]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                ax.hist(
                    vals,
                    bins=shared_bins,
                    density=True,
                    histtype="step",
                    color=clr,
                    linestyle=m["linestyle"],
                    lw=1.5,
                    label=f"{m['name']}: {cat_label}",
                )
    ax.set_xlabel("SNVQ")
    ax.set_ylabel("Density")
    ax.set_title("SNVQ by Mixed Status — TP reads, Test Set")
    ax.set_xlim(xlim_min, xlim_max)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _compute_trinuc_snvq_comparison(  # noqa: C901, PLR0912, PLR0915
    m_test: pd.DataFrame,
    y_test: np.ndarray,
    snvqs: dict[str, np.ndarray],
    model_names: list[str],
    model_descs: list[dict],
    q1: float = 0.1,
    q2: float = 0.9,
) -> str:
    """Plot median SNVQ by trinucleotide context for all models.

    Two-row layout: each row has a SNVQ panel on top and a density histogram below.
    Row 1 = forward trinucs (A>C .. G>T), Row 2 = complement (C>A .. T>G).
    """
    from matplotlib import gridspec  # noqa: PLC0415

    from ugbio_srsnv.srsnv_utils import (  # noqa: PLC0415
        FLOW_ORDER,
        get_trinuc_context_with_alt_fwd_vectorized,
        is_cycle_skip,
    )

    tp_mask = y_test == 1
    if tp_mask.sum() == 0:
        return ""

    all_df = m_test.copy()
    all_df["tcwa"] = (
        all_df["X_PREV1"].astype(str)
        + all_df["REF"].astype(str)
        + all_df["X_NEXT1"].astype(str)
        + all_df["ALT"].astype(str)
    )
    is_forward = all_df["REV"].astype(int) != 1
    all_df["tcwa"] = get_trinuc_context_with_alt_fwd_vectorized(all_df["tcwa"], is_forward)
    for name in model_names:
        all_df[f"snvq_{name}"] = snvqs[name]
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
    n_total_ctx = 192
    n_panel = 96

    all_df["tcwa_idx"] = all_df["tcwa"].map(tcwa_to_idx)
    all_df = all_df.dropna(subset=["tcwa_idx"])
    all_df["tcwa_idx"] = all_df["tcwa_idx"].astype(int)
    tp_df = all_df[all_df["label"] == 1]

    med: dict[str, np.ndarray] = {}
    lo: dict[str, np.ndarray] = {}
    hi: dict[str, np.ndarray] = {}
    for name in model_names:
        med[name] = np.full(n_total_ctx, np.nan)
        lo[name] = np.full(n_total_ctx, np.nan)
        hi[name] = np.full(n_total_ctx, np.nan)

    for idx, grp in tp_df.groupby("tcwa_idx"):
        for name in model_names:
            col = f"snvq_{name}"
            med[name][idx] = grp[col].median()
            lo[name][idx] = grp[col].quantile(q1)
            hi[name][idx] = grp[col].quantile(q2)

    n_tp = int(tp_mask.sum())
    n_fp = int((~tp_mask).sum())
    hist_tp = np.zeros(n_total_ctx)
    hist_fp = np.zeros(n_total_ctx)
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

        for m in model_descs:
            name = m["name"]
            qual_ax.fill_between(
                x_ext, _extend(lo[name][sl]), _extend(hi[name][sl]), step="mid", color=m["color"], alpha=0.15
            )
            qual_ax.step(
                x_ext,
                _extend(med[name][sl]),
                where="mid",
                color=m["color"],
                lw=1.5,
                linestyle=m["linestyle"],
                label=m["name"],
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
    model_legend_handles = [plt.Line2D([0], [0], color=m["color"], lw=2, linestyle=m["linestyle"]) for m in model_descs]
    fig.legend(
        model_legend_handles,
        [m["name"] for m in model_descs],
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


def _compute_error_analysis(  # noqa: PLR0915, C901
    m_test: pd.DataFrame,
    y_test: np.ndarray,
    model_names: list[str],
    baseline: str,
    model_descs: list[dict],
) -> dict:
    """Run all error analysis sub-sections."""
    ea: dict = {}
    m_test = m_test.copy()
    non_baseline = [n for n in model_names if n != baseline]

    sliced_sections = [
        (
            "sub_type",
            "sub_type_plot",
            "AUC by Substitution Type",
            "Substitution",
            lambda df: df.__setitem__("sub_type", df["REF"].astype(str) + ">" + df["ALT"].astype(str)) or "sub_type",
        ),
        (
            "mixed_st",
            "mixed_st_plot",
            "AUC by ppmSeq Start Tag (st)",
            "Start Tag",
            lambda df: df.__setitem__(
                "st_filled",
                df["st"].astype(str).replace("", "(empty)").replace("nan", "(empty)"),
            )
            or "st_filled",
        ),
        (
            "end_tag",
            "end_tag_plot",
            "AUC by ppmSeq End Tag (et)",
            "End Tag",
            lambda df: df.__setitem__(
                "et_filled",
                df["et"].astype(str).replace("", "(empty)").replace("nan", "(empty)"),
            )
            or "et_filled",
        ),
        (
            "tm_tag",
            "tm_tag_plot",
            "AUC by ppmSeq Combined Tag (tm)",
            "Tag",
            lambda df: df.__setitem__(
                "tm_filled",
                df["tm"].astype(str).replace("", "(empty)").replace("nan", "(empty)"),
            )
            or "tm_filled",
        ),
    ]

    for ea_key, plot_key, title, xlabel, col_fn in sliced_sections:
        col = col_fn(m_test)
        ea[ea_key] = _sliced_auc(m_test, y_test, col, model_names, baseline)
        ea[plot_key] = _make_grouped_bar(ea[ea_key], title, model_names, model_descs, xlabel)

    ea["hmer"] = _sliced_auc(m_test, y_test, "X_HMER_REF", model_names, baseline)
    ea["hmer_plot"] = _make_grouped_bar(
        ea["hmer"], "AUC by Homopolymer Length (X_HMER_REF)", model_names, model_descs, "Homopolymer Length"
    )

    ea["edist"] = _sliced_auc(m_test, y_test, "EDIST", model_names, baseline)
    ea["edist_plot"] = _make_grouped_bar(ea["edist"], "AUC by Edit Distance", model_names, model_descs, "Edit Distance")

    ea["index_bin"] = _sliced_auc_numeric_bins(m_test, y_test, "INDEX", model_names, baseline, n_bins=10)
    ea["index_bin_plot"] = _make_grouped_bar(
        ea["index_bin"], "AUC by Read Position (INDEX)", model_names, model_descs, "Position Bin"
    )

    ea["strand"] = _sliced_auc(m_test, y_test, "REV", model_names, baseline)
    for r in ea["strand"]:
        r["value"] = "Reverse" if r["value"] == "1" else "Forward"
    ea["strand_plot"] = _make_grouped_bar(ea["strand"], "AUC by Strand", model_names, model_descs, "Strand")

    ea["rq_bin"] = _sliced_auc_numeric_bins(m_test, y_test, "rq", model_names, baseline, n_bins=10)
    ea["rq_bin_plot"] = _make_grouped_bar(ea["rq_bin"], "AUC by Read Quality (rq)", model_names, model_descs, "rq Bin")

    ea["trinuc_plot"], ea["trinuc"] = _make_trinuc_heatmap(m_test, y_test, model_names, baseline, model_descs)
    ea["agreement"] = _compute_error_agreement(m_test, y_test, model_names, model_descs)

    # Format numeric values for template display
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
        if key not in ea or not isinstance(ea[key], list):
            continue
        for r in ea[key]:
            for name in model_names:
                for metric_key in ("auc", "aupr"):
                    if metric_key in r and isinstance(r[metric_key], dict) and name in r[metric_key]:
                        v = r[metric_key][name]
                        r[metric_key][name] = _fmt(v) if (v is not None and not isinstance(v, str)) else "N/A"
            for name in non_baseline:
                if "ri_auc" in r and isinstance(r["ri_auc"], dict) and name in r["ri_auc"]:
                    v = r["ri_auc"][name]
                    if not isinstance(v, dict):
                        r["ri_auc"][name] = {
                            "val": _fmt(v, 1) if v is not None else "N/A",
                            "cls": _improvement_class(v) if v is not None else "",
                        }

    return ea


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def render_html(report: dict, output_path: Path) -> None:
    """Render the HTML report from the Jinja2 template."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,  # noqa: S701
    )
    template = env.get_template(REPORT_TEMPLATE)
    html = template.render(**report)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_model_spec(spec: str) -> dict:
    """Parse ``name=...,parquet=...,metadata=...`` into a model descriptor."""
    parts = {}
    for part in spec.split(","):
        key, _, val = part.partition("=")
        parts[key.strip()] = val.strip()
    required = {"name", "parquet", "metadata"}
    missing = required - parts.keys()
    if missing:
        msg = f"Model spec missing {missing}: {spec}"
        raise ValueError(msg)
    result: dict = {"name": parts["name"], "parquet": Path(parts["parquet"]), "metadata": Path(parts["metadata"])}
    if "color" in parts:
        result["color"] = parts["color"]
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate SRSNV multi-model comparison report")
    ap.add_argument(
        "--model", action="append", dest="models_raw", help="Model spec: name=...,parquet=...,metadata=... (repeatable)"
    )
    ap.add_argument("--xgb-parquet", type=Path, help="(Legacy) XGBoost parquet")
    ap.add_argument("--dnn-parquet", type=Path, help="(Legacy) DNN parquet")
    ap.add_argument("--xgb-metadata", type=Path, help="(Legacy) XGBoost metadata JSON")
    ap.add_argument("--dnn-metadata", type=Path, help="(Legacy) DNN metadata JSON")
    ap.add_argument("--output", type=Path, required=True, help="Output HTML path")
    return ap.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.models_raw:
        models = [_parse_model_spec(s) for s in args.models_raw]
    elif args.xgb_parquet and args.dnn_parquet:
        models = [
            {"name": "XGBoost", "parquet": args.xgb_parquet, "metadata": args.xgb_metadata},
            {"name": "DNN", "parquet": args.dnn_parquet, "metadata": args.dnn_metadata},
        ]
    else:
        msg = "Provide --model specs or legacy --xgb-parquet/--dnn-parquet arguments"
        raise ValueError(msg)

    data = load_data(models)
    report = compute_report_data(data)
    render_html(report, args.output)


def main():
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
