"""Generate an HTML QC report from sequencing (Sorter) statistics JSON and CSV files."""

from __future__ import annotations

import json
import math
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ugbio_core.logger import logger
from ugbio_core.sorter_utils import get_base_coverage_from_sorter, read_sorter_statistics_csv

from ugbio_seq_qc.file_resolution import resolve_sample_files

_PERCENTILE_COLORS = ["red", "orange", "green", "orange", "red"]
_PERCENTILE_LABELS = ["P5", "P25", "P50", "P75", "P95"]
_PERCENTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
_DASH_STYLES = ["solid", "dash", "dot", "dashdot"]

_EXOME_GROUP = ["Exome", "ACMG59", "Exon 1", "Exon >1"]
_UNIQUE_GROUP = ["Unique", "Non-unique"]


def _add_percentile_lines(fig: go.Figure, data: np.ndarray, x_cap: int, *, show_mean: bool = True) -> None:
    """Add vertical percentile and mean lines to a figure."""
    if data.size == 0:
        return
    cumsum = np.cumsum(data)
    total = cumsum[-1]
    if total == 0:
        return

    for pct, color, label in zip(_PERCENTILES, _PERCENTILE_COLORS, _PERCENTILE_LABELS, strict=False):
        pbin = int(np.searchsorted(cumsum, pct * total))
        if pbin <= x_cap:
            fig.add_vline(x=pbin, line={"color": color, "width": 1.5, "dash": "dash"})
            fig.add_annotation(
                x=pbin,
                y=1.02,
                yref="paper",
                text=f"{label}={pbin}",
                showarrow=False,
                font={"size": 12, "color": color},
                yanchor="bottom",
            )

    if show_mean:
        mean_val = np.sum(np.arange(len(data)) * data) / total
        if mean_val <= x_cap:
            fig.add_vline(x=mean_val, line={"color": "black", "width": 2, "dash": "dot"})
            fig.add_annotation(
                x=mean_val,
                y=0.95,
                yref="paper",
                text=f"mean={mean_val:.1f}",
                showarrow=False,
                font={"size": 12, "color": "black"},
                yanchor="bottom",
            )


def _build_summary_table_html(csv_df: pd.DataFrame, basename: str) -> str:
    """Build a styled 2-column HTML summary table."""
    n = len(csv_df)
    ncols = 2
    rows_per_col = math.ceil(n / ncols)

    tables = []
    for c in range(ncols):
        start = c * rows_per_col
        end = min(start + rows_per_col, n)
        chunk = csv_df.iloc[start:end]
        rows_html = ""
        for _, row in chunk.iterrows():
            rows_html += (
                f"<tr>"
                f"<td style='background:#e8f0fe; font-weight:600; padding:8px 12px; font-size:18px;'>"
                f"{row['metric']}</td>"
                f"<td style='padding:8px 12px; font-size:18px;'>{row['value']}</td>"
                f"</tr>"
            )
        tables.append(
            f"<table style='border-collapse:collapse; width:100%;'>"
            f"<thead><tr>"
            f"<th style='background:#1a73e8; color:white; padding:10px 12px; text-align:left; font-size:19px;'>"
            f"Metric</th>"
            f"<th style='background:#1a73e8; color:white; padding:10px 12px; text-align:left; font-size:19px;'>"
            f"Value</th>"
            f"</tr></thead>"
            f"<tbody>{rows_html}</tbody></table>"
        )

    return (
        "<div style='display:grid; grid-template-columns: 1fr 1fr; gap:16px;'>"
        + "".join(f"<div>{t}</div>" for t in tables)
        + "</div>"
        + f"<p style='font-size:15px; color:#666; margin-top:12px;'>Summary table: data from {basename}.csv</p>"
    )


def build_coverage_boxplot(base_coverage: dict, csv_df: pd.DataFrame, basename: str) -> go.Figure:
    """Build the base coverage boxplot normalized by median coverage."""
    median_cvg = float(csv_df.loc[csv_df["metric"] == "median_cvg", "value"].iloc[0])
    if median_cvg == 0:
        fig = go.Figure()
        fig.update_layout(
            title={"text": f"{basename} (no coverage data)", "font": {"size": 24}},
            template="plotly_white",
            height=200,
        )
        return fig
    bc = base_coverage

    gc_keys = sorted(k for k in bc if k.startswith("GC "))
    known = {"Genome"} | set(_EXOME_GROUP) | set(_UNIQUE_GROUP) | set(gc_keys)
    rest = sorted(k for k in bc if k not in known)

    regions = (
        (["Genome"] if "Genome" in bc else [])
        + [k for k in _EXOME_GROUP if k in bc]
        + gc_keys
        + [k for k in _UNIQUE_GROUP if k in bc]
        + rest
    )

    box_data = []
    colors = []
    for region in regions:
        hist = np.array(bc[region], dtype=float)
        total = hist.sum()
        if total == 0:
            continue
        cdf = np.cumsum(hist) / total

        box_data.append(
            {
                "region": region,
                "median": int(np.searchsorted(cdf, 0.5)) / median_cvg,
                "q1": int(np.searchsorted(cdf, 0.25)) / median_cvg,
                "q3": int(np.searchsorted(cdf, 0.75)) / median_cvg,
                "lowerfence": int(np.searchsorted(cdf, 0.05)) / median_cvg,
                "upperfence": int(np.searchsorted(cdf, 0.95)) / median_cvg,
            }
        )
        if region == "Genome":
            colors.append("blue")
        elif region in _EXOME_GROUP:
            colors.append("darkorange")
        elif region.startswith("GC "):
            colors.append("green")
        elif region in _UNIQUE_GROUP:
            colors.append("red")
        else:
            colors.append("purple")

    fig = go.Figure()
    median_annotations = []
    for i, d in enumerate(box_data):
        fig.add_trace(
            go.Box(
                x=[d["region"]],
                median=[d["median"]],
                q1=[d["q1"]],
                q3=[d["q3"]],
                lowerfence=[d["lowerfence"]],
                upperfence=[d["upperfence"]],
                marker_color=colors[i],
                fillcolor=colors[i],
                line_color=colors[i],
                showlegend=False,
                opacity=0.7,
            )
        )
        median_annotations.append(
            {
                "x": d["region"],
                "y": d["median"],
                "text": f"{d['median']:.2f}",
                "showarrow": False,
                "font": {"size": 9, "color": "white"},
                "yanchor": "middle",
            }
        )

    fig.update_layout(
        title={"text": basename, "font": {"size": 24}},
        xaxis_title="",
        yaxis_title="Coverage / Median Coverage",
        xaxis={"categoryorder": "array", "categoryarray": [d["region"] for d in box_data]},
        template="plotly_white",
        height=550,
        xaxis_tickangle=-45,
        yaxis_range=[0, 2.5],
        font={"size": 16},
        annotations=median_annotations
        + [
            {
                "text": (
                    f'Base coverage boxplot: data from JSON "base_coverage" key, '
                    f"each region's distribution normalized by median_cvg={median_cvg:.1f} from CSV. "
                    f"Whiskers at P5/P95."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.45,
                "xanchor": "left",
                "showarrow": False,
                "font": {"size": 13, "color": "gray"},
            }
        ],
        margin={"b": 160},
    )
    return fig


def build_cvg_histogram(stats_json: dict, basename: str) -> go.Figure:
    """Build the coverage histogram capped at 99th percentile."""
    data = np.array(stats_json["cvg"], dtype=float)
    cumsum = np.cumsum(data)
    total = cumsum[-1]
    p99_bin = int(np.searchsorted(cumsum, 0.99 * total))
    bins = np.arange(p99_bin + 1)
    data_capped = data[: p99_bin + 1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bins,
            y=data_capped,
            mode="lines",
            line={"width": 2},
            showlegend=False,
        )
    )

    _add_percentile_lines(fig, data, p99_bin)

    fig.update_layout(
        title={"text": basename, "font": {"size": 24}},
        xaxis_title="Coverage",
        yaxis_title="Count",
        template="plotly_white",
        height=500,
        font={"size": 16},
        annotations=list(fig.layout.annotations)
        + [
            {
                "text": (
                    'Coverage histogram: data from JSON "cvg" key, histogram capped at 99th percentile. '
                    "Vertical lines show P5/P25/P50/P75/P95 and mean."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.32,
                "xanchor": "left",
                "showarrow": False,
                "font": {"size": 13, "color": "gray"},
            }
        ],
        margin={"b": 120},
    )
    return fig


def build_read_length_plot(stats_json: dict, basename: str) -> go.Figure:
    """Build overlaid read length distributions capped at 99th percentile."""
    keys = [k for k in stats_json if "read_length" in k and isinstance(stats_json[k], list)]

    fig = go.Figure()
    plotted_keys = []
    for idx, key in enumerate(keys):
        data = np.array(stats_json[key], dtype=float)
        if data.size == 0:
            continue
        cumsum = np.cumsum(data)
        total = cumsum[-1]
        if total == 0:
            continue
        p99_bin = int(np.searchsorted(cumsum, 0.99 * total))
        bins = np.arange(p99_bin + 1)
        data_capped = data[: p99_bin + 1]

        fig.add_trace(
            go.Scatter(
                x=bins,
                y=data_capped,
                mode="lines",
                line={"width": 2.5, "dash": _DASH_STYLES[idx % len(_DASH_STYLES)]},
                name=key,
            )
        )
        plotted_keys.append(key)

        if len(plotted_keys) == 1:
            _add_percentile_lines(fig, data, p99_bin)

    caption_key = plotted_keys[0] if plotted_keys else "read_length"
    fig.update_layout(
        title={"text": basename, "font": {"size": 24}},
        xaxis_title="Read Length",
        yaxis_title="Count",
        template="plotly_white",
        height=500,
        font={"size": 16},
        annotations=list(fig.layout.annotations)
        + [
            {
                "text": (
                    f'Read length distributions: data from JSON keys containing "read_length", '
                    f'capped at 99th percentile. Percentile lines shown for "{caption_key}".'
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.32,
                "xanchor": "left",
                "showarrow": False,
                "font": {"size": 13, "color": "gray"},
            }
        ],
        margin={"b": 120},
    )
    return fig


def build_bqual_plot(stats_json: dict, basename: str) -> go.Figure:
    """Build overlaid base quality distributions."""
    keys = [k for k in stats_json if "bqual" in k and isinstance(stats_json[k], list)]

    fig = go.Figure()
    plotted_keys = []
    for idx, key in enumerate(keys):
        data = np.array(stats_json[key], dtype=float)
        if data.size == 0:
            continue
        bins = np.arange(len(data))

        fig.add_trace(
            go.Scatter(
                x=bins,
                y=data,
                mode="lines",
                line={"width": 2.5, "dash": _DASH_STYLES[idx % len(_DASH_STYLES)]},
                name=key,
            )
        )
        plotted_keys.append(key)

        if len(plotted_keys) == 1:
            _add_percentile_lines(fig, data, len(data) - 1)

    caption_key = plotted_keys[0] if plotted_keys else "bqual"
    fig.update_layout(
        title={"text": basename, "font": {"size": 24}},
        xaxis_title="Base Quality",
        yaxis_title="Count",
        template="plotly_white",
        height=500,
        font={"size": 16},
        annotations=list(fig.layout.annotations)
        + [
            {
                "text": (
                    f'Base quality distributions: data from JSON keys containing "bqual". '
                    f'Percentile lines shown for "{caption_key}".'
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.32,
                "xanchor": "left",
                "showarrow": False,
                "font": {"size": 13, "color": "gray"},
            }
        ],
        margin={"b": 120},
    )
    return fig


def build_mapq_plot(stats_json: dict, basename: str) -> go.Figure:
    """Build MAPQ bar plot with percentage labels."""
    data = np.array(stats_json["mapq"], dtype=float)
    total = data.sum()
    max_val = int(np.max(np.nonzero(data)[0])) if np.any(data > 0) else len(data) - 1
    bins = np.arange(max_val + 1)
    data_trimmed = data[: max_val + 1]
    pcts = data_trimmed / total * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(x=bins, y=data_trimmed, width=0.8, showlegend=False))

    annotations = []
    for i, pct in enumerate(pcts):
        if pct > 0.1:  # noqa: PLR2004
            annotations.append(
                {
                    "x": bins[i],
                    "y": data_trimmed[i],
                    "text": f"{pct:.1f}%",
                    "showarrow": False,
                    "font": {"size": 10, "color": "black"},
                    "yanchor": "bottom",
                    "yshift": 4,
                    "textangle": -90,
                }
            )

    fig.update_layout(
        title={"text": basename, "font": {"size": 24}},
        xaxis_title="MAPQ",
        yaxis_title="Count",
        template="plotly_white",
        height=500,
        font={"size": 16},
        annotations=annotations
        + [
            {
                "text": (
                    'MAPQ distribution: data from JSON "mapq" key. '
                    "Percentage labels shown for bars >0.1% of total reads."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.32,
                "xanchor": "left",
                "showarrow": False,
                "font": {"size": 13, "color": "gray"},
            }
        ],
        margin={"b": 120},
    )
    return fig


def _assemble_html(basename: str, table_html: str, figures: list[go.Figure]) -> str:
    """Assemble a self-contained HTML report from table and figures."""
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>QC Report - {basename}</title>",
        "</head><body style='font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px;'>",
        "<h1 style='font-size:36px;'>Ultima Genomics Sequencing QC Report</h1>",
        f"<h2 style='font-size:26px; color:#555;'>{basename}</h2>",
        table_html,
    ]

    for i, fig in enumerate(figures):
        include_plotlyjs = "cdn" if i == 0 else False
        parts.append(fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs))

    parts.append("</body></html>")
    return "\n".join(parts)


def generate_seq_qc_report(json_path: Path, csv_path: Path, output_html: Path) -> Path:
    """
    Generate an HTML QC report from sequencing (Sorter) stats files.

    Parameters
    ----------
    json_path : Path
        Path to the sorter statistics JSON file.
    csv_path : Path
        Path to the sorter statistics CSV file.
    output_html : Path
        Path for the output HTML report.

    Returns
    -------
    Path
        Path to the generated HTML report.

    """
    logger.info(f"Reading sequencing stats from {json_path} and {csv_path}")
    csv_df = read_sorter_statistics_csv(str(csv_path), edit_metric_names=False, as_dataframe=True)
    with open(json_path, encoding="utf-8") as f:
        stats_json = json.load(f)
    base_coverage = get_base_coverage_from_sorter(str(json_path))

    basename = json_path.stem

    logger.info("Building report figures")
    table_html = _build_summary_table_html(csv_df, basename)
    figures = [
        build_coverage_boxplot(base_coverage, csv_df, basename),
        build_cvg_histogram(stats_json, basename),
        build_read_length_plot(stats_json, basename),
        build_bqual_plot(stats_json, basename),
        build_mapq_plot(stats_json, basename),
    ]

    html_content = _assemble_html(basename, table_html, figures)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_content, encoding="utf-8")
    logger.info(f"Report written to {output_html}")
    return output_html


def _resolve_paths(args) -> tuple[Path, Path, Path]:
    """Resolve JSON/CSV/output paths from arguments, fetching from S3 if needed."""
    if args.json and args.csv:
        json_path = Path(args.json)
        csv_path = Path(args.csv)
    elif args.input_dir:
        logger.info(f"Resolving sample files from {args.input_dir}")
        json_path, csv_path, _ = resolve_sample_files(args.input_dir)
    else:
        raise ValueError("Either --input-dir or both --json and --csv must be provided")

    output_html = Path(args.output) if args.output else json_path.with_suffix(".html")
    return json_path, csv_path, output_html


def parse_args(argv: list[str]):
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Generate HTML QC report from sequencing stats files")
    parser.add_argument(
        "--input-dir",
        help="Directory (local or s3://) containing the JSON and CSV sorter stats files",
    )
    parser.add_argument("--json", help="Explicit path to the sorter stats JSON file")
    parser.add_argument("--csv", help="Explicit path to the sorter stats CSV file")
    parser.add_argument("--output", help="Output HTML path (default: same basename as input with .html)")
    return parser.parse_args(argv)


def run(argv: list[str]) -> None:
    """Main entry point for the sequencing QC report CLI."""
    args = parse_args(argv[1:])
    json_path, csv_path, output_html = _resolve_paths(args)
    generate_seq_qc_report(json_path, csv_path, output_html)


def main():
    """CLI entry point."""
    run(sys.argv)


if __name__ == "__main__":
    main()
