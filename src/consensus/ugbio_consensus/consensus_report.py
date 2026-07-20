"""
ReadFuserAlignSort (consensus tool) performance & duplex HTML report.

Generates a self-contained HTML report summarising, for one or more samples of a
ReadFuserAlignSort run, the performance of the consensus tool and its duplex
behaviour. All inputs are **local files** supplied by the user (no S3/DB access).

Per sample the report covers:

* **Sorter QC** - alignment / duplication / coverage metrics parsed from the
  local ``sorter_stats_csv`` (post-consensus).
* **Duplex family metrics** - average MI-family size and covered depth for
  *both-strands duplex* families and for *single-strand duplicate* families,
  measured directly from the consensus reads' ``rs:B:i`` tag
  (``[n_forward, n_reverse]``) - see :mod:`ugbio_consensus.duplex_metrics`.
* **On-target metrics (optional)** - when a ``--targets`` BED is supplied
  (e.g. an exome capture BED), on-target rate and on-target mean coverage from
  the local coverage bedGraph; otherwise genome-wide coverage only.

The report is target-agnostic: the Quotient exome case is just the run where a
targets BED happens to be provided.

CLI
---
Single sample::

    consensus_report \\
        --cram Z0315.cram --sorter-stats-csv Z0315.csv \\
        --sorter-stats-json Z0315.json --bedgraph Z0315_0.bedGraph.gz \\
        --reference Homo_sapiens_assembly38.fasta \\
        --targets exome.bed --output report.html

Multiple samples (one ``--sample`` block per sample)::

    consensus_report \\
        --sample name=Z0315 cram=Z0315.cram sorter_stats_csv=Z0315.csv \\
            sorter_stats_json=Z0315.json bedgraph=Z0315_0.bedGraph.gz \\
        --sample name=Z0316 cram=Z0316.cram sorter_stats_csv=Z0316.csv \\
            sorter_stats_json=Z0316.json bedgraph=Z0316_0.bedGraph.gz \\
        --reference ref.fasta --targets exome.bed --output run_report.html
"""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ugbio_core.logger import logger
from ugbio_core.sorter_utils import read_sorter_statistics_csv

from ugbio_consensus import duplex_metrics
from ugbio_consensus.consensus_log import parse_consensus_log
from ugbio_consensus.on_target import bed_covered_size, compute_coverage_from_bedgraph, sorted_bed

# Sorter-stats metrics surfaced in the QC table (present in RFAS output CSVs).
DEFAULT_QC_METRICS = [
    "PF_Barcode_reads",
    "PCT_PF_Reads_aligned",
    "PCT_SOFTCLIPPED_bases",
    "Mean_Read_Length",
    "Mismatch_Rate",
    "Indel_Rate",
    "PCT_Chimeras",
    "Mean_cvg",
    "PCT_duplicates",
]

# Chromosome scanned for the rs-tag duplex analysis. One whole chromosome gives a
# stable, representative family-size / duplex estimate without reading the whole
# (very large) CRAM. chr20 is a common QC-representative autosome; override with
# --duplex-chrom (e.g. "20" for a b37 reference). When a targets BED is provided,
# the scan is restricted to the parts of this chromosome inside the targets.
DEFAULT_DUPLEX_CHROM = "chr20"

_CATEGORY_LABEL = {
    duplex_metrics.DUPLEX: "Both-strands duplex",
    duplex_metrics.SINGLE_STRAND: "Single-strand duplicate",
    duplex_metrics.SINGLETON: "Singleton / pass-through",
}
_CATEGORY_COLOR = {
    duplex_metrics.DUPLEX: "#2c7fb8",
    duplex_metrics.SINGLE_STRAND: "#f03b20",
    duplex_metrics.SINGLETON: "#bdbdbd",
}


@dataclass
class SampleInputs:
    """Local input files for a single sample.

    Attributes
    ----------
    name : str
        Sample name used as the row/label (e.g. ``Z0315``).
    cram : str
        Local path to the consensus CRAM (indexed; ``.crai`` alongside or via
        ``crai``).
    sorter_stats_csv : str
        Local path to the sorter stats CSV.
    sorter_stats_json : str
        Local path to the sorter stats JSON.
    bedgraph : str | None
        Local path to the MAPQ>=0 coverage bedGraph (needed for coverage /
        on-target metrics). ``None`` to skip coverage.
    consensus_log : str | None
        Local path to the consensus tool stdout log (``consensus_stdout``), for
        consensus-tool performance metrics. ``None`` to skip.
    crai : str | None
        Explicit ``.crai`` path (defaults to ``<cram>.crai``).
    """

    name: str
    cram: str
    sorter_stats_csv: str
    sorter_stats_json: str
    bedgraph: str | None = None
    consensus_log: str | None = None
    crai: str | None = None

    def crai_path(self) -> str | None:
        if self.crai:
            return self.crai
        default = self.cram + ".crai"
        return default if os.path.exists(default) else None


def genome_size_from_sorter_json(sorter_stats_json: str) -> int:
    """Return the callable genome size (bp) from a sorter JSON coverage histogram.

    ``base_coverage["Genome"]`` is a histogram of depth -> #bases; its sum is the
    number of callable bases, used as the denominator for genome-wide mean depth.

    Parameters
    ----------
    sorter_stats_json : str
        Local path to the sorter stats JSON.

    Returns
    -------
    int
        Callable genome size in bp.
    """
    with open(sorter_stats_json, encoding="utf-8") as fh:
        stats = json.load(fh)
    return int(sum(stats["base_coverage"]["Genome"]))


def read_targets_bed(targets_bed: str, tmp_dir: str) -> tuple[str, int]:
    """Prepare a sorted targets BED and return ``(sorted_bed_path, target_size_bp)``.

    Parameters
    ----------
    targets_bed : str
        Path to the targets BED.
    tmp_dir : str
        Scratch directory for the sorted copy.

    Returns
    -------
    tuple[str, int]
        Sorted BED path and merged target size in bp.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    sorted_path = sorted_bed(targets_bed, os.path.join(tmp_dir, "targets_sorted.bed"))
    return sorted_path, bed_covered_size(targets_bed)


def bed_intervals_on_chrom(bed_path: str, chrom: str) -> list[tuple[str, int, int]]:
    """Read the BED intervals that lie on ``chrom`` as ``(chrom, start, end)`` tuples.

    Used to restrict the duplex scan to the parts of a single chromosome that are
    inside the targets BED.

    Parameters
    ----------
    bed_path : str
        Path to the targets BED.
    chrom : str
        Chromosome to keep (e.g. ``chr20``).

    Returns
    -------
    list[tuple[str, int, int]]
        BED intervals on ``chrom`` (may be empty if the chromosome is not targeted).
    """
    intervals = []
    with open(bed_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split("\t")
            if fields[0] == chrom:
                intervals.append((fields[0], int(fields[1]), int(fields[2])))
    return intervals


def analyze_sample(  # noqa: PLR0913
    sample: SampleInputs,
    reference: str,
    *,
    targets_sorted_bed: str | None = None,
    target_size: int | None = None,
    duplex_intervals: list[tuple[str, int | None, int | None]],
    max_duplex_reads: int | None = None,
) -> dict:
    """Compute the full metric set for one sample from its local inputs.

    Parameters
    ----------
    sample : SampleInputs
        Local input paths for the sample.
    reference : str
        Reference FASTA path (to decode the CRAM).
    targets_sorted_bed : str | None, optional
        Sorted targets BED; enables on-target metrics.
    target_size : int | None, optional
        Merged target size in bp (required with ``targets_sorted_bed``).
    duplex_intervals : list[tuple[str, int, int | None]]
        Regions to scan for the rs-tag duplex analysis (an ``end`` of ``None``
        means "to the end of the contig"). Built by :func:`build_metrics_table`:
        the whole duplex chromosome, or its intersection with the targets BED.
    max_duplex_reads : int | None, optional
        Cap on reads scanned for duplex metrics (sampling large targets).

    Returns
    -------
    dict
        Flat metric record for the sample (QC + duplex + coverage).
    """
    logger.info("Analyzing sample %s", sample.name)
    record: dict = {"sample": sample.name}

    # --- Sorter QC metrics (post-consensus) ---
    qc = read_sorter_statistics_csv(sample.sorter_stats_csv)
    for metric in DEFAULT_QC_METRICS:
        record[metric] = qc.get(metric)

    # --- Coverage (genome-wide, and on-target if a targets BED was given) ---
    genome_size = genome_size_from_sorter_json(sample.sorter_stats_json)
    if sample.bedgraph:
        cov = compute_coverage_from_bedgraph(
            sample.bedgraph,
            genome_size,
            targets_bed_sorted=targets_sorted_bed,
            target_size=target_size,
        )
        record["genome_mean_cvg"] = cov.genome_mean_cvg
        record["on_target_rate"] = cov.on_target_rate
        record["target_mean_cvg"] = cov.target_mean_cvg
    else:
        record["genome_mean_cvg"] = None
        record["on_target_rate"] = None
        record["target_mean_cvg"] = None

    # --- Duplex / consensus family metrics from the rs:B:i tag ---
    fam = duplex_metrics.collect_family_metrics_from_rs_tags(
        sample.cram,
        duplex_intervals,
        reference,
        index_path=sample.crai_path(),
        max_reads=max_duplex_reads,
    )
    per_cat = fam["per_category"]
    for category in duplex_metrics.CATEGORIES:
        record[f"{category}_avg_family_size"] = per_cat.loc[category, "avg_family_size"]
        record[f"{category}_coverage"] = per_cat.loc[category, "coverage"]
        record[f"{category}_n_reads"] = per_cat.loc[category, "n_reads"]
    record["duplex_scan_reads"] = fam["n_reads_scanned"]
    record["duplex_scan_bp"] = fam["total_interval_bp"]

    # --- Consensus-tool performance metrics from the stdout log (optional) ---
    if sample.consensus_log:
        for key, value in parse_consensus_log(sample.consensus_log).items():
            record[f"consensus_{key}"] = value

    return record


def build_metrics_table(  # noqa: PLR0913
    samples: list[SampleInputs],
    reference: str,
    *,
    work_dir: str,
    targets_bed: str | None = None,
    duplex_chrom: str = DEFAULT_DUPLEX_CHROM,
    max_duplex_reads: int | None = None,
) -> pd.DataFrame:
    """Analyze every sample and return a tidy per-sample metrics table.

    Parameters
    ----------
    samples : list[SampleInputs]
        Local sample inputs.
    reference : str
        Reference FASTA path.
    work_dir : str
        Scratch directory (for the sorted targets BED).
    targets_bed : str | None, optional
        Optional targets BED enabling on-target metrics. When given, the duplex
        scan is restricted to the parts of ``duplex_chrom`` inside the targets.
    duplex_chrom : str, optional
        Chromosome scanned for the rs-tag duplex analysis (default
        :data:`DEFAULT_DUPLEX_CHROM`). Without a targets BED the whole chromosome
        is scanned; with one, only its targeted intervals.
    max_duplex_reads : int | None, optional
        Cap on reads scanned per sample for duplex metrics.

    Returns
    -------
    pd.DataFrame
        One row per sample, sorted by ``sample``.
    """
    targets_sorted, target_size = (None, None)
    if targets_bed:
        targets_sorted, target_size = read_targets_bed(targets_bed, os.path.join(work_dir, "targets"))
        logger.info("Targets BED: %s (%.1f Mb)", targets_bed, target_size / 1e6)
        # Restrict the duplex scan to the targeted intervals on duplex_chrom.
        duplex_intervals = bed_intervals_on_chrom(targets_bed, duplex_chrom)
        if not duplex_intervals:
            logger.warning("No targets on %s; duplex scan will find no reads", duplex_chrom)
        logger.info("Duplex scan: %d targeted intervals on %s", len(duplex_intervals), duplex_chrom)
    else:
        # Whole chromosome (end=None -> resolved to contig length from the CRAM header).
        duplex_intervals = [(duplex_chrom, 0, None)]
        logger.info("Duplex scan: whole chromosome %s", duplex_chrom)

    records = [
        analyze_sample(
            s,
            reference,
            targets_sorted_bed=targets_sorted,
            target_size=target_size,
            duplex_intervals=duplex_intervals,
            max_duplex_reads=max_duplex_reads,
        )
        for s in samples
    ]
    return pd.DataFrame(records).sort_values("sample").reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.Series:
    """Median across samples for the headline consensus/duplex metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Per-sample table from :func:`build_metrics_table`.

    Returns
    -------
    pd.Series
        Median of the key metrics (family sizes, coverages, on-target rate, dup %).
    """
    cols = [
        "PCT_duplicates",
        "Mean_cvg",
        "on_target_rate",
        "target_mean_cvg",
        f"{duplex_metrics.DUPLEX}_avg_family_size",
        f"{duplex_metrics.SINGLE_STRAND}_avg_family_size",
        f"{duplex_metrics.DUPLEX}_coverage",
        f"{duplex_metrics.SINGLE_STRAND}_coverage",
    ]
    present = [c for c in cols if c in df.columns]
    return df[present].median(numeric_only=True)


# --------------------------------------------------------------------------- #
# HTML report assembly
# --------------------------------------------------------------------------- #
def _bar_by_sample(df: pd.DataFrame, columns: dict[str, str], title: str, ylabel: str) -> go.Figure:
    """Grouped bar chart: one trace per metric column, x = samples."""
    fig = go.Figure()
    for col, label in columns.items():
        if col not in df.columns:
            continue
        color = None
        for cat, ccolor in _CATEGORY_COLOR.items():
            if col.startswith(cat):
                color = ccolor
        fig.add_bar(x=df["sample"], y=df[col], name=label, marker_color=color)
    fig.update_layout(
        title=title,
        yaxis_title=ylabel,
        barmode="group",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.3},
        height=430,
    )
    return fig


def build_family_size_figure(df: pd.DataFrame) -> go.Figure:
    """Average MI-family size per sample: duplex vs single-strand."""
    return _bar_by_sample(
        df,
        {
            f"{duplex_metrics.DUPLEX}_avg_family_size": _CATEGORY_LABEL[duplex_metrics.DUPLEX],
            f"{duplex_metrics.SINGLE_STRAND}_avg_family_size": _CATEGORY_LABEL[duplex_metrics.SINGLE_STRAND],
        },
        "Average MI-family size (from rs:B:i tag)",
        "Average family size (reads)",
    )


def build_duplex_coverage_figure(df: pd.DataFrame) -> go.Figure:
    """Consensus-read coverage per sample: duplex vs single-strand."""
    return _bar_by_sample(
        df,
        {
            f"{duplex_metrics.DUPLEX}_coverage": _CATEGORY_LABEL[duplex_metrics.DUPLEX],
            f"{duplex_metrics.SINGLE_STRAND}_coverage": _CATEGORY_LABEL[duplex_metrics.SINGLE_STRAND],
        },
        "Duplex coverage (consensus-read depth over scanned regions)",
        "Coverage (x)",
    )


def build_coverage_figure(df: pd.DataFrame) -> go.Figure | None:
    """Coverage per sample: genome-wide, on-target, and sorter Mean_cvg."""
    if "genome_mean_cvg" not in df.columns or df["genome_mean_cvg"].isna().all():
        return None
    fig = go.Figure()
    fig.add_bar(x=df["sample"], y=df["genome_mean_cvg"], name="Genome mean cvg", marker_color="#636363")
    if "target_mean_cvg" in df.columns and df["target_mean_cvg"].notna().any():
        fig.add_bar(x=df["sample"], y=df["target_mean_cvg"], name="On-target mean cvg", marker_color="#31a354")
    fig.update_layout(
        title="Coverage per sample",
        yaxis_title="Mean coverage (x)",
        barmode="group",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.3},
        height=430,
    )
    return fig


def build_on_target_figure(df: pd.DataFrame) -> go.Figure | None:
    """On-target rate per sample (only when a targets BED was used)."""
    if "on_target_rate" not in df.columns or df["on_target_rate"].isna().all():
        return None
    fig = go.Figure()
    fig.add_bar(
        x=df["sample"],
        y=df["on_target_rate"] * 100,
        marker_color="#31a354",
        text=[f"{v:.1f}%" if pd.notna(v) else "" for v in df["on_target_rate"] * 100],
        textposition="outside",
    )
    fig.update_layout(
        title="On-target rate per sample",
        yaxis_title="On-target rate (%)",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=430,
    )
    return fig


def _consensus_metrics_table(df: pd.DataFrame) -> pd.DataFrame | None:
    """Extract the ``consensus_*`` columns into a samples-as-rows table.

    Returns ``None`` when no consensus stdout log was provided for any sample.
    """
    cols = [c for c in df.columns if c.startswith("consensus_")]
    if not cols:
        return None
    out = df[["sample", *cols]].copy()
    out.columns = ["sample", *[c.removeprefix("consensus_") for c in cols]]
    return out


def build_consensus_performance_figure(df: pd.DataFrame) -> go.Figure | None:
    """Consensus tool performance rates per sample (from the stdout log).

    Shows the fraction of records left unhandled and the fraction that were
    duplicate-set members, i.e. how much of the input the consensus step
    collapsed vs passed through.
    """
    rate_cols = {
        "consensus_PCT_dup_set_members": "Dup-set members (%)",
        "consensus_PCT_unhandled": "Unhandled (%)",
    }
    present = {c: label for c, label in rate_cols.items() if c in df.columns and df[c].notna().any()}
    if not present:
        return None
    fig = go.Figure()
    for col, label in present.items():
        fig.add_bar(x=df["sample"], y=df[col], name=label)
    fig.update_layout(
        title="Consensus tool performance (from stdout log)",
        yaxis_title="% of records",
        barmode="group",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.3},
        height=430,
    )
    return fig


_SCIENTIFIC_NOTATION_THRESHOLD = 1e6


def _fmt(value: object) -> str:
    """Format a scalar for the HTML table."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, float):
        if abs(value) >= _SCIENTIFIC_NOTATION_THRESHOLD:
            return f"{value:.3e}"
        if 0 < abs(value) < 1:
            return f"{value:.3f}"
        return f"{value:.2f}"
    return str(value)


def _build_table_html(df: pd.DataFrame, title: str) -> str:
    """Render a DataFrame as a styled, vertical HTML table.

    Metric names run down the first column and each sample is its own value
    column (the frame is transposed on its ``sample`` column). A frame without a
    ``sample`` column (e.g. the summary metric/value table) is already vertical
    and is rendered as-is.
    """
    if "sample" in df.columns:
        table = df.set_index("sample").T
        table.index.name = "metric"
        col_headers = list(table.columns)
        row_labels = list(table.index)
        value_rows = [list(table.loc[label]) for label in row_labels]
        header = "<th style='padding:6px 10px;text-align:left'>metric</th>" + "".join(
            f"<th style='padding:6px 10px;text-align:left'>{c}</th>" for c in col_headers
        )
    else:
        # Already-vertical frame (e.g. metric / median): first column is the label.
        row_labels = list(df.iloc[:, 0])
        value_rows = [list(row) for row in df.iloc[:, 1:].to_numpy()]
        header = "".join(f"<th style='padding:6px 10px;text-align:left'>{c}</th>" for c in df.columns)

    rows = ""
    for label, values in zip(row_labels, value_rows, strict=True):
        cells = "".join(f"<td style='padding:6px 10px;border-top:1px solid #eee'>{_fmt(v)}</td>" for v in values)
        rows += (
            f"<tr><th style='padding:6px 10px;text-align:left;border-top:1px solid #eee;"
            f"background:#fafafa'>{label}</th>{cells}</tr>"
        )
    return (
        f"<h2 style='font-size:22px;color:#000000'>{title}</h2>"
        "<table style='border-collapse:collapse;font-size:13px;margin-bottom:24px'>"
        f"<thead style='background:#f3f3f3'><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"
    )


def _assemble_html(title: str, tables_html: list[str], figures: list[go.Figure]) -> str:
    """Assemble a self-contained HTML report from tables and figures."""
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "</head><body style='font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; "
        "padding: 20px; background:#ffffff; color:#000000;'>",
        f"<h1 style='font-size:34px;'>{title}</h1>",
        "<p style='color:#000000'>Consensus tool (ReadFuserAlignSort) performance and duplex metrics. "
        "Family size and duplex classification are derived from the per-read <code>rs:B:i</code> tag "
        "(<code>[n_forward, n_reverse]</code>).</p>",
    ]
    parts.extend(tables_html)
    for i, fig in enumerate(figures):
        include_plotlyjs = "cdn" if i == 0 else False
        parts.append(fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs))
    parts.append("</body></html>")
    return "\n".join(parts)


def generate_report(df: pd.DataFrame, output_html: str, *, title: str = "Consensus Tool Report") -> str:
    """Build the HTML report from a per-sample metrics table.

    Parameters
    ----------
    df : pd.DataFrame
        Per-sample table from :func:`build_metrics_table`.
    output_html : str
        Output HTML path.
    title : str, optional
        Report title.

    Returns
    -------
    str
        Path to the written HTML report.
    """
    summary = summarize(df).rename("median (all samples)").to_frame().reset_index()
    summary.columns = ["metric", "median (all samples)"]

    # consensus_* metrics get their own table below; drop them from the per-sample
    # table so they are not shown twice.
    per_sample_df = df[[c for c in df.columns if not c.startswith("consensus_")]]
    tables = [
        _build_table_html(summary, "Summary (median across samples)"),
        _build_table_html(per_sample_df, "Per-sample metrics"),
    ]
    consensus_table = _consensus_metrics_table(df)
    if consensus_table is not None:
        tables.append(_build_table_html(consensus_table, "Consensus tool performance (from stdout log)"))
    figures = [
        f
        for f in (
            build_family_size_figure(df),
            build_duplex_coverage_figure(df),
            build_coverage_figure(df),
            build_on_target_figure(df),
            build_consensus_performance_figure(df),
        )
        if f is not None
    ]
    html = _assemble_html(title, tables, figures)
    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    Path(output_html).write_text(html, encoding="utf-8")
    logger.info("Report written to %s", output_html)
    return output_html


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_sample_block(tokens: list[str]) -> SampleInputs:
    """Parse a ``--sample key=value ...`` block into a :class:`SampleInputs`."""
    kv = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"--sample expects key=value tokens, got '{token}'")
        key, value = token.split("=", 1)
        kv[key.strip()] = value.strip()
    missing = {"name", "cram", "sorter_stats_csv", "sorter_stats_json"} - kv.keys()
    if missing:
        raise ValueError(f"--sample block missing required keys: {sorted(missing)}")
    return SampleInputs(
        name=kv["name"],
        cram=kv["cram"],
        sorter_stats_csv=kv["sorter_stats_csv"],
        sorter_stats_json=kv["sorter_stats_json"],
        bedgraph=kv.get("bedgraph"),
        consensus_log=kv.get("consensus_log"),
        crai=kv.get("crai"),
    )


def _samples_from_args(args: Namespace) -> list[SampleInputs]:
    """Build the sample list from either the single-sample flags or --sample blocks."""
    if args.sample:
        return [_parse_sample_block(block) for block in args.sample]
    if not (args.cram and args.sorter_stats_csv and args.sorter_stats_json):
        raise SystemExit(
            "Provide either --sample blocks, or --cram/--sorter-stats-csv/--sorter-stats-json for one sample."
        )
    name = args.name or Path(args.cram).stem
    return [
        SampleInputs(
            name=name,
            cram=args.cram,
            sorter_stats_csv=args.sorter_stats_csv,
            sorter_stats_json=args.sorter_stats_json,
            bedgraph=args.bedgraph,
            consensus_log=args.consensus_log,
            crai=args.crai,
        )
    ]


def parse_args(argv: list[str]) -> Namespace:
    ap = ArgumentParser(description="ReadFuserAlignSort consensus performance & duplex HTML report (local inputs)")
    ap.add_argument("--reference", required=True, help="Reference FASTA (to decode the CRAM)")
    ap.add_argument("--output", required=True, help="Output HTML report path")
    ap.add_argument("--targets", help="Optional targets BED for on-target metrics (e.g. exome)")
    ap.add_argument(
        "--duplex-chrom",
        dest="duplex_chrom",
        default=DEFAULT_DUPLEX_CHROM,
        help=(
            f"Chromosome scanned for duplex/MI-family metrics (default {DEFAULT_DUPLEX_CHROM}; "
            "e.g. '20' for b37). With --targets, restricted to its targeted intervals on this chromosome."
        ),
    )
    ap.add_argument("--max-duplex-reads", type=int, help="Cap reads scanned per sample for duplex metrics")
    ap.add_argument("--title", default="Consensus Tool Report", help="Report title")
    ap.add_argument("--work-dir", help="Scratch dir (default: alongside --output)")

    # Single-sample flags
    ap.add_argument("--name", help="Sample name (single-sample mode; default: CRAM stem)")
    ap.add_argument("--cram", help="Consensus CRAM (single-sample mode)")
    ap.add_argument("--sorter-stats-csv", dest="sorter_stats_csv", help="Sorter stats CSV (single-sample mode)")
    ap.add_argument("--sorter-stats-json", dest="sorter_stats_json", help="Sorter stats JSON (single-sample mode)")
    ap.add_argument("--bedgraph", help="Coverage bedGraph, MAPQ>=0 (single-sample mode)")
    ap.add_argument("--consensus-log", dest="consensus_log", help="Consensus tool stdout log (single-sample mode)")
    ap.add_argument("--crai", help="Explicit .crai index (single-sample mode)")

    # Multi-sample: repeatable key=value block
    ap.add_argument(
        "--sample",
        nargs="+",
        action="append",
        metavar="key=value",
        help=(
            "Repeatable sample block: name= cram= sorter_stats_csv= sorter_stats_json= "
            "[bedgraph=] [consensus_log=] [crai=]"
        ),
    )
    return ap.parse_args(argv)


def run(argv: list[str]) -> pd.DataFrame:
    args = parse_args(argv[1:])
    samples = _samples_from_args(args)
    logger.info("Analyzing %d sample(s)", len(samples))

    work_dir = args.work_dir or os.path.join(os.path.dirname(os.path.abspath(args.output)), "consensus_report_work")
    os.makedirs(work_dir, exist_ok=True)

    metrics = build_metrics_table(
        samples,
        args.reference,
        work_dir=work_dir,
        targets_bed=args.targets,
        duplex_chrom=args.duplex_chrom,
        max_duplex_reads=args.max_duplex_reads,
    )

    # Write the per-sample table as a sidecar CSV, plus the manifest, for provenance.
    csv_path = os.path.splitext(args.output)[0] + "_per_sample.csv"
    metrics.to_csv(csv_path, index=False)
    pd.DataFrame([asdict(s) for s in samples]).to_csv(os.path.splitext(args.output)[0] + "_manifest.csv", index=False)

    generate_report(metrics, args.output, title=args.title)
    logger.info("Summary (median across %d samples):\n%s", len(metrics), summarize(metrics).to_string())
    return metrics


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
