"""Generate an HTML QC report comparing sorter statistics across multiple samples."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from ugbio_core.logger import logger
from ugbio_core.sorter_stats_report import (
    _build_bqual_plot,
    _build_coverage_boxplot,
    _build_cvg_histogram,
    _build_mapq_plot,
    _build_read_length_plot,
    _find_sorter_basename,
    _list_s3_dir,
)
from ugbio_core.sorter_utils import get_base_coverage_from_sorter, read_sorter_statistics_csv


@dataclass
class SampleData:
    label: str
    stats_json: dict
    csv_df: pd.DataFrame
    base_coverage: dict


def _derive_sample_label(basename: str) -> str:
    parts = basename.split("-", 1)
    if len(parts) > 1:
        return parts[1]
    return basename


def _parse_library_info_xml(xml_path: Path) -> set[str]:
    """Parse LibraryInfo XML and return the set of expected sample folder suffixes.

    Each sample maps to a folder like: RUNID-SAMPLENAME-INDEX_LABEL-INDEX_SEQUENCE
    Returns the set of "SAMPLENAME-INDEX_LABEL-INDEX_SEQUENCE" strings.
    """
    tree = ET.parse(xml_path)  # noqa: S314
    root = tree.getroot()
    suffixes = set()
    for sample_elem in root.iter("Sample"):
        sample_id = sample_elem.get("Id", "")
        name = sample_id.split("@")[0] if "@" in sample_id else sample_id
        index_label = sample_elem.get("Index_Label", "")
        index_seq = sample_elem.get("Index_Sequence", "")
        if name and index_label and index_seq:
            suffixes.add(f"{name}-{index_label}-{index_seq}")
    return suffixes


_MIN_NAMED_SEGMENTS = 3
_LARGE_VALUE_THRESHOLD = 100


def _has_named_sample(dirname: str) -> bool:
    """Check if a directory name has a sample name (RUNID-NAME-BARCODE-SEQ pattern).

    Junk folders follow RUNID-BARCODE-BARCODE_SEQ (3 parts after split on first dash).
    Real samples follow RUNID-SAMPLENAME-BARCODE-BARCODE_SEQ (4+ parts).
    """
    parts = dirname.split("-", 1)
    if len(parts) < 2:  # noqa: PLR2004
        return False
    after_runid = parts[1]
    segments = after_runid.split("-")
    return len(segments) >= _MIN_NAMED_SEGMENTS


def _discover_samples_local(run_dir: Path) -> list[Path]:
    xml_files = list(run_dir.glob("*_LibraryInfo.xml"))
    allowed_suffixes: set[str] | None = None
    if xml_files:
        logger.info(f"Using sample sheet: {xml_files[0].name}")
        allowed_suffixes = _parse_library_info_xml(xml_files[0])

    sample_dirs = []
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir():
            continue
        dirname = d.name
        if allowed_suffixes is not None:
            parts = dirname.split("-", 1)
            suffix = parts[1] if len(parts) > 1 else ""
            if suffix not in allowed_suffixes:
                continue
        elif not _has_named_sample(dirname):
            continue

        files = [f.name for f in d.iterdir() if f.is_file()]
        json_files = [f for f in files if f.endswith(".json") and "ppmSeq" not in f and "_unmatched" not in f]
        csv_files = {f.removesuffix(".csv") for f in files if f.endswith(".csv")}
        if any(f.removesuffix(".json") in csv_files for f in json_files):
            sample_dirs.append(d)
    return sample_dirs


def _find_s3_library_info_xml(s3_uri: str, all_files: list[str]) -> set[str] | None:
    """Look for a LibraryInfo XML in the S3 file listing and parse it if found."""
    xml_files = [f for f in all_files if f.endswith("_LibraryInfo.xml")]
    if not xml_files:
        return None

    try:
        from ugbio_cloud_utils.cloud_sync import cloud_sync  # noqa: PLC0415
    except ImportError:
        return None

    xml_name = xml_files[0]
    logger.info(f"Using sample sheet: {xml_name}")
    local_path = Path(cloud_sync(f"{s3_uri.rstrip('/')}/{xml_name}"))
    return _parse_library_info_xml(local_path)


def _filter_subdirs(subdirs: list[tuple[str, str]], allowed_suffixes: set[str] | None) -> list[str]:
    filtered = []
    for dirname, full_uri in subdirs:
        if allowed_suffixes is not None:
            dir_parts = dirname.split("-", 1)
            suffix = dir_parts[1] if len(dir_parts) > 1 else ""
            if suffix not in allowed_suffixes:
                continue
        elif not _has_named_sample(dirname):
            continue
        filtered.append(full_uri)
    return filtered


def _list_s3_subdirs(s3_uri: str) -> list[str]:
    try:
        import boto3  # noqa: PLC0415
    except ImportError:
        raise RuntimeError("No S3 access from this environment") from None  # noqa: B904

    uri = s3_uri.rstrip("/")
    parts = uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = (parts[1] + "/") if len(parts) > 1 else ""

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    all_files: list[str] = []
    subdirs: list[tuple[str, str]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key[len(prefix) :]
            if filename:
                all_files.append(filename)
        for cp in page.get("CommonPrefixes", []):
            subdir_key = cp["Prefix"]
            subdir_name = subdir_key[len(prefix) :].rstrip("/")
            if subdir_name:
                subdirs.append((subdir_name, f"s3://{bucket}/{subdir_key}"))

    allowed_suffixes = _find_s3_library_info_xml(s3_uri, all_files)

    return _filter_subdirs(subdirs, allowed_suffixes)


def _load_sample(sample_dir: str | Path) -> SampleData | None:
    sample_dir_str = str(sample_dir)
    if sample_dir_str.startswith("s3://"):
        try:
            from ugbio_cloud_utils.cloud_sync import cloud_sync  # noqa: PLC0415
        except ImportError:
            raise RuntimeError("No S3 access from this environment") from None  # noqa: B904

        all_files = _list_s3_dir(sample_dir_str)
        try:
            basename = _find_sorter_basename(all_files)
        except ValueError:
            return None
        json_name = f"{basename}.json"
        csv_name = f"{basename}.csv"
        if json_name not in all_files or csv_name not in all_files:
            return None
        json_path = Path(cloud_sync(f"{sample_dir_str.rstrip('/')}/{json_name}"))
        csv_path = Path(cloud_sync(f"{sample_dir_str.rstrip('/')}/{csv_name}"))
    else:
        dir_path = Path(sample_dir_str)
        all_files = [f.name for f in dir_path.iterdir() if f.is_file()]
        try:
            basename = _find_sorter_basename(all_files)
        except ValueError:
            return None
        json_path = dir_path / f"{basename}.json"
        csv_path = dir_path / f"{basename}.csv"
        if not json_path.exists() or not csv_path.exists():
            return None

    with open(json_path, encoding="utf-8") as f:
        stats_json = json.load(f)
    csv_series = read_sorter_statistics_csv(str(csv_path), edit_metric_names=False)
    csv_df = csv_series.reset_index()
    csv_df.columns = ["metric", "value"]
    base_coverage = get_base_coverage_from_sorter(str(json_path))
    label = _derive_sample_label(basename)

    return SampleData(label=label, stats_json=stats_json, csv_df=csv_df, base_coverage=base_coverage)


def _resolve_and_load_samples(args) -> list[SampleData]:
    sample_dirs: list[str] = []

    if args.run_dir:
        run_dir = args.run_dir
        if run_dir.startswith("s3://"):
            sample_dirs.extend(_list_s3_subdirs(run_dir))
        else:
            sample_dirs.extend(str(d) for d in _discover_samples_local(Path(run_dir)))

    if args.input_dir:
        sample_dirs.extend(args.input_dir)

    if not sample_dirs:
        raise ValueError("No sample directories found. Provide --run-dir or --input-dir.")

    samples = []
    for sd in sample_dirs:
        logger.info(f"Loading sample from {sd}")
        sample = _load_sample(sd)
        if sample is not None:
            samples.append(sample)
        else:
            logger.warning(f"Skipping {sd}: no valid JSON+CSV pair found")

    if not samples:
        raise ValueError("No valid samples found in any of the provided directories.")

    return samples


def _build_multi_summary_table_html(samples: list[SampleData]) -> str:
    all_metrics = samples[0].csv_df["metric"].tolist()

    header_cells = (
        "<th style='background:#1a73e8; color:white; padding:10px 12px; "
        "text-align:left; font-size:18px;'>Metric</th>"
    )
    for s in samples:
        header_cells += (
            f"<th style='background:#1a73e8; color:white; padding:10px 12px; "
            f"text-align:right; font-size:18px;'>{s.label}</th>"
        )

    body_rows = ""
    for metric in all_metrics:
        cells = f"<td style='background:#e8f0fe; font-weight:600; padding:8px 12px; " f"font-size:18px;'>{metric}</td>"
        for s in samples:
            row = s.csv_df.loc[s.csv_df["metric"] == metric]
            val = row["value"].iloc[0] if len(row) > 0 else "N/A"
            if isinstance(val, float):
                if val == int(val) and abs(val) > _LARGE_VALUE_THRESHOLD:
                    val = f"{int(val):,}"
                else:
                    val = f"{val:g}"
            cells += f"<td style='padding:8px 12px; text-align:right; font-size:18px;'>{val}</td>"
        body_rows += f"<tr>{cells}</tr>"

    return (
        "<div style='overflow-x:auto; margin:20px 0;'>"
        f"<table style='border-collapse:collapse; width:100%;'>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{body_rows}</tbody></table></div>"
    )


_SECTION_NAMES = [
    "Coverage Boxplot",
    "Coverage Histogram",
    "Read Length",
    "Base Quality",
    "MAPQ",
]

_SECTION_IDS = [
    "coverage-boxplot",
    "coverage-histogram",
    "read-length",
    "base-quality",
    "mapq",
]


def _assemble_multi_html(title: str, table_html: str, figure_groups: list[list[str]]) -> str:
    toc_items = '<li><a href="#summary-table">Summary Table</a></li>'
    for name, sid in zip(_SECTION_NAMES, _SECTION_IDS, strict=False):
        toc_items += f'<li><a href="#{sid}">{name}</a></li>'

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Multi-Sample QC Report - {title}</title>",
        "</head><body style='font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px;'>",
        "<h1 style='font-size:36px;'>Ultima Genomics Multi-Sample QC Report</h1>",
        f"<h2 style='font-size:26px; color:#555;'>{title}</h2>",
        "<nav style='margin:20px 0; padding:16px; background:#f8f9fa; border-radius:8px;'>",
        "<h2 style='margin:0 0 12px 0; font-size:28px;'>Contents</h2>",
        f"<ol style='margin:0; padding-left:24px; font-size:20px; line-height:1.8;'>{toc_items}</ol>",
        "</nav>",
        '<h2 id="summary-table" style="font-size:30px; margin-top:40px;">Summary Table</h2>',
        table_html,
    ]

    first_figure = True
    for group, name, sid in zip(figure_groups, _SECTION_NAMES, _SECTION_IDS, strict=False):
        parts.append(f'<h2 id="{sid}" style="font-size:30px; margin-top:40px;">{name}</h2>')
        for fig_html in group:
            if first_figure:
                parts.append(fig_html)
                first_figure = False
            else:
                parts.append(fig_html)

    parts.append("</body></html>")
    return "\n".join(parts)


def generate_multi_sample_report(samples: list[SampleData], output_html: Path, title: str) -> Path:
    """Generate a multi-sample HTML QC report."""
    logger.info(f"Generating multi-sample report for {len(samples)} samples")

    table_html = _build_multi_summary_table_html(samples)

    figure_groups: list[list[str]] = []
    first_figure = True

    for build_fn, needs_bc in [
        (_build_coverage_boxplot, True),
        (_build_cvg_histogram, False),
        (_build_read_length_plot, False),
        (_build_bqual_plot, False),
        (_build_mapq_plot, False),
    ]:
        group = []
        for s in samples:
            if needs_bc:
                fig = build_fn(s.base_coverage, s.csv_df, s.label)
            else:
                fig = build_fn(s.stats_json, s.label)
            include_plotlyjs = "cdn" if first_figure else False
            group.append(fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs))
            first_figure = False
        figure_groups.append(group)

    html_content = _assemble_multi_html(title, table_html, figure_groups)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_content, encoding="utf-8")
    logger.info(f"Report written to {output_html}")
    return output_html


def parse_args(argv: list[str]):
    parser = ArgumentParser(description="Generate multi-sample HTML QC report from sorter stats")
    parser.add_argument(
        "--run-dir",
        help="Run directory (local or s3://) to auto-discover sample subdirectories",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        help="Explicit sample directory (repeatable). Local path or s3:// URI.",
    )
    parser.add_argument("--output", help="Output HTML path (default: multi_sample_report.html in run-dir or cwd)")
    parser.add_argument("--title", help="Report title (default: derived from run directory name)")
    return parser.parse_args(argv)


def run(argv: list[str]) -> None:
    args = parse_args(argv[1:])

    if not args.run_dir and not args.input_dir:
        raise ValueError("At least one of --run-dir or --input-dir must be provided.")

    samples = _resolve_and_load_samples(args)

    if args.title:
        title = args.title
    elif args.run_dir:
        run_name = Path(args.run_dir.rstrip("/")).name
        parts = run_name.split("-", 1)
        title = f"Run {parts[0]}" if parts[0].isdigit() else run_name
    else:
        title = "Multi-Sample Report"

    if args.output:
        output_html = Path(args.output)
    elif args.run_dir and not args.run_dir.startswith("s3://"):
        output_html = Path(args.run_dir) / "multi_sample_report.html"
    else:
        output_html = Path("multi_sample_report.html")

    generate_multi_sample_report(samples, output_html, title)


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
