"""Generate an HTML QC report comparing sorter statistics across multiple samples."""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

from ugbio_core.logger import logger
from ugbio_core.sorter_sample_discovery import SampleData, resolve_and_load_samples
from ugbio_core.sorter_stats_report import (
    _build_bqual_plot,
    _build_coverage_boxplot,
    _build_cvg_histogram,
    _build_mapq_plot,
    _build_read_length_plot,
)

_LARGE_VALUE_THRESHOLD = 100

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


def _build_multi_summary_table_html(samples: list[SampleData]) -> str:
    all_metrics = samples[0].csv_df["metric"].tolist()

    th_style = (
        "background:#1a73e8; color:white; padding:10px 12px; "
        "text-align:right; font-size:16px; position:sticky; top:0; z-index:1;"
    )
    th_corner_style = (
        "background:#1a73e8; color:white; padding:10px 12px; "
        "text-align:left; font-size:16px; position:sticky; top:0; left:0; z-index:2;"
    )
    td_sticky_style = (
        "background:#e8f0fe; font-weight:600; padding:8px 12px; "
        "font-size:16px; position:sticky; left:0; z-index:1; white-space:nowrap;"
    )

    header_cells = f"<th style='{th_corner_style}'>Sample</th>"
    for metric in all_metrics:
        header_cells += f"<th style='{th_style}'>{metric}</th>"

    body_rows = ""
    for s in samples:
        cells = f"<td style='{td_sticky_style}'>{s.label}</td>"
        for metric in all_metrics:
            row = s.csv_df.loc[s.csv_df["metric"] == metric]
            val = row["value"].iloc[0] if len(row) > 0 else "N/A"
            if isinstance(val, float):
                if val == int(val) and abs(val) > _LARGE_VALUE_THRESHOLD:
                    val = f"{int(val):,}"
                else:
                    val = f"{val:g}"
            cells += f"<td style='padding:8px 12px; text-align:right; font-size:16px;'>{val}</td>"
        body_rows += f"<tr>{cells}</tr>"

    return (
        "<div style='overflow:auto; margin:20px 0; max-height:600px;'>"
        "<table style='border-collapse:collapse;'>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{body_rows}</tbody></table></div>"
    )


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

    for group, name, sid in zip(figure_groups, _SECTION_NAMES, _SECTION_IDS, strict=False):
        parts.append(f'<h2 id="{sid}" style="font-size:30px; margin-top:40px;">{name}</h2>')
        parts.extend(group)

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

    samples = resolve_and_load_samples(args.run_dir, args.input_dir)

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
