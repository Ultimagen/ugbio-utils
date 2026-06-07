# ugbio_seq_qc

Sequencing QC HTML reports for Ultima Genomics runs.

This package generates self-contained, plotly-based HTML QC reports from
**Sorter** statistics output (the per-sample JSON/CSV files produced by the
on-tool Sorter). It lives outside `ugbio_core` because it depends on
`ugbio_cloud_utils` (`cloud_sync`) and `boto3` for fetching inputs from S3 —
dependencies that the dependency-free `ugbio_core` foundation must not carry.

## Reports

- **`seq_qc_report`** — single-sample report. One sample's JSON + CSV → one HTML
  page with a summary table and coverage / read-length / base-quality / MAPQ
  figures.
- **`seq_qc_multi_sample_report`** — multi-sample report. Auto-discovers the real
  samples in a run (via the `LibraryInfo` sample sheet when present, otherwise by
  name pattern), then renders one figure per sample grouped by metric, plus a
  samples-as-rows / metrics-as-columns comparison table with frozen header row and
  first column.

Both accept a local directory or an `s3://` URI.

## Usage

```bash
# Single sample — explicit files or a directory to auto-detect the basename
seq_qc_report --json sample.json --csv sample.csv --output report.html
seq_qc_report --input-dir s3://.../603559-L13064-Z0152-CATGCAACACTAGAT/

# Multi-sample — a whole run directory (local or s3://)
seq_qc_multi_sample_report --run-dir s3://.../603559/ --output run_report.html

# Multi-sample — explicit per-sample directories
seq_qc_multi_sample_report --input-dir s3://.../sampleA/ --input-dir s3://.../sampleB/
```

## Modules

| Module | Role |
|--------|------|
| `seq_qc_report.py` | Single-sample report: plotly figure builders (shared) + HTML assembly + CLI. |
| `seq_qc_multi_sample_report.py` | Multi-sample HTML assembly + comparison table + CLI; reuses the figure builders from `seq_qc_report`. |
| `sample_discovery.py` | Sample discovery: `LibraryInfo` XML parsing, name-pattern filtering, sample loading into `SampleData`. |
| `file_resolution.py` | Shared S3 / local file-resolution helpers (`list_s3_dir`, `list_s3_subdirs`, `resolve_sample_files`, ...). |
