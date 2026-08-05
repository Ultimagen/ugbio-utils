---
name: mrd-localize
description: Use when the user wants to download an MRDFeatureMap Omics run from S3 to a local directory and create a run_report.sh script for local re-analysis.
---

# MRD Localize Skill

Download all outputs of an MRDFeatureMap Omics run from S3 to a local directory
and generate a ready-to-run `run_report.sh` script.

## When to Use This Skill

Activate this skill when the user provides an S3 path to an MRDFeatureMap run
output folder and asks to "download", "localize", or "set up" a local run.

Example triggers:
- "Download sa31 from s3://…/out to /proj/mrd/sa31"
- "Set up a local run for this S3 output: s3://…/1804422/out"
- "Create a run script similar to sa54 for this run"

## Required Input

- **S3 output folder** — the `…/out` prefix for the run (e.g.
  `s3://ultimagen-pipelines-471112545487-us-east-1-outputs/MRDFeatureMap/mrd-fullcov-v1-29-0/1804422/out`)
- **Local destination directory** — where to place the downloaded files (e.g. `/proj/mrd/sa31`)
- **Number of synthetic controls** — how many db-control VCFs / parquets to include (default: 5)

If the user doesn't provide the local directory or control count, infer them or ask.

## Step-by-Step Procedure

### 1 — Inspect the S3 output folder

Run these in parallel to get file names:

```bash
BASE="<s3_out_folder>"

# Single-file subdirs
for d in coverage_bed coverage_bed_index matched_signature_vcf filter_funnel_json; do
  echo "=== $d ===" && aws s3 ls "${BASE}/${d}/"
done
# Count multi-file subdirs
aws s3 ls "${BASE}/intersected_featuremaps_parquet/" | wc -l
aws s3 ls "${BASE}/db_signatures_vcf/" | wc -l
# Top-level (coverage bed index is sometimes here)
aws s3 ls "${BASE}/" | grep -v PRE
```

### 2 — Find featuremap_df and srsnv_metadata paths

Download the run metadata JSON and extract the input paths:

```bash
RUN_BASE="${BASE%/out}"   # strip /out
aws s3 cp "${RUN_BASE}/logs/metadata.json" /tmp/run_meta.json

python3 -c "
import json
with open('/tmp/run_meta.json') as f: d=json.load(f)
inputs = d['inputs']
if isinstance(inputs, str): inputs = json.loads(inputs)
for k,v in inputs.items():
    if any(x in k.lower() for x in ['featuremap_df','srsnv','metadata']):
        print(k, ':', v)
"
```

The keys to find are `featuremap_df_file` and `srsnv_metadata_json`.
Both live under `s3://…-bioinfo-resources/run-data/…/srsnv/<sample>/`.

### 3 — Determine sample basename

Extract it from the coverage_bed filename (e.g. `sa_31_Z0052_v1290.regions.bed.gz`
→ basename = `sa_31_Z0052`).

### 4 — Create local directory structure

```bash
mkdir -p <dest>/db_signatures_vcf <dest>/intersected_featuremaps_parquet <dest>/out
```

### 5 — Download files

Run single-file downloads in parallel, then the two recursive copies:

```bash
# Single files (parallel)
aws s3 cp "${BASE}/coverage_bed/<bed_file>"             "<dest>/" &
aws s3 cp "${BASE}/coverage_bed_index/<csi_file>"       "<dest>/" &
aws s3 cp "${BASE}/matched_signature_vcf/<vcf_file>"    "<dest>/" &
aws s3 cp "${BASE}/filter_funnel_json/<json_file>"      "<dest>/" &
aws s3 cp "<featuremap_df_s3>"                          "<dest>/" &
aws s3 cp "<srsnv_metadata_s3>"                         "<dest>/" &
wait

# Recursive (parallel)
aws s3 cp "${BASE}/db_signatures_vcf/"                  "<dest>/db_signatures_vcf/"                  --recursive --include "*.vcf.gz" &
aws s3 cp "${BASE}/intersected_featuremaps_parquet/"    "<dest>/intersected_featuremaps_parquet/"     --recursive --include "*.parquet" &
wait
```

### 6 — Create tabix index for matched VCF

```bash
tabix -p vcf "<dest>/<matched_vcf>.vcf.gz"
```

### 7 — Verify counts

```bash
echo "db VCFs:  $(find <dest>/db_signatures_vcf -name '*.vcf.gz' | wc -l)"
echo "parquets: $(find <dest>/intersected_featuremaps_parquet -name '*.parquet' | wc -l)"
ls -lh <dest>/*.parquet <dest>/*.json <dest>/*.bed.gz <dest>/*.vcf.gz
```

Expected: 30 db VCFs, 31 parquets (00 = matched, 01–30 = db_control).

### 8 — Create run_report.sh

Use the template below, substituting actual filenames.
The `head -6` / `head -5` selects matched + N synthetic controls (default N=5).

```bash
#!/usr/bin/env bash
# Run MRD analysis report locally for <sample_basename>
# Omics run: <run_id>  (MRDFeatureMap/<run_tag>)
set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="${WORKDIR}/out"
BASENAME="<sample_basename>"

mkdir -p "${OUTDIR}"

# --- Intersected featuremap parquets: matched (00) + first N db-controls ---
PARQUETS=$(find "${WORKDIR}/intersected_featuremaps_parquet" -name "*.parquet" | sort | head -<N+1> | tr '\n' ' ')

# --- DB-control synthetic signature VCFs (first N only) ---
DB_VCFS=$(find "${WORKDIR}/db_signatures_vcf" -name "*.vcf.gz" | sort | head -<N> | tr '\n' ' ')

uv run --project /proj/src/ugbio-utils python -m ugbio_mrd.generate_mrd_report \
    --intersected-featuremaps ${PARQUETS} \
    --matched-signature-vcf "${WORKDIR}/<matched_vcf>.vcf.gz" \
    --db-control-signatures-vcf ${DB_VCFS} \
    --coverage-bed "${WORKDIR}/<sample_basename>_<run_tag>.regions.bed.gz" \
    --featuremap-file "${WORKDIR}/<sample_basename>.featuremap_df.parquet" \
    --srsnv-metadata-json "${WORKDIR}/<sample_basename>.srsnv_metadata.json" \
    --filter-funnel-json "${WORKDIR}/<sample_basename>_<run_tag>.filter_funnel.json" \
    --output-dir "${OUTDIR}" \
    --output-basename "${BASENAME}" \
    "$@"
```

The `"$@"` at the end allows passing extra CLI flags (e.g.
`--thresh-multi-read-pvalue-low-tf 0.05`) without editing the script.

## Key File Name Patterns

| File | Naming convention |
|------|-------------------|
| Coverage BED | `<sample>_<run_tag>.regions.bed.gz` |
| Coverage index | `<sample>_<run_tag>.regions.bed.gz.csi` |
| Matched VCF | `UCL<NNNN>.hg38.anon.with_vaf.unique_vars.pon_filtered.filtered.exact_alt_filtered.vcf.gz` |
| Filter funnel JSON | `<sample>_<run_tag>.filter_funnel.json` |
| Featuremap DF | `<sample>.featuremap_df.parquet` |
| SRSNV metadata | `<sample>.srsnv_metadata.json` |
| Parquets (subdir 00) | `<sample>.<patient_id>.matched.intersection.parquet` |
| Parquets (subdir 01+) | `<sample>.syn<N>_<patient_id>.db_control.intersection.parquet` |
| DB VCFs | `syn<N>_<patient_id>.hg38.anon...vcf.gz` |

## Notes

- The matched VCF has **no** `.tbi` in the S3 output — always create it locally
  with `tabix -p vcf`.
- `featuremap_df_file` and `srsnv_metadata_json` live in
  `s3://…-bioinfo-resources/…/srsnv/<sample>/` — **not** in the MRDFeatureMap
  output bucket.
- The run_id and run_tag appear in the `logs/metadata.json` and in the S3 key
  path itself (e.g. `mrd-fullcov-v1-29-0/1804422`).
- Existing examples: `/proj/mrd/sa24/`, `/proj/mrd/sa31/`, `/proj/mrd/sa54_Z0119/`
  — refer to their `run_report.sh` for reference.
