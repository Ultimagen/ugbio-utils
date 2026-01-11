# ugbio_comparison

This module includes comparison python scripts and utilities for bioinformatics pipelines. It provides tools for comparing VCF callsets against ground truth datasets, with support for both small variant and structural variant (SV) comparisons.

## JIRA Issue ID
BIOIN-2028

## Overview

The comparison module provides two main CLI scripts for variant comparison:

1. **run_comparison_pipeline** - Compare small variant callsets to ground truth using VCFEVAL
2. **sv_comparison_pipeline** - Compare structural variant (SV) callsets using Truvari

## Installation

To install the comparison module with all dependencies:

```bash
uv sync --package ugbio_comparison
```

## CLI Scripts

### 1. run_comparison_pipeline

Compare VCF callsets to ground truth using VCFEVAL as the comparison engine. This pipeline supports parallel processing by chromosome, annotation with various genomic features, and detailed concordance analysis.

#### Purpose

- Compare variant calls against a ground truth dataset
- Generate concordance metrics (TP, FP, FN)
- Annotate variants with coverage, mappability, and other genomic features
- Support for both single-interval and whole-genome comparisons
- Reinterpret variants based on sequence context

#### Usage

```bash
run_comparison_pipeline \
  --n_parts <number_of_parts> \
  --input_prefix <input_vcf_prefix> \
  --output_file <output_h5_file> \
  --output_interval <output_bed_file> \
  --gtr_vcf <ground_truth_vcf> \
  --highconf_intervals <high_confidence_bed> \
  --reference <reference_fasta> \
  --call_sample_name <sample_name> \
  --truth_sample_name <truth_sample_name>
```

#### Key Parameters

- `--n_parts`: Number of parts the VCF is split into (use 0 for a single complete VCF)
- `--input_prefix`: Prefix of the input VCF file(s)
- `--output_file`: Output HDF5 file containing concordance results
- `--output_interval`: Output BED file of intersected intervals
- `--gtr_vcf`: Ground truth VCF file for comparison
- `--cmp_intervals`: Optional regions for comparison (BED/interval_list)
- `--highconf_intervals`: High confidence intervals (BED/interval_list)
- `--reference`: Reference genome FASTA file
- `--reference_dict`: Reference genome dictionary file
- `--call_sample_name`: Name of the call sample
- `--truth_sample_name`: Name of the truth sample

#### Optional Parameters

- `--coverage_bw_high_quality`: BigWig file with high MAPQ coverage
- `--coverage_bw_all_quality`: BigWig file with all MAPQ coverage
- `--annotate_intervals`: Interval files for annotation (can be specified multiple times)
- `--runs_intervals`: Homopolymer runs intervals
- `--hpol_filter_length_dist`: Length and distance to hmer run to mark (default: 10 10)
- `--ignore_filter_status`: Ignore variant filter status
- `--disable_reinterpretation`: Disable variant reinterpretation
- `--scoring_field`: Alternative scoring field to use (copied to TREE_SCORE)
- `--flow_order`: Sequencing flow order (4 cycle)
- `--n_jobs`: Number of parallel jobs for chromosome processing (default: -1 for all CPUs)
- `--use_tmpdir`: Store temporary files in temporary directory
- `--verbosity`: Logging level (ERROR, WARNING, INFO, DEBUG)

#### Example

```bash
run_comparison_pipeline \
  --n_parts 0 \
  --input_prefix /data/sample.filtered \
  --output_file /results/sample.comp.h5 \
  --output_interval /results/sample.comp.bed \
  --gtr_vcf /reference/HG004_truth.vcf.gz \
  --highconf_intervals /reference/HG004_highconf.bed \
  --reference /reference/Homo_sapiens_assembly38.fasta \
  --call_sample_name SAMPLE-001 \
  --truth_sample_name HG004 \
  --n_jobs 8 \
  --ignore_filter_status \
  --verbosity INFO
```

### 2. sv_comparison_pipeline

Compare structural variant (SV) callsets using Truvari for benchmarking. This pipeline collapses VCF files, runs Truvari bench, and generates concordance dataframes.

#### Purpose

- Compare SV calls against a ground truth dataset using Truvari
- Collapse overlapping variants before comparison
- Generate detailed concordance metrics for SVs
- Support for different SV types (DEL, INS, DUP, etc.)
- Output results in HDF5 format with base and calls concordance

#### Usage

```bash
sv_comparison_pipeline \
  --calls <input_calls_vcf> \
  --gt <ground_truth_vcf> \
  --output_filename <output_h5_file> \
  --outdir <truvari_output_dir>
```

#### Key Parameters

- `--calls`: Input calls VCF file
- `--gt`: Input ground truth VCF file
- `--output_filename`: Output HDF5 file with concordance results
- `--outdir`: Full path to output directory for Truvari results

#### Optional Parameters

- `--hcr_bed`: High confidence region BED file
- `--pctseq`: Percentage of sequence identity (default: 0.0)
- `--pctsize`: Percentage of size identity (default: 0.0)
- `--maxsize`: Maximum size for SV comparison in bp (default: 50000, use -1 for unlimited)
- `--custom_info_fields`: Custom INFO fields to read from VCFs (can be specified multiple times)
- `--ignore_filter`: Ignore FILTER field in VCF (removes --passonly flag from Truvari)
- `--skip_collapse`: Skip VCF collapsing step for calls (ground truth is always collapsed)
- `--verbosity`: Logging level (default: INFO)

#### Example

```bash
sv_comparison_pipeline \
  --calls /data/sample.sv.vcf.gz \
  --gt /reference/HG004_sv_truth.vcf.gz \
  --output_filename /results/sample.sv_comp.h5 \
  --outdir /results/truvari_output \
  --hcr_bed /reference/HG004_sv_highconf.bed \
  --maxsize 100000 \
  --pctseq 0.7 \
  --pctsize 0.7 \
  --verbosity INFO
```

#### CNV Comparison

For copy number variant (CNV) comparisons, consider using a larger `--maxsize` value or -1 for unlimited:

```bash
sv_comparison_pipeline \
  --calls /data/sample.cnv.vcf.gz \
  --gt /reference/truth.cnv.vcf.gz \
  --output_filename /results/sample.cnv_comp.h5 \
  --outdir /results/truvari_cnv \
  --maxsize -1 \
  --ignore_filter
```

## Output Files

### run_comparison_pipeline

- **HDF5 file** (`output_file`): Contains concordance dataframes with classifications (TP, FP, FN)
  - `concordance` key: Main concordance results
  - `input_args` key: Input parameters used
  - Per-chromosome keys (for whole-genome mode)
- **BED files**: Generated from concordance results for visualization
- **Interval BED file** (`output_interval`): Intersected comparison intervals

### sv_comparison_pipeline

- **HDF5 file** (`output_filename`): Contains two concordance dataframes:
  - `base` key: Ground truth concordance (TP, FN)
  - `calls` key: Calls concordance (TP, FP)
- **Truvari directory** (`outdir`): Contains Truvari bench results:
  - `tp-base.vcf.gz`: True positive variants in ground truth
  - `tp-comp.vcf.gz`: True positive variants in calls
  - `fn.vcf.gz`: False negative variants
  - `fp.vcf.gz`: False positive variants
  - `summary.json`: Summary statistics
  - Collapsed and sorted VCF files

## Run with Docker

You can run the comparison tools using Docker:

```bash
docker run -v <local_data>:/data -v <local_output>:/output \
  337532070941.dkr.ecr.us-east-1.amazonaws.com/ugbio_comparison:latest \
  run_comparison_pipeline <arguments>
```

For SV comparison:

```bash
docker run -v <local_data>:/data -v <local_output>:/output \
  337532070941.dkr.ecr.us-east-1.amazonaws.com/ugbio_comparison:latest \
  sv_comparison_pipeline <arguments>
```

## Dependencies

The comparison module depends on:
- `ugbio_core[ml,vcfbed]` - Core utilities and VCF/BED processing
- External tools: bcftools, VCFEVAL (RTG Tools), Truvari

## Notes

- For best performance with large genomes, use parallel processing (`--n_jobs`)
- The `run_comparison_pipeline` supports both single-interval and whole-genome modes
- VCFEVAL requires an SDF index of the reference genome
- Truvari comparison includes automatic VCF collapsing and sorting
- Use `--ignore_filter_status` or `--ignore_filter` to compare all variants regardless of FILTER field
