# ugbio_comparison

This module includes comparison python scripts and utilities for bioinformatics pipelines. It provides tools for comparing VCF callsets against ground truth datasets, with support for both small variant, structural variant (SV) and copy number (CNV) comparisons.

## Overview

The comparison module provides two main CLI scripts for variant comparison:

1. **run_comparison_pipeline** - Compare small variant callsets to ground truth using VCFEVAL
2. **sv_comparison_pipeline** - Compare structural variant (SV) callsets using Truvari

## Installation

To install the comparison module with all dependencies:

```bash
pip install ugbio-comparison
```

The tool can also be run from the docker image in dockerhub [`ultimagenomics/ugbio_comparison`](https://hub.docker.com/r/ultimagenomics/ugbio_comparison).

## CLI Scripts

### 1. run_comparison_pipeline

Compare VCF callsets to ground truth using VCFEVAL as the comparison engine. This pipeline supports, annotation with various genomic features, and detailed concordance analysis of specific variant types (downstream).

#### Purpose

- Compare variant calls against a ground truth dataset
- Generate concordance metrics (TP, FP, FN)
- Annotate variants with coverage, mappability, and other genomic features
- Annotate variants with properties like SNV/Indel/homopolymer Indel etc.

#### Usage

```bash
run_comparison_pipeline \
  --input_prefix <input_vcf_prefix> \
  --output_file <output_h5_file> \
  --output_interval <output_bed_file> \
  --gtr_vcf <ground_truth_vcf> \
  --highconf_intervals <high_confidence_bed> \
  --reference <reference_fasta> \
  --call_sample_name <sample_name> \
  --truth_sample_name <truth_sample_name> \
```

#### Key Parameters

- `--input_prefix`: Prefix of the input VCF file(s)
- `--output_file`: Output HDF5 file containing concordance results
- `--output_interval`: Output BED file of intersected intervals
- `--gtr_vcf`: Ground truth VCF file for comparison (e.g. GIAB VCF)
- `--cmp_intervals`: Optional regions for comparison (BED/interval_list)
- `--highconf_intervals`: High confidence intervals (e.g. GIAB HCR BED)
- `--reference`: Reference genome FASTA file
- `--reference_dict`: Reference genome dictionary file
- `--call_sample_name`: Name of the call sample
- `--truth_sample_name`: Name of the truth sample

#### Optional Parameters

- `--coverage_bw_high_quality`: Input BigWig file with high MAPQ coverage
- `--coverage_bw_all_quality`: Input BigWig file with all MAPQ coverage
- `--annotate_intervals`: Interval files for annotation (can be specified multiple times)
- `--runs_intervals`: Homopolymer runs intervals (BED file), used for annotation of closeness to homopolymer indel
- `--ignore_filter_status`: Ignore variant filter status
- `--enable_reinterpretation`: Enable variant reinterpretation (i.e. reinterpret variants using likely false hmer indel)
- `--scoring_field`: Alternative scoring field to use (copied to TREE_SCORE)
- `--flow_order`: Sequencing flow order (4 cycle, TGCA)
- `--n_jobs`: Number of parallel jobs for chromosome processing (default: -1 for all CPUs)
- `--use_tmpdir`: Store temporary files in temporary directory
- `--verbosity`: Logging level (ERROR, WARNING, INFO, DEBUG)

#### Output Files

- **HDF5 file** (`output_file`): Contains concordance dataframes with classifications (TP, FP, FN)
  - `concordance` key: Main concordance results
  - `input_args` key: Input parameters used
  - Per-chromosome keys (for whole-genome mode)
- **BED files**: Generated from concordance results for visualization

#### Example

```bash
run_comparison_pipeline \
  --input_prefix /data/sample.filtered \
  --output_file /results/sample.comp.h5 \
  --output_interval /results/sample.comp.bed \
  --gtr_vcf /reference/HG004_truth.vcf.gz \
  --highconf_intervals /reference/HG004_highconf.bed \
  --reference /reference/Homo_sapiens_assembly38.fasta \
  --call_sample_name SAMPLE-001 \
  --truth_sample_name HG004 \
  --n_jobs 8 \
  --verbosity INFO \
```

### 2. sv_comparison_pipeline

Compare structural variant (SV) callsets using Truvari for benchmarking. This pipeline collapses VCF files, runs Truvari bench, and generates concordance dataframes.

#### Purpose

- Compare SV calls against a ground truth dataset using Truvari
- We recommend using SV ground truth callsets from NIST as the source of truth
- Collapse overlapping variants before comparison
- Generate detailed concordance metrics for SVs
- Support for different SV types (DEL, INS, DUP, etc.)
- Output results in HDF5 format with base and calls concordance

#### Usage

```bash
sv_comparison_pipeline \
  --calls <input_calls_vcf> \
  --gt <ground_truth_vcf> \
  --hcr_bed <high confidence bed> \
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

#### Output files

- **HDF5 file** (`output_filename`): Contains two concordance dataframes:
  - `base` key: Ground truth concordance (TP, FN)
  - `calls` key: Calls concordance (TP, FP)
- **Truvari directory** (`outdir`): Contains Truvari bench results:
  - `tp-base.vcf.gz`: True positive variants in ground truth
  - `tp-comp.vcf.gz`: True positive variants in calls
  - `fn.vcf.gz`: False negative variants
  - `fp.vcf.gz`: False positive variants
  - `summary.json`: Summary statistics

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


## Dependencies

The following binary tools are included in the Docker image and need to be installed for standalone running:
- **bcftools** 1.20 - VCF/BCF manipulation
- **samtools** 1.20 - SAM/BAM/CRAM manipulation
- **bedtools** 2.31.0 - Genome interval operations
- **bedops** - BED file operations
- **GATK** 4.6.0.0 - Genome Analysis Toolkit
- **Picard** 3.3.0 - Java-based command-line tools for manipulating high-throughput sequencing data
- **RTG Tools** 3.12.1 - Provides VCFEVAL for variant comparison

## Notes

- For best performance with large genomes, use parallel processing (`--n_jobs`)
- The `run_comparison_pipeline` supports both single-interval and whole-genome modes
- VCFEVAL requires an SDF index of the reference genome
- Truvari comparison includes automatic VCF collapsing and sorting
- Use `--ignore_filter_status` or `--ignore_filter` to compare all variants regardless of FILTER field
