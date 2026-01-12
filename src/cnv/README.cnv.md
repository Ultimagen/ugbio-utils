# ugbio_cnv

This module provides Python scripts and utilities for Copy Number Variation (CNV) analysis in bioinformatics pipelines.

## Overview

The CNV module integrates multiple CNV calling algorithms and provides tools for processing, filtering, combining, annotating, and visualizing CNV calls. It supports both germline and somatic CNV detection workflows.

Note that the package itself does not call variants, it just provides facilities for preparing data, calling, changing the format and combining callsets.

The package is designed to work with the following CNV callers:

- **cn.mops** - Read depth-based CNV caller using a Bayesian approach
- **CNVpytor** - Read depth analysis for CNV detection
- **ControlFREEC** - Control-FREEC for somatic CNV detection

## Installation

### Using UV (Recommended)

Install the CNV module and its dependencies:

```bash
pip install ugbio-cnv
```

Pre-built docker image can be downloaded from Dockerhub: [`ultimagenomics/ugbio_cnv`](https://hub.docker.com/r/ultimagenomics/ugbio_filtering)


## Available Tools

### CNV Processing

#### `process_cnmops_cnvs`
Process CNV calls from cn.mops: filter by length and low-complexity regions, annotate, and convert to VCF format.

```bash
process_cnmops_cnvs \
  --input_bed_file cnmops_calls.bed \
  --cnv_lcr_file ug_cnv_lcr.bed \
  --min_cnv_length 10000 \
  --intersection_cutoff 0.5 \
  --out_directory ./output
```

**Key Parameters:**
- `--input_bed_file` - Input BED file from cn.mops
- `--cnv_lcr_file` - UG-CNV-LCR BED file for filtering low-complexity regions
- `--min_cnv_length` - Minimum CNV length to report (default: 10000)
- `--intersection_cutoff` - Overlap threshold for bedtools subtract (default: 0.5)

### Combining CNV Calls

#### `combine_cnv_vcfs`
Combine CNV VCF files from different callers (cn.mops and CNVpytor) into a single sorted and indexed VCF.

```bash
combine_cnv_vcfs \
  --cnmops_vcf cnmops1.vcf cnmops2.vcf \
  --cnvpytor_vcf cnvpytor1.vcf cnvpytor2.vcf \
  --output_vcf combined.vcf.gz \
  --fasta_index reference.fasta.fai \
  --out_directory ./output
```

#### `combine_cnmops_cnvpytor_cnv_calls`
Advanced tool for combining and merging CNV calls with configurable distance thresholds.

```bash
combine_cnmops_cnvpytor_cnv_calls \
  --cnmops_vcf cnmops_calls.vcf \
  --cnvpytor_vcf cnvpytor_calls.vcf \
  --output_vcf merged.vcf.gz \
  --fasta_index reference.fasta.fai \
  --merge_distance 1000
```

    #### `filter_dup_cnmmops_cnv_calls`
Add CNMOPS_SHORT_DUPLICATION filter to short duplications in cn.mops calls.

```bash
filter_dup_cnmmops_cnv_calls \
  --input_vcf cnmops_calls.vcf \
  --output_vcf filtered.vcf \
  --min_dup_length 1000
```

### Annotation Tools

#### `annotate_vcf_with_regions`
Annotate CNV calls with custom genomic regions or calculate gap (N) percentage in reference genome.

```bash
annotate_vcf_with_regions \
  --input_vcf calls.vcf \
  --output_vcf annotated.vcf \
  --annotation_bed regions.bed \
  --annotation_name CUSTOM_REGION
```

#### `analyze_cnv_breakpoint_reads`
Analyze single-ended reads at CNV breakpoints to identify supporting evidence for duplications and deletions.

```bash
analyze_cnv_breakpoint_reads \
  --input_vcf cnv_calls.vcf \
  --input_bam sample.bam \
  --output_vcf annotated.vcf \
  --window_size 100
```

### Somatic CNV Tools (ControlFREEC)

#### `annotate_FREEC_segments`
Annotate segments from ControlFREEC output as gain/loss/neutral based on fold-change thresholds.

```bash
annotate_FREEC_segments \
  --input_segments_file segments.txt \
  --gain_cutoff 1.03 \
  --loss_cutoff 0.97 \
  --out_directory ./output
```

### Visualization

#### `plot_cnv_results`
Generate coverage plots along the genome for germline and tumor samples.

```bash
plot_cnv_results \
  --sample_name SAMPLE \
  --germline_cov_file germline_coverage.bed \
  --tumor_cov_file tumor_coverage.bed \
  --cnv_file cnv_calls.bed \
  --out_directory ./plots
```

#### `plot_FREEC_neutral_AF`
Generate histogram of allele frequencies at neutral (non-CNV) locations.

```bash
plot_FREEC_neutral_AF \
  --input_file neutral_regions.txt \
  --sample_name SAMPLE \
  --out_directory ./plots
```

## Dependencies

The module depends on:

- **Python 3.11+**
- **ugbio_core** - Core utilities from this workspace
- **CNVpytor** (1.3.1) - Python-based CNV caller
- **cn.mops** (R package) - Bayesian CNV detection
- **Bioinformatics tools**: samtools, bedtools, bcftools
- **R 4.3.1** with Bioconductor packages

## Key R Scripts

The module includes R scripts in the `cnmops/` directory:

- `cnv_calling_using_cnmops.R` - Main cn.mops calling script
- `get_reads_count_from_bam.R` - Extract read counts from BAM files
- `create_reads_count_cohort_matrix.R` - Build cohort matrix for cn.mops
- `normalize_reads_count.R` - Normalize read counts across samples

## Notes

- See germline and somatic CNV calling workflows published in GH repository `Ultimagen\healthomics-workflows`
for the reference implementations of the suggested workflows.
- For optimal CNV calling, use cohort-based approaches when multiple samples are available
- Filter CNV calls using the provided LCR (low-complexity region) files to reduce false positives
- Consider minimum CNV length thresholds based on your sequencing depth and biological context
- The module supports both GRCh37 and GRCh38 reference genomes
