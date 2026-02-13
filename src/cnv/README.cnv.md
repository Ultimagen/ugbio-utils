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
Process CNV calls in BED format from cn.mops and ControlFREEC: filter by length and low-complexity regions, annotate, and convert to VCF format.

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
- `--cnv_lcr_file` - UG-CNV-LCR BED file for filtering low-complexity regions (see workflows for the BED)
- `--min_cnv_length` - CNVs below this length will be marked (default: 10000)
- `--intersection_cutoff` - Overlap threshold with the cnv lcr(default: 0.5)

### Combining CNV Calls

Tools for combining and analyzing CNV calls (currently implemented combining of CNV calls from cn.mops and CNVPytor) are all aggregated under CLI interface `combine_cnmops_cnvpytor_cnv_calls`. This CLI contains the following tools - each can also be called by a standalone script:

```
    concat              Combine CNV VCFs from different callers (cn.mops and cnvpytor)
    filter_cnmops_dups  Filter short duplications from cn.mops calls in the combined CNV VCF
    annotate_gaps       Annotate CNV calls with percentage of gaps (Ns) from reference genome
    annotate_regions    Annotate CNV calls with region annotations from BED file
    merge_records       Merge adjacent or nearby CNV records in a VCF file
```

#### `concat`
Concatenate CNV VCF files from different callers (cn.mops and CNVpytor) into a single sorted and indexed VCF.
The tool adds "source" tag for each CNV

```bash
combine_cnv_vcfs \
  --cnmops_vcf cnmops1.vcf cnmops2.vcf \
  --cnvpytor_vcf cnvpytor1.vcf cnvpytor2.vcf \
  --output_vcf combined.vcf.gz \
  --fasta_index reference.fasta.fai \
  --out_directory ./output
```

#### `annotate_regions`
Annotate CNV calls with custom genomic regions that they overlap. The BED is expected to contain |-separated names of regions in the fourth column. The annotation is added to the info field under tag REGION_ANNOTATION

```bash
annotate_regions \
  --input_vcf calls.vcf.gz \
  --output_vcf annotated.vcf.gz \
  --annotation_bed regions.bed
```
#### `annotate_gaps`
Annotate CNV calls with percentage of Ns that they cover. Adds an info tag GAPS_PERCENTAGE

```bash
annotate_gaps \
  --calls_vcf calls.vcf.gz \
  --output_vcf annotated.vcf.gz \
  --ref_fasta Homo_sapiens_assembly38.fasta
```

#### `merge_records`

Combines overlapping records in the VCF

```bash
merge_records
   --input_vcf calls.vcf.gz \
   --output_vcf calls.combined.vcf.gz \
   --distance 0
```

#### `analyze_cnv_breakpoint_reads`
Analyze single-ended reads at CNV breakpoints to identify supporting evidence for duplications and deletions.
Counts of supporting evidence appear as info tags in the VCF

```bash
analyze_cnv_breakpoint_reads \
  --vcf-file cnv_calls.vcf.gz \
  --bam-file sample.bam \
  --output-file annotated.vcf.gz \
  --cushion 100 \
  --reference-fasta Homo_sapiens_assembly38.fasta
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

The module includes R scripts in the `cnmops/` directory. They are used by cn.mops pipeline and are not intended for standalone usage.

- `cnv_calling_using_cnmops.R` - Main cn.mops calling script
- `get_reads_count_from_bam.R` - Extract read counts from BAM files
- `create_reads_count_cohort_matrix.R` - Build cohort matrix for cn.mops
- `normalize_reads_count.R` - Normalize read counts across samples
- `rebin_cohort_reads_count.R` - Re-bin existing cohort to larger bin sizes

### Re-binning CNmops Cohorts

#### `rebin_cohort_reads_count.R`

Re-bin an existing cn.mops cohort from smaller bins to larger bins by aggregating read counts. This allows you to adjust the resolution of existing cohorts without regenerating from BAM files, which is useful for:
- Reducing computational memory requirements for large cohorts
- Faster CNV calling with coarser resolution
- Testing different bin sizes without re-processing BAM files

**Usage:**
```bash
Rscript cnmops/rebin_cohort_reads_count.R \
  -i cohort_1000bp.rds \
  -owl 1000 \
  -nwl 5000 \
  -o cohort_5000bp.rds \
  --save_csv
```

**Parameters:**
- `-i, --input_cohort_file` - Input cohort RDS file (required)
- `-owl, --original_window_length` - Original bin size in bp (required)
- `-nwl, --new_window_length` - New bin size in bp (required, must be divisible by original)
- `-o, --output_file` - Output RDS file (default: `rebinned_cohort_reads_count.rds`)
- `--save_csv` - Also save as CSV format
- `--save_hdf` - Also save as HDF5 format

**Important Notes:**
- New window length must be larger than and divisible by the original window length
- Genomic coordinates use 1-based, right-closed format (e.g., 1-1000, 1001-2000, ...)
- Partial bins at chromosome ends are preserved without artificial extension
- Read counts are summed from all original bins within each new bin
- Total read counts per sample are preserved across the rebinning

**Example:**
```bash
# Re-bin HapMap2 cohort from 1000 bp to 5000 bp
Rscript cnmops/rebin_cohort_reads_count.R \
  -i HapMap2_65samples_cohort_v2.0.hg38.ReadsCount.rds \
  -owl 1000 \
  -nwl 5000 \
  -o HapMap2_65samples_cohort_v2.0.hg38.ReadsCount.5000bp.rds

# Result: 3,088,281 bins → 617,665 bins (5x reduction)
# Last bin on chr1: 248955001-248956422 (partial bin, not extended to 248960000)
```
## Notes

- See germline and somatic CNV calling workflows published in GH repository `Ultimagen\healthomics-workflows`
for the reference implementations of the suggested workflows.
- For optimal CNV calling, use cohort-based approaches when multiple samples are available
- Filter CNV calls using the provided LCR (low-complexity region) files to reduce false positives
- Consider minimum CNV length thresholds based on your sequencing depth and biological context
- The module supports both GRCh37 and GRCh38 reference genomes
## Key Components

### process_cnvs

Process CNV calls from CN.MOPS or ControlFREEC in BED format: filter by length and UG-CNV-LCR, annotate with coverage statistics, and convert to VCF format.

**Note:** This module is called programmatically (not via CLI) from other pipeline scripts.

#### Programmatic Usage

The `process_cnvs` module is typically invoked from other pipeline components. Here are examples:

**Basic usage (minimal filtering):**
```python
from ugbio_cnv import process_cnvs

process_cnvs.run([
    "process_cnvs",
    "--input_bed_file", "cnv_calls.bed",
    "--fasta_index_file", "reference.fasta.fai",
    "--sample_name", "sample_001"
])
```

**With LCR filtering and length thresholds:**
```python
from ugbio_cnv import process_cnvs

process_cnvs.run([
    "process_cnvs",
    "--input_bed_file", "cnv_calls.bed",
    "--cnv_lcr_file", "ug_cnv_lcr.bed",
    "--min_cnv_length", "10000",
    "--intersection_cutoff", "0.5",
    "--fasta_index_file", "reference.fasta.fai",
    "--sample_name", "sample_001",
    "--out_directory", "/path/to/output/"
])
```

**Full pipeline with coverage annotations:**
```python
from ugbio_cnv import process_cnvs

process_cnvs.run([
    "process_cnvs",
    "--input_bed_file", "cnv_calls.bed",
    "--cnv_lcr_file", "ug_cnv_lcr.bed",
    "--min_cnv_length", "10000",
    "--sample_norm_coverage_file", "sample.normalized_coverage.bed",
    "--cohort_avg_coverage_file", "cohort.average_coverage.bed",
    "--fasta_index_file", "reference.fasta.fai",
    "--sample_name", "sample_001",
    "--out_directory", "/path/to/output/",
    "--verbosity", "INFO"
])
```

**Input:** BED file with CNV calls from CN.MOPS or ControlFREEC

**Output:** Filtered and annotated VCF file with CNV calls (`.vcf.gz` and `.vcf.gz.tbi`)
