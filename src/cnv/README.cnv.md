# ugbio_cnv

This module provides Python scripts and utilities for Copy Number Variation (CNV) analysis in bioinformatics pipelines.

## Overview

The CNV module integrates multiple CNV calling algorithms and provides tools for processing, filtering, combining, annotating, and visualizing CNV calls. It supports both germline and somatic CNV detection workflows.

Apart from the depth CNV caller, the package does not call variants itself: it provides facilities for preparing
data, calling, changing the format and combining callsets.

The package is designed to work with the following CNV callers:

- **Depth CNV caller** (this package) - cohort-free germline read-depth caller, see below
- **cn.mops** - cohort read-depth caller (R scripts in `cnmops/`), optional extra candidates for the depth caller
- **ControlFREEC** - Control-FREEC for somatic CNV detection

## Installation

### Using UV (Recommended)

Install the CNV module and its dependencies:

```bash
pip install ugbio-cnv
```

Pre-built docker image can be downloaded from Dockerhub: [`ultimagenomics/ugbio_cnv`](https://hub.docker.com/r/ultimagenomics/ugbio_cnv)


## Available Tools

### CNV Processing

#### `process_cnvs`
Process CNV calls in BED format from cn.mops or ControlFREEC: filter by length and low-complexity regions, annotate, and convert to VCF format.

```bash
process_cnvs \
  --input_bed_file cnv_calls.bed \
  --cnv_lcr_file ug_cnv_lcr.bed \
  --min_cnv_length 10000 \
  --intersection_cutoff 0.5 \
  --out_directory ./output
```

**Key Parameters:**
- `--input_bed_file` - Input CNV BED file
- `--cnv_lcr_file` - UG-CNV-LCR BED file for filtering low-complexity regions (see workflows for the BED)
- `--min_cnv_length` - CNVs below this length will be marked (default: 10000)
- `--intersection_cutoff` - Overlap threshold with the cnv lcr(default: 0.5)

#### `add_cipos_to_vcf`
Add CIPOS (± half the window size) to the calls of a cn.mops VCF.

```bash
add_cipos_to_vcf --input_vcf calls.vcf.gz --window_size 1000 --output_vcf calls.cipos.vcf.gz
```

### Breakpoint evidence and SV merge

#### `analyze_cnv_breakpoint_reads`
Analyze reads at CNV breakpoints to identify supporting evidence for duplications and deletions.
Counts of supporting evidence appear as info tags in the VCF

```bash
analyze_cnv_breakpoint_reads \
  --vcf-file cnv_calls.vcf.gz \
  --bam-file sample.bam \
  --output-file annotated.vcf.gz \
  --cushion 100 \
  --reference-fasta Homo_sapiens_assembly38.fasta
```

Discordant pairs count too, in their own fields: `CNV_DUP_PAIRS` / `CNV_DEL_PAIRS`, `CNV_*_PAIRS_FRAC`
(over `CNV_TOTAL_READS`) and `*_PAIRS_MEDIAN_SPAN`. `CNV_*_READS`, `CNV_*_FRAC` and
`*_READS_MEDIAN_INSERT_SIZE` stay split-read only. Each fragment votes at most once per kind of evidence, so
a fragment with both a split read and a discordant pair counts in both. The pair fields are always written;
on single-end input they are 0. Pairs are classified by `pair_svtype` (FR = deletion, RF = duplication), the rule
`pair_cnv_candidates` uses too.

```bash
analyze_cnv_breakpoint_reads \
  --vcf-file cnv_calls.vcf.gz \
  --bam-file sample.cram \
  --output-file annotated.vcf.gz \
  --cushion 1500 \
  --reference-fasta Homo_sapiens_assembly38.fasta \
  --min-pair-span 800
```

- An everted (RF) pair is DUP evidence and an FR pair DEL evidence, if the pair starts in the start window and ends in the end window.
- A pair must span at least `--min-pair-span` bases (template length, default 800). Its alignment must have MAPQ of at least `--min-pair-mapping-quality` (default 20).
- With `--output-bam`, pair mates get read group `PAIR`, apart from the split reads (`DUP` / `DEL`).

#### `merge_cnv_sv`
Merge CNV calls with SV calls (e.g. GRIDSS) of the same sample.

- SV deletions and duplications in the size range are added.
- Where a CNV and an SV call overlap (truvari collapse), the higher-QUAL call is kept, with the sources and IDs of both.

```bash
merge_cnv_sv \
  --cnv_vcf cnv_calls.vcf.gz \
  --sv_vcf sv_calls.vcf.gz \
  --output_vcf merged.vcf.gz \
  --fasta_index Homo_sapiens_assembly38.fasta.fai \
  --min_sv_length 500
```

### Depth CNV caller (cohort-free germline calling)

Calls germline CNVs (≥ 500 bp) of one sample from its own read depth, without a cohort, with one call per event.

Steps for one sample:

```bash
# once per reference: GC, N and, per BED track, covered fraction per 100 bp bin
# (the caller reads the lowmap and segdup tracks; a missing one counts as 0)
build_depth_annotation --reference-fasta ref.fasta --output ref.depth_annotation.parquet \
  --bed lowmap=GRCh38_lowmappabilityall.bed.gz segdup=GRCh38_segdups.bed.gz

# depth in 100 bp bins, all reads and MAPQ >= 20
mosdepth -n -x -b 100 -f ref.fasta -Q 0 sample.q0 sample.cram
mosdepth -n -x -b 100 -f ref.fasta -Q 20 sample.q20 sample.cram

# depth calls: GC correction, weighted change-point segmentation, one call per event
call_depth_cnvs --depth-mq0 sample.q0.regions.bed.gz --depth-mq20 sample.q20.regions.bed.gz \
  --annotation ref.depth_annotation.parquet --sample-name SAMPLE \
  --fasta-index ref.fasta.fai --output-vcf sample.depth.vcf.gz

# adds events from clusters of split reads that the depth misses
split_read_cnv_candidates --alignment sample.cram --reference-fasta ref.fasta --depth-vcf sample.depth.vcf.gz \
  --depth-mq0 sample.q0.regions.bed.gz --depth-mq20 sample.q20.regions.bed.gz \
  --annotation ref.depth_annotation.parquet --output-vcf sample.cand.vcf.gz

# paired-end only: adds events from clusters of discordant pairs
pair_cnv_candidates --alignment sample.cram --reference-fasta ref.fasta --input-vcf sample.cand.vcf.gz \
  --depth-mq0 sample.q0.regions.bed.gz --depth-mq20 sample.q20.regions.bed.gz \
  --annotation ref.depth_annotation.parquet --output-vcf sample.pair_cand.vcf.gz

# optional: adds cn.mops calls (process_cnvs output) the depth caller misses
cnmops_cnv_candidates --input-vcf sample.pair_cand.vcf.gz --cnmops-vcf sample.cnmops.vcf.gz \
  --depth-mq0 sample.q0.regions.bed.gz --depth-mq20 sample.q20.regions.bed.gz \
  --annotation ref.depth_annotation.parquet --output-vcf sample.cnmops_cand.vcf.gz

# split-read and pair evidence at the breakpoints, on the last candidates VCF
# (then samtools sort + index the evidence BAM)
analyze_cnv_breakpoint_reads --bam-file sample.cram --vcf-file sample.cnmops_cand.vcf.gz \
  --reference-fasta ref.fasta --cushion 1500 --output-file sample.ann.vcf.gz --output-bam sample.evidence.bam

# moves the breakpoints to split-read junctions, or shrinks deletions to the discordant-pair interval
refine_cnv_breakpoints --input-vcf sample.ann.vcf.gz --alignment sample.cram --reference-fasta ref.fasta \
  --output-vcf sample.refined.vcf.gz

# drops candidates that are short, inside a depth call, or duplicates
filter_nested_cnv_candidates --input-vcf sample.refined.vcf.gz --output-vcf sample.calls.vcf.gz
```

Sex chromosomes are inferred from the depth (or set with `--karyotype`, e.g. `XX`, `XY`, `XXY`, `X0`: one X makes
chrX haploid, a Y makes chrY haploid, no Y leaves chrY uncalled). For a sample with one X, pass
`--par-bed` (the `par_regions` genome resource); otherwise the chrX pseudoautosomal regions look like gains.

Filter the calls with `filter_variants_pipeline` and a `cnv` model. Pass the fields in
`ugbio_filtering.transformers.CNV_INFO_FIELDS` as `--custom_annotations` and use `--overwrite_qual_tag`. Without
`--decision_threshold`, the threshold stored with the model is used (28 for `cnv_filtering_model_v2.0.0.pkl`), else 30.

Then merge overlapping PASS calls of the same type into one call (DEL and DUP are never merged, so a position has at
most two PASS calls). A non-PASS call at least half inside a PASS call of either type is dropped; the other non-PASS
calls are kept unchanged:

```bash
collapse_cnv_calls --input-vcf sample.filtered.vcf.gz --output-vcf sample.cnv.vcf.gz
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
  --germline_coverage germline_coverage.bed \
  --tumor_coverage tumor_coverage.bed \
  --duplication_cnv_calls dup_calls.bed \
  --deletion_cnv_calls del_calls.bed \
  --out_directory ./plots
```

#### `plot_FREEC_neutral_AF`
Generate histogram of allele frequencies at neutral (non-CNV) locations.

```bash
plot_FREEC_neutral_AF \
  --mpileup sample.pileup \
  --cnvs_file sample_CNVs \
  --sample_name SAMPLE \
  --out_directory ./plots
```

## Dependencies

The module depends on:

- **Python 3.11+**
- **ugbio_core** - Core utilities from this workspace
- **Bioinformatics tools**: samtools, bedtools, bcftools
- **R** with cn.mops (Ultimagen fork), argparse, magrittr, rhdf5, rtracklayer, GenomicRanges: only for the optional
  cn.mops path (`cnmops/`; in the cnv image)

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
- `-owl, --original_window_length` - Original bin size in bp (optional), autodetected if not given
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
  -nwl 5000 \
  -o HapMap2_65samples_cohort_v2.0.hg38.ReadsCount.5000bp.rds

# Result: 3,088,281 bins → 617,665 bins (5x reduction)
# Last bin on chr1: 248955001-248956422 (partial bin, not extended to 248960000)
```
## Notes

- See germline and somatic CNV calling workflows published in GH repository `Ultimagen\healthomics-workflows`
for the reference implementations of the suggested workflows.
- `process_cnvs` (cn.mops, ControlFREEC) can filter calls with the UG-CNV-LCR (low-complexity region) BED
- Consider minimum CNV length thresholds based on your sequencing depth and biological context
- The module supports both GRCh37 and GRCh38 reference genomes
## Key Components

### process_cnvs

Process CNV calls from CN.MOPS or ControlFREEC in BED format: filter by length and UG-CNV-LCR, annotate with coverage statistics, and convert to VCF format.

**Note:** The pipelines run it as the `process_cnvs` CLI; it can also be called from Python.

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
