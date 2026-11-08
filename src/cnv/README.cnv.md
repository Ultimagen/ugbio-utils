# ugbio_cnv

This module provides Python scripts and utilities for Copy Number Variation (CNV) analysis in bioinformatics pipelines.

## Overview

The CNV module integrates multiple CNV calling algorithms and provides tools for processing, filtering, combining, annotating, and visualizing CNV calls. It supports both germline and somatic CNV detection workflows.

Note that the package itself does not call variants, it just provides facilities for preparing data, calling, changing the format and combining callsets.

The package is designed to work with the following CNV callers:

- **Depth CNV caller** (this package) - cohort-free germline read-depth caller, see below
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

#### `process_cnvs`
Process CNV calls in BED format from ControlFREEC: filter by length and low-complexity regions, annotate, and convert to VCF format.

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
on single-end input they are 0.

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
Merge CNV calls with SV calls (e.g. GRIDSS) of the same sample: SV deletions and duplications in the size range are
added, and where a CNV and an SV call overlap (truvari collapse) the higher-QUAL call is kept with the sources and IDs of
both.

```bash
merge_cnv_sv \
  --cnv_vcf cnv_calls.vcf.gz \
  --sv_vcf sv_calls.vcf.gz \
  --output_vcf merged.vcf.gz \
  --fasta_index Homo_sapiens_assembly38.fasta.fai \
  --min_sv_length 1000
```

### Depth CNV caller (cohort-free germline calling)

Calls germline CNVs (≥500 bp) of one sample from its own read depth, with no cohort, and gives one call per event.
Depth in 100 bp windows (mosdepth, all reads and MAPQ≥20) is GC-corrected on the sample, segmented with a weighted
change-point search (segdup / low-mappability windows weighted by their own noise), and same-direction calls are
merged across noise and masked gaps. Split-read junction clusters add short events the depth misses. Breakpoints are
then annotated with the existing evidence tool and snapped to split-read junctions.

Order of the steps for one sample:

```bash
# once per reference
build_depth_annotation --reference-fasta ref.fasta --lowmap-bed GRCh38_lowmappabilityall.bed.gz \
  --segdup-bed GRCh38_segdups.bed.gz --output ref.depth_annotation.parquet

mosdepth -n -x -b 100 -f ref.fasta -Q 0 sample.q0 sample.cram
mosdepth -n -x -b 100 -f ref.fasta -Q 20 sample.q20 sample.cram

call_depth_cnvs --depth-mq0 sample.q0.regions.bed.gz --depth-mq20 sample.q20.regions.bed.gz \
  --annotation ref.depth_annotation.parquet --sample-name SAMPLE \
  --fasta-index ref.fasta.fai --output-vcf sample.depth.vcf.gz
split_read_cnv_candidates --alignment sample.cram --reference-fasta ref.fasta --depth-vcf sample.depth.vcf.gz \
  --depth-mq0 sample.q0.regions.bed.gz --depth-mq20 sample.q20.regions.bed.gz \
  --annotation ref.depth_annotation.parquet --output-vcf sample.cand.vcf.gz
analyze_cnv_breakpoint_reads --bam-file sample.cram --vcf-file sample.cand.vcf.gz --reference-fasta ref.fasta \
  --cushion 1500 --output-file sample.ann.vcf.gz --output-bam sample.evidence.bam   # then samtools sort + index
snap_cnv_breakpoints --input-vcf sample.ann.vcf.gz --alignment sample.cram --reference-fasta ref.fasta \
  --output-vcf sample.snap.vcf.gz
filter_nested_cnv_candidates --input-vcf sample.snap.vcf.gz --output-vcf sample.calls.vcf.gz
```

- `filter_nested_cnv_candidates` drops split-read and pair candidates that are < 500 bp after snapping, >= 50% inside
  a same-type depth call, or with >= 50% reciprocal overlap with a same-type candidate kept before them (snapping often
  moves several candidates of one deletion to the same breakpoints; split-read candidates are kept first, then pair
  candidates, by the reads in their cluster).

- Discordant-pair candidates (paired-end, optional; not in the pipeline yet): `pair_cnv_candidates` runs after
  `split_read_cnv_candidates` on its output (same `--depth-mq0` / `--depth-mq20` / `--annotation`, and `--input-vcf`
  instead of `--depth-vcf`). FR pairs with a span >= 800 (the evidence step's `--min-pair-span` default) give deletion
  candidates; RF (everted) pairs with the mate > 50 bp after the read give duplication candidates (an everted pair
  lies inside the duplicated copy, so a span floor would miss small duplications); no insert size is estimated.
  Clusters of >= 3 pairs: a deletion is the gap between the reads and their mates (snapping's pair rule), kept from
  300 bp since the 500 bp minimum is applied after snapping; a duplication contains the pairs, 500 bp - 1 Mb.
  With the split-read candidates' depth check, and not already called. They get `CNV_SOURCE=pair` and
  `PAIR_READS`; overlapping pair clusters are collapsed to the one with most pairs, and
  `filter_nested_cnv_candidates` treats them like split-read candidates. The filter model
  `cnv_filtering_model_v2.0.0.pkl` is trained with them (`source_pair` feature).
- Filter training: `cnv_known_event_pieces` cuts a sample's known long CNV into 10–100 kb depth calls (the windows the
  caller would call), as extra positives for `training_prep_depth_cnv_pipeline --known_events` (README.filtering.md).
- Sex chromosomes: `call_depth_cnvs` infers them from the median depth of the clean (unique, non-segdup) chrX and
  chrY bins relative to the autosomes, with the karyotype table of `estimate_ploidy`. One chrX copy makes chrX
  haploid, two diploid; chrY is haploid when present and not called otherwise. If the depth is ambiguous, chrX and
  chrY are not called (the autosomes are). `--sex male|female` sets it instead. The setup is written to the VCF
  header (`depth_cnv_haploid_contigs`, `depth_cnv_skipped_contigs`, `depth_cnv_karyotype`), and
  `split_read_cnv_candidates` reads it from there.
- The output is filtered with `filter_variants_pipeline` and a `cnv_depth` model (`ugbio_filtering`); the INFO fields
  the model reads are listed in `ugbio_filtering.transformers.DEPTH_CNV_INFO_FIELDS` (pass each as
  `--custom_annotations`, and use `--overwrite_qual_tag` so QUAL holds the score for `merge_cnv_sv`).
- Calls carry `CNV_SOURCE=depth` or `split_read`, `CN_MEAN`, `BASE_CN`, `Z`, `SEGDUP`, `UNIQ_FRAC`, `LOW_CONF` and the
  snapping fields `SNAP`, `SNAP_N`, `DEPTH_POS`, `DEPTH_END` (see `cnv_vcf_consts.DEPTH_CNV_INFO_TAGS`).

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
- **Bioinformatics tools**: samtools, bedtools, bcftools

## Notes

- See germline and somatic CNV calling workflows published in GH repository `Ultimagen\healthomics-workflows`
for the reference implementations of the suggested workflows.
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
