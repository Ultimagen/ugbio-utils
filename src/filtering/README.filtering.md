# ugbio_filtering

This module includes filtering python scripts for bioinformatics pipelines.
It provides tools for variant filtering, model training, and quality control of variant calls.

## Installation

To install the filtering module with all dependencies:

```bash
pip install ugbio_filtering
```

The tool with all required dependencies can be also used in docker image: [`ultimagenomics/ugbio_filtering`](https://hub.docker.com/r/ultimagenomics/ugbio_filtering)
## CLI Scripts

The filtering module provides several command-line tools for different stages of the variant filtering pipeline:

### Variant Filtering

#### `filter_variants_pipeline`

Applies machine learning-based filtering to VCF files after GATK variant calling.

**Purpose**: Filter variants using trained models, blacklists, and custom annotation bed files to improve callset quality.

**Usage**:
```bash
filter_variants_pipeline \
  --input_file input.vcf.gz \
  --output_file filtered.vcf.gz \
  --model_file model.pkl \
  [--custom_annotations ANNOTATION1 ANNOTATION2] \
  [--decision_threshold 30] \
  [--treat_multiallelics --ref_fasta reference.fa] \
  [--recalibrate_genotype] \
  [--overwrite_qual_tag] \
  [--limit_to_contigs chr1 chr2]
```

**Key Parameters**:
- `--input_file`: Input VCF file (requires .tbi index)
- `--output_file`: Output VCF file with filtering annotations
- `--model_file`: Pickle file containing trained XGBoost model and transformer
- `--decision_threshold`: Score threshold for filtering variants (default: 30)
- `--treat_multiallelics`: Apply special handling for multiallelic sites
- `--recalibrate_genotype`: Allow model to re-call genotypes

#### `filter_low_af_ratio_to_background`

Filters somatic variants based on allele frequency ratio to background.

**Purpose**: Remove variants with low AF ratio in GT ALT alleles compared to background, useful for somatic variant calling.

**Usage**:
```bash
filter_low_af_ratio_to_background \
  input.vcf.gz \
  output.vcf.gz \
  [--af_ratio_threshold 10] \
  [--af_ratio_threshold_h_indels 0] \
  [--tumor_vaf_threshold_h_indels 0] \
  [--new_filter LowAFRatioToBackground]
```

**Key Parameters**:
- `input.vcf.gz`: Input VCF file
- `output.vcf.gz`: Output VCF file
- `--af_ratio_threshold`: AF ratio threshold for SNPs and non-h-indels (default: 10)
- `--af_ratio_threshold_h_indels`: AF ratio threshold for h-indels (default: 0)
- `--tumor_vaf_threshold_h_indels`: Tumor VAF threshold for h-indel filtering (default: 0)
- `--new_filter`: Name of the FILTER tag to add (default: LowAFRatioToBackground)

### Model Training

#### `train_models_pipeline`

Trains machine learning models for variant filtering using prepared ground truth data.

**Purpose**: Train XGBoost models on labeled training data to distinguish true variants from false positives.

**Usage**:
```bash
train_models_pipeline \
  --train_dfs train1.h5 train2.h5 \
  --test_dfs test1.h5 test2.h5 \
  --output_file_prefix model_output \
  [--gt_type exact|approximate] \
  [--vcf_type single_sample|deep_variant|cnv_depth] \
  [--custom_annotations ANNOTATION1 ANNOTATION2] \
  [--verbosity INFO]
```

**Key Parameters**:
- `--train_dfs`: Training HDF5 files (output from training_prep_pipeline)
- `--test_dfs`: Test HDF5 files for model evaluation
- `--output_file_prefix`: Prefix for output .pkl (model) and .h5 (results) files
- `--gt_type`: Ground truth type - "exact" or "approximate" (default: exact)
- `--vcf_type`: VCF type - "single_sample", "deep_variant", or "cnv_depth" (depth CNV caller; default: single_sample)
- `--custom_annotations`: Additional INFO annotations to include in training

#### `training_prep_pipeline`

Prepares training data by comparing variant calls to ground truth.

**Purpose**: Generate labeled training datasets (true positives, false positives, false negatives) for model training.

**Usage**:
```bash
training_prep_pipeline \
  --call_vcf calls.vcf.gz \
  --gt_type exact|approximate \
  --output_prefix training_data \
  [--base_vcf truth.vcf.gz] \
  [--reference reference.fa] \
  [--reference_sdf reference.sdf] \
  [--hcr high_confidence.bed] \
  [--blacklist blacklist.pkl] \
  [--custom_annotations ANNOTATION1 ANNOTATION2] \
  [--contigs_to_read chr1 chr2] \
  [--contig_for_test chr3] \
  [--ignore_genotype] \
  [--verbosity INFO]
```

**Key Parameters**:
- `--call_vcf`: VCF file with variant calls to evaluate
- `--gt_type`: Ground truth type - "exact" (requires base_vcf) or "approximate"
- `--base_vcf`: Truth VCF file (required for exact ground truth)
- `--reference`: Reference FASTA file prefix (requires .fai index)
- `--reference_sdf`: Reference SDF folder (RTG format). If not provided, uses `<reference>.sdf`
- `--hcr`: High confidence regions BED file
- `--output_prefix`: Prefix for output HDF5 files (train and test sets)
- `--contig_for_test`: Chromosome to use as test set
- `--ignore_genotype`: Ignore genotype when comparing to ground truth

#### `training_prep_depth_cnv_pipeline`

Prepares training data for the depth CNV caller's filter (`--vcf_type cnv_depth`).

**Purpose**: label the calls of a sample with truth (e.g. HG002) and write train / test HDF5 files, split by chromosome.

- Every call ≥ `--min_len` (500 bp) inside the benchmark BED is labelled with truvari bench against the truth:
  matched → 1, unmatched → 0 (pctovl 0.5, pctsize 0.5, pctseq 0, refdist 100 kb, duplications matched to insertions).
- `--known_events CALL_VCF EVENTS_BED` (repeatable) adds positives from samples with known events (e.g. cell lines):
  calls of the event's type (BED: chrom, start, end, svtype) with ≥ 50% of their length inside an event.
- The depth caller gives one call per event, so a long known event adds one row. `cnv_known_event_pieces` (ugbio_cnv)
  cuts it into training pieces: windows of 10–100 kb inside the event, scored with the depth caller's statistics and
  kept when the caller would call them (no N, ≤ 10% segdup / low-mappability bins). After the pipeline's evidence and
  snapping steps they go in as `--known_events` too, and teach the filter that a depth-only call without junction
  reads can be real.

```bash
# pieces of a known event (per sample; --haploid-contigs chrX for a male sample), then evidence and snapping
cnv_known_event_pieces --depth-mq0 YLC444.q0.regions.bed.gz --depth-mq20 YLC444.q20.regions.bed.gz \
  --annotation ref.depth_annotation.parquet --events-bed ylc444.events.bed --sample-name YLC444 \
  --fasta-index ref.fasta.fai --n-pieces 15 --min-len 10000 --max-len 100000 --seed 1002 \
  --output-vcf YLC444.pieces.vcf.gz
analyze_cnv_breakpoint_reads --bam-file YLC444.cram --vcf-file YLC444.pieces.vcf.gz --reference-fasta ref.fasta \
  --cushion 1500 --output-file YLC444.pieces.ann.vcf.gz --output-bam YLC444.pieces.ev.bam
snap_cnv_breakpoints --input-vcf YLC444.pieces.ann.vcf.gz --alignment YLC444.cram --reference-fasta ref.fasta \
  --output-vcf YLC444.pieces.snap.vcf.gz

training_prep_depth_cnv_pipeline \
  --call_vcf HG002.depth_cnv.vcf.gz \
  --truth_vcf GRCh38_HG2-T2TQ100-V1.1_stvar.ge500.vcf.gz \
  --benchmark_bed GRCh38_HG2-T2TQ100-V1.1_stvar.benchmark.bed \
  --train_chromosomes chr1 chr10 chr11 chr12 chr13 chr14 chr15 \
  --known_events YLC444.depth_cnv.vcf.gz ylc444.events.bed \
  --known_events YLC444.pieces.snap.vcf.gz ylc444.events.bed \
  --output_prefix hg002_depth_cnv

train_models_pipeline --train_dfs hg002_depth_cnv.train.h5 --test_dfs hg002_depth_cnv.test.h5 \
  --output_file_prefix cnv_filtering_model --vcf_type cnv_depth --gt_type approximate --decision_threshold 28
```

For `cnv_depth`, `train_models_pipeline` also stores `decision_threshold` in the pickle: the out-of-bag probability with
the best F1 on the training rows as a FilterVCF score (30 + 10 log10(p / (1 − p))), to pass as
`filter_variants_pipeline --decision_threshold`. `train_models_pipeline --decision_threshold <int>` stores the given
threshold instead.

`cnv_filtering_model_v2.0.0.pkl` was trained with these commands (threshold 28), on the paired-end pipeline's
calls: depth, split-read and discordant-pair candidates after snapping and `filter_nested_cnv_candidates`, for HG002
613168 (training chromosomes) and the YLC444 (chr21 DUP) / GSM704 (chrX DEL) events, with their pieces (15 YLC444 DUP
pieces, seed 1002; 13 GSM704 DEL pieces, seed 1012; 10–100 kb).

`cnv_filtering_model_se_v2.0.0.pkl` (single-end, threshold 28) was trained with the same commands on the single-end
pipeline's calls of HG002 434421 (training chromosomes), with the same YLC444 / GSM704 events and pieces. These come
from paired-end CRAMs, so before training their discordant-pair candidates were dropped and their pair evidence
(`CNV_*_PAIRS*`, `*_PAIRS_MEDIAN_SPAN`) set to 0, as in single-end calls. On the HG002 test chromosomes it raises
single-end deletion recall (≥ 500 bp) from 0.547 to 0.588 at precision 0.969 (v2.0.0: 0.977).

The same model filters the optional cn.mops candidates (`cnmops_cnv_candidates`, `CNV_SOURCE=cn.mops`): they have
no source feature of their own and are scored on their depth and read evidence like depth calls.

## Typical Workflows

### Training and Applying a Filtering Model

1. **Prepare training data**:
   ```bash
   training_prep_pipeline \
     --call_vcf calls.vcf.gz \
     --base_vcf truth.vcf.gz \
     --gt_type exact \
     --reference ref.fa \
     --hcr hcr.bed \
     --output_prefix training
   ```

2. **Train the model**:
   ```bash
   train_models_pipeline \
     --train_dfs training_train.h5 \
     --test_dfs training_test.h5 \
     --output_file_prefix model
   ```

3. **Apply filtering**:
   ```bash
   filter_variants_pipeline \
     --input_file new_calls.vcf.gz \
     --model_file model.pkl \
     --output_file filtered_calls.vcf.gz
   ```
## Dependencies

The filtering module depends on the following external tools: bcftools, samtools, GATK. RTG tools and picard
