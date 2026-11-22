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
- `--decision_threshold`: Score threshold for filtering variants (default: the threshold stored with the model, else 30)
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
  [--vcf_type single_sample|deep_variant|cnv] \
  [--custom_annotations ANNOTATION1 ANNOTATION2] \
  [--verbosity INFO]
```

**Key Parameters**:
- `--train_dfs`: Training HDF5 files (output from training_prep_pipeline)
- `--test_dfs`: Test HDF5 files for model evaluation
- `--output_file_prefix`: Prefix for output .pkl (model) and .h5 (results) files
- `--gt_type`: Ground truth type - "exact" or "approximate" (default: exact)
- `--vcf_type`: VCF type - "single_sample", "deep_variant", or "cnv" (depth CNV caller; default: single_sample)
- `--custom_annotations`: Additional INFO annotations to include in training
- `--decision_threshold`: cnv only; FilterVCF threshold stored with the model (default: the out-of-bag score with the
  best F1)

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

#### `training_prep_cnv_pipeline`

Prepares training data for the depth CNV caller's filter (`--vcf_type cnv`): labels the calls of a sample with
truth (e.g. HG002) and writes train / test HDF5 files, split by chromosome.

```bash
# labels the calls >= 500 bp in the benchmark BED with truvari: matched 1, unmatched 0
training_prep_cnv_pipeline \
  --call_vcf HG002.depth_cnv.vcf.gz \
  --truth_vcf GRCh38_HG2-T2TQ100-V1.1_stvar.ge500.vcf.gz \
  --benchmark_bed GRCh38_HG2-T2TQ100-V1.1_stvar.benchmark.bed \
  --train_chromosomes chr1 chr10 chr11 chr12 chr13 chr14 chr15 \
  --output_prefix hg002_depth_cnv

# the threshold is stored in the pickle; without --decision_threshold it is chosen by the best F1
train_models_pipeline --train_dfs hg002_depth_cnv.train.h5 --test_dfs hg002_depth_cnv.test.h5 \
  --output_file_prefix cnv_filtering_model --vcf_type cnv --gt_type approximate --decision_threshold 28
```

Optional: samples with known long events (e.g. cell lines) can add training rows. The caller gives one call per event,
so a long event alone adds one row; split_cnv_into_pieces (BioinfoResearch) cuts 10-100 kb pieces of the events and
writes them with their own truth VCF and benchmark BED. After evidence and refinement as in the pipeline, the pieces are
labelled with the same command against that truth VCF and benchmark BED, and their training table is added to
`--train_dfs`.

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
