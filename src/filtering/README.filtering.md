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
  [--vcf_type single_sample|deep_variant|cnv] \
  [--custom_annotations ANNOTATION1 ANNOTATION2] \
  [--verbosity INFO]
```

**Key Parameters**:
- `--train_dfs`: Training HDF5 files (output from training_prep_pipeline)
- `--test_dfs`: Test HDF5 files for model evaluation
- `--output_file_prefix`: Prefix for output .pkl (model) and .h5 (results) files
- `--gt_type`: Ground truth type - "exact" or "approximate" (default: exact)
- `--vcf_type`: VCF type - "single_sample", "deep_variant", or "cnv" (default: single_sample)
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

#### `training_prep_cnv_pipeline`

Prepares training data specifically for CNV filtering models.

**Purpose**: Generate labeled CNV training datasets by comparing CNV calls to truth set.

**Usage**:
```bash
training_prep_cnv_pipeline \
  --call_vcf cnv_calls.vcf.gz \
  --base_vcf cnv_truth.vcf.gz \
  --output_prefix cnv_training_data \
  [--hcr high_confidence.bed] \
  [--custom_annotations ANNOTATION1 ANNOTATION2] \
  [--train_fraction 0.25] \
  [--ignore_cnv_type] \
  [--skip_collapse] \
  [--verbosity INFO]
```

**Key Parameters**:
- `--call_vcf`: CNV call VCF file
- `--base_vcf`: CNV truth VCF file
- `--output_prefix`: Prefix for output HDF5 files
- `--train_fraction`: Fraction of CNVs for training, rest for testing (default: 0.25)
- `--ignore_cnv_type`: Ignore CNV type when matching to truth
- `--skip_collapse`: Skip collapsing variants before comparison

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
