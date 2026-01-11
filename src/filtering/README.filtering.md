# ugbio_filtering

This module includes filtering python scripts and utils for bioinformatics pipelines. It provides tools for variant filtering, model training, systematic error correction (SEC), and quality control of variant calls.

## Installation

To install the filtering module with all dependencies:

```bash
uv sync --package ugbio_filtering
```

## CLI Scripts

The filtering module provides several command-line tools for different stages of the variant filtering pipeline:

### Variant Filtering

#### `filter_variants_pipeline`

Applies machine learning-based filtering to VCF files after GATK variant calling.

**Purpose**: Filter variants using trained models, blacklists, and custom annotations to improve callset quality.

**Usage**:
```bash
filter_variants_pipeline \
  --input_file input.vcf.gz \
  --output_file filtered.vcf.gz \
  --model_file model.pkl \
  [--blacklist blacklist.pkl] \
  [--custom_annotations ANNOTATION1 ANNOTATION2] \
  [--decision_threshold 30] \
  [--blacklist_cg_insertions] \
  [--treat_multiallelics --ref_fasta reference.fa] \
  [--recalibrate_genotype] \
  [--overwrite_qual_tag] \
  [--limit_to_contigs chr1 chr2]
```

**Key Parameters**:
- `--input_file`: Input VCF file (requires .tbi index)
- `--output_file`: Output VCF file with filtering annotations
- `--model_file`: Pickle file containing trained XGBoost model and transformer
- `--blacklist`: Optional pickle file with blacklisted regions
- `--decision_threshold`: Score threshold for filtering variants (default: 30)
- `--blacklist_cg_insertions`: Flag to filter CCG/GGC insertions
- `--treat_multiallelics`: Apply special handling for multiallelic sites
- `--recalibrate_genotype`: Allow model to re-call genotypes

#### `filter_low_af_ratio_to_background`

Filters variants based on allele frequency ratio to background.

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
- `--reference`: Reference FASTA file prefix (requires .fai and .sdf)
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

### Systematic Error Correction (SEC)

#### `error_correction_training`

Collects statistics from gVCF files for systematic error correction model training.

**Purpose**: Gather allele count statistics at ground truth positions to build conditional allele distributions.

**Usage**:
```bash
error_correction_training \
  --relevant_coords regions.bed \
  --ground_truth_vcf truth.vcf.gz \
  --gvcf_file sample.g.vcf.gz \
  --sample_id SAMPLE_NAME \
  --output_file allele_distributions.tsv
```

**Key Parameters**:
- `--relevant_coords`: BED file with genomic regions to analyze
- `--ground_truth_vcf`: VCF file with true genotypes (tabix indexed)
- `--gvcf_file`: gVCF file with raw read information
- `--sample_id`: Sample identifier
- `--output_file`: Output TSV file with allele distributions

#### `merge_conditional_allele_distributions`

Combines conditional allele distributions from multiple samples.

**Purpose**: Aggregate statistics from multiple samples to create a comprehensive error correction model.

**Usage**:
```bash
merge_conditional_allele_distributions \
  --conditional_allele_distribution_files file_list.txt \
  --output_prefix merged_model
```

**Key Parameters**:
- `--conditional_allele_distribution_files`: Text file containing paths to individual distribution files (one per line)
- `--output_prefix`: Prefix for output pickle files (per chromosome)

#### `correct_systematic_errors`

Applies systematic error correction to filter false positive calls.

**Purpose**: Use conditional allele distribution models to identify and filter systematic sequencing errors.

**Usage**:
```bash
correct_systematic_errors \
  --relevant_coords regions.bed \
  --model model_chr*.pkl \
  --gvcf sample.g.vcf.gz \
  --output_file corrected.vcf.gz \
  [--strand_enrichment_pval_thresh 0.00001] \
  [--lesser_strand_enrichment_pval_thresh 0.05]
```

**Key Parameters**:
- `--relevant_coords`: BED file with regions to analyze
- `--model`: Pickle file(s) with conditional allele distributions (supports glob patterns)
- `--gvcf`: gVCF file with raw read information
- `--output_file`: Output VCF, pickle, or BED file
- `--strand_enrichment_pval_thresh`: P-value threshold for strand bias (default: 0.00001)
- `--lesser_strand_enrichment_pval_thresh`: Lesser strand bias threshold (default: 0.05)

#### `assess_sec_concordance`

Evaluates the performance of systematic error correction.

**Purpose**: Compare variant calls before and after SEC to ground truth and generate accuracy metrics.

**Usage**:
```bash
assess_sec_concordance \
  --concordance_h5_input comparison.h5 \
  --genome_fasta reference.fa \
  --raw_exclude_list raw_blacklist.bed \
  --sec_exclude_list sec_blacklist.bed \
  --hcr high_confidence.bed \
  --output_prefix evaluation \
  [--dataset_key all] \
  [--ignore_genotype]
```

**Key Parameters**:
- `--concordance_h5_input`: HDF5 file with variant comparison results
- `--genome_fasta`: Reference genome FASTA file
- `--raw_exclude_list`: BED file with raw exclude list (SEC input)
- `--sec_exclude_list`: BED file with SEC call types (SEC output)
- `--hcr`: High confidence regions BED file
- `--output_prefix`: Prefix for output statistics and error analysis files
- `--ignore_genotype`: Ignore genotype when comparing to ground truth

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

### Systematic Error Correction Pipeline

1. **Collect statistics per sample**:
   ```bash
   error_correction_training \
     --relevant_coords regions.bed \
     --ground_truth_vcf truth.vcf.gz \
     --gvcf_file sample.g.vcf.gz \
     --sample_id SAMPLE \
     --output_file sample_dist.tsv
   ```

2. **Merge distributions from multiple samples**:
   ```bash
   merge_conditional_allele_distributions \
     --conditional_allele_distribution_files samples.txt \
     --output_prefix sec_model
   ```

3. **Apply error correction**:
   ```bash
   correct_systematic_errors \
     --relevant_coords regions.bed \
     --model sec_model_*.pkl \
     --gvcf sample.g.vcf.gz \
     --output_file corrected.vcf.gz
   ```

4. **Evaluate results**:
   ```bash
   assess_sec_concordance \
     --concordance_h5_input comparison.h5 \
     --genome_fasta ref.fa \
     --raw_exclude_list raw.bed \
     --sec_exclude_list sec.bed \
     --hcr hcr.bed \
     --output_prefix evaluation
   ```

## Development

### Running Tests

Run tests using pytest in the dev container:
```bash
uv run pytest src/filtering/tests/
```

Or using Docker:
```bash
docker run --rm -v .:/workdir <docker_image> run_tests /workdir/src/filtering
```

### Building the Docker Image

```bash
docker build -t ugbio_filtering:latest -f src/filtering/Dockerfile .
```

## Dependencies

The filtering module depends on:
- `ugbio_core`: Core utilities for VCF processing
- `ugbio_comparison`: Variant comparison utilities
- `pickle-secure`: Secure pickle operations
- `biopython`: Biological sequence manipulation
- `dill`: Enhanced pickling
- External tools: bcftools, samtools, GATK (included in Docker image)

## Additional Notes

- All VCF input files must be bgzip-compressed and tabix-indexed (.tbi)
- Models are stored as pickle files containing XGBoost classifiers and preprocessing transformers
- Custom annotations allow extending the feature set beyond standard VCF fields
- SEC (Systematic Error Correction) is particularly effective for ultra-high accuracy sequencing platforms
