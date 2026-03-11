# CLAUDE.md - ugbio_utils Development Guide

This file provides guidance for Claude Code and developers working with the ugbio_utils UV workspace.

## Project Overview

**ugbio_utils** is Ultima Genomics' modular bioinformatics toolkit organized as a UV workspace with 13 independent Python packages. Each package is independently versioned, tested, and deployed as a Docker container.

- **Current Version:** 1.20.0-0
- **Python Target:** 3.11+
- **Package Manager:** UV (not pip/conda)
- **License:** Apache-2.0
- **Total Codebase:** 281 Python files, 99 test files
- **Docker Registry:** AWS ECR + Docker Hub

> **⚠️ IMPORTANT:** Always activate the UV virtual environment before running Python scripts:
> ```bash
> source .venv/bin/activate  # OR use: uv run python script.py
> ```
> See [Environment Setup](#important-always-activate-virtual-environment) for details.

## Architecture

### Workspace Structure

```
src/
├── core/         (74 files) ugbio_core         - Foundation (ALL depend on this)
├── cnv/          (41 files) ugbio_cnv          - CNV calling (JALIGN, CNVpytor, FREEC)
├── comparison/   (11 files) ugbio_comparison   - Variant comparison (Truvari)
├── featuremap/   (21 files) ugbio_featuremap   - Feature extraction (VCF→Parquet)
├── filtering/    (50 files) ugbio_filtering    - ML filtering, SEC, training
├── mrd/          (6 files)  ugbio_mrd          - Minimal residual disease
├── ppmseq/       (6 files)  ugbio_ppmseq       - PPMSeq QC analysis
├── srsnv/        (20 files) ugbio_srsnv        - Single-read SNV calling
├── methylation/  (12 files) ugbio_methylation  - Methylation analysis
├── single_cell/  (8 files)  ugbio_single_cell  - Single-cell genomics
├── cloud_utils/  (4 files)  ugbio_cloud_utils  - AWS/GCS integration
├── omics/        (25 files) ugbio_omics        - AWS HealthOmics workflows
├── freec/        (2 files)  ugbio_freec        - FREEC CNV config
└── pypgx/        (0 files)  ugbio_pypgx        - Pharmacogenomics (stub)
```

### Module Dependency Hierarchy

**Level 0 (Foundation):**
- ugbio_core (no dependencies)
- ugbio_cloud_utils (no workspace dependencies)
- ugbio_freec (no workspace dependencies)

**Level 1 (Direct Core Dependencies):**
- ugbio_cnv, ugbio_comparison, ugbio_ppmseq, ugbio_methylation, ugbio_single_cell, ugbio_omics

**Level 2 (Secondary Dependencies):**
- ugbio_featuremap (depends: core, ppmseq)
- ugbio_filtering (depends: comparison → core)
- ugbio_srsnv (depends: core, ppmseq, featuremap)

**Level 3 (Complex Dependencies):**
- ugbio_mrd (depends: core, ppmseq, featuremap)

## Development Commands

### Environment Setup

```bash
# Install UV (required once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Full workspace with all packages and extras
uv sync --all-extras --all-packages

# Specific module only
uv sync --package ugbio-core
uv sync --package ugbio-filtering
```

### IMPORTANT: Always Activate Virtual Environment

**When working in ugbio_utils, ALWAYS activate the UV virtual environment first:**

```bash
# Navigate to ugbio_utils directory
cd /path/to/BioinfoResearch/VariantCalling/ugbio_utils

# Activate the virtual environment
source .venv/bin/activate

# Now run your Python scripts
python script.py
```

**Why this matters:**
- The `.venv` contains all installed bioinformatics packages (bcftools, samtools, pysam, etc.)
- Without activation, Python scripts will fail with `ModuleNotFoundError`
- All ugbio modules (ugbio_core, ugbio_filtering, ugbio_omics, etc.) are installed in this environment

**Alternative: Use `uv run` (no activation needed):**
```bash
# UV automatically uses the virtual environment
uv run python script.py
uv run pytest src/core/tests/
```

**Rare: If packages are missing after activation:**
```bash
# Only needed if you encounter ImportError after activation
# This reinstalls all packages (takes ~2-5 minutes)
uv sync --all-extras --all-packages
```

**Common scenarios:**
- ✓ `source .venv/bin/activate && python script.py` - Recommended
- ✓ `uv run python script.py` - Alternative (no activation)
- ✗ `python script.py` - Will fail without activation

### Testing

```bash
# Specific module tests (preferred)
uv run pytest src/core/tests/
uv run pytest src/filtering/tests/
uv run pytest src/cnv/tests/

uv run pytest --durations=0 src/ --ignore src/cnv/cnmops

# Specific test file
uv run pytest src/filtering/tests/unit/test_train_models_pipeline.py -v

# Docker-based testing
docker run --rm -v .:/workdir <image> run_tests /workdir/src/core
```

**Note:** Tests under `src/cnv/cnmops/` are not expected to pass and should be excluded from test runs.

### Linting & Formatting

```bash
# Pre-commit runs automatically on git commit, or manually:
uv run pre-commit install  # Install hooks
uv run pre-commit run --all-files  # Run all checks

# Ruff is configured via .ruff.toml (line length 120, Python 3.11 target)
```

## Development Workflow & Testing Requirements

### Adding New Features or Modifying Existing Code

**MANDATORY: All code changes MUST include tests and pass the full test suite.**

When adding features or modifying existing functionality:

1. **Write Tests First or Alongside Implementation**
   - Add unit tests for new functions/classes
   - Add system/integration tests for end-to-end workflows
   - Update existing tests if behavior changes

2. **Run Module-Specific Tests During Development**
   ```bash
   # Run tests for the module you're working on
   uv run pytest src/cnv/tests/ -v
   uv run pytest src/filtering/tests/ -v
   ```

3. **Run Full Module Test Suite Before Committing**
   ```bash
   # Ensure ALL tests pass, not just new ones
   uv run pytest src/cnv/tests/ -v --tb=short
   ```

4. **Verify Pre-commit Checks Pass**
   ```bash
   uv run pre-commit run --files <modified_files>
   ```

5. **Update Documentation**
   - Update docstrings for modified functions
   - Update module README if adding new features
   - Update CLAUDE.md if changing architecture or adding new patterns

### Test Coverage Guidelines

- **Unit tests** (`tests/unit/`): Test individual functions and classes in isolation
- **System tests** (`tests/system/`): Test complete workflows and integration
- **Minimum requirement**: New code must have corresponding tests
- **Best practice**: Aim for >80% code coverage on modified modules

### Example Workflow

```bash
# 1. Make code changes
vim src/cnv/ugbio_cnv/process_cnvs.py

# 2. Add/update tests
vim src/cnv/tests/unit/test_process_cnvs_unit.py
vim src/cnv/tests/system/test_process_cnvs.py

# 3. Run tests for your module
uv run pytest src/cnv/tests/ -v

# 4. Run pre-commit checks
uv run pre-commit run --files src/cnv/ugbio_cnv/process_cnvs.py

# 5. Commit only after all tests pass
git add src/cnv/ugbio_cnv/process_cnvs.py src/cnv/tests/
git commit -m "Add new feature to process_cnvs"
```

## Core Modules

### ugbio_core - Foundation Layer

### Building Docker Images

Docker images can be built via GitHub Actions workflow:

```bash
# Trigger Docker build for a specific module
gh workflow run "build-ugbio-member-docker.yml" --ref <branch-name> -f member=<module>

# Examples:
gh workflow run "build-ugbio-member-docker.yml" --ref patch-1.20.0-BIOIN-2648 -f member=cnv
gh workflow run "build-ugbio-member-docker.yml" --ref main -f member=core
gh workflow run "build-ugbio-member-docker.yml" --ref my-branch -f member=filtering

# Check build status
gh run list --workflow build-ugbio-member-docker.yml --limit 5

# View specific run details
gh run view <run-id>
```

**Available modules**: `core`, `cnv`, `comparison`, `featuremap`, `filtering`, `mrd`, `ppmseq`, `srsnv`, `methylation`, `single_cell`, `cloud_utils`, `omics`, `freec`

**Docker registries**:
- **AWS ECR** (internal): `337532070941.dkr.ecr.us-east-1.amazonaws.com/ugbio_<module>:<tag>`
- **Docker Hub** (public): `ultimagenomics/ugbio_<module>:<tag>`

## Coding Conventions
**Purpose:** Provides VCF processing, utilities, and infrastructure for all other modules.

**Key Components:**
- `vcfbed/vcftools.py` - Main VCF→DataFrame conversion via `get_vcf_df()`
- `exec_utils.py` - SimplePipeline wrapper for shell command execution
- `h5_utils.py` - HDF5 file I/O
- `stats_utils.py` - Statistical utilities (AUCPR metric, etc.)
- `logger.py` - Centralized logging (currently under refinement for DEBUG level control)
- `reports/` - Jupyter-based report generation
- `dna/`, `flow_format/`, `concordance/` - Specialized modules

**Entry Points:** 7
- Core processing: `annotate_contig`, `intersect_bed_regions`, `sorter_stats_to_mean_coverage`, `sorter_to_h5`, `convert_h5_to_json`, `collect_existing_metrics`, `generate_report`

**Optional Extras:**
- `[vcfbed]` - VCF processing (pybigwig, biopython, truvari, etc.)
- `[ml]` - Machine learning (scikit-learn, xgboost)
- `[reports]` - Report generation (papermill, jupyter, seaborn)
- `[parquet]` - Parquet support (pyarrow, fastparquet)

### ugbio_filtering - ML-Based Variant Filtering

**Purpose:** ML model training, variant filtering, and Systematic Error Correction (SEC).

**Key Pipelines:**
- `train_models_pipeline.py` (MAIN - currently under BIOIN-2620 development)
- `training_prep_pipeline.py` - Prepare training data from VCFs
- `training_prep_cnv_pipeline.py` - CNV-specific training (with new AUCPR metric)
- `filter_variants_pipeline.py` - Apply trained models

**Key Modules:**
- `transformers.py` - Feature extraction and transformation
- `variant_filtering_utils.py` - Core filtering utilities
- `multiallelics.py`, `spandel.py`, `blacklist.py` - Specialized filtering

**SEC (Systematic Error Correction):**
- `sec/error_correction_training.py` - Train error models
- `sec/correct_systematic_errors.py` - Apply corrections
- `sec/assess_sec_concordance.py` - Validate SEC

**Entry Points:** 8
- `train_models_pipeline`, `training_prep_pipeline`, `training_prep_cnv_pipeline`, `filter_variants_pipeline`, `error_correction_training`, `merge_conditional_allele_distributions`, `assess_sec_concordance`, `correct_systematic_errors`

### ugbio_featuremap - Feature Extraction & Pileup

**Purpose:** Convert VCFs to feature maps for machine learning.

**Key Modules:**
- `featuremap_to_dataframe.py` - Main VCF→Parquet with multi-sample support (DATA-8973)
- `featuremap_xgb_prediction.py` - XGBoost integration
- `create_somatic_featuremap.py` - Somatic feature maps
- `integrate_mpileup_to_sfm.py` - Mpileup integration

**Entry Points:** 7
- `featuremap_to_dataframe`, `filter_featuremap`, `add_aggregate_params_and_xgb_score_to_pileup_featuremap`, `create_somatic_featuremap`, `integrate_mpileup_to_sfm`, `somatic_featuremap_fields_transformation`

### ugbio_cnv - Copy Number Variation

**Purpose:** CNV detection and analysis (JALIGN, CNVpytor, FREEC, BicSeq2).

**Key Modules:**
- `jalign.py` - JALIGN alignment algorithm
- `run_jalign.py` - JALIGN execution
- `analyze_cnv_breakpoint_reads.py` - Read analysis for CNV breakpoints (RECENTLY REFINED - requires consistent insert size)
- `run_cnvpytor.py` - CNVpytor integration
- `process_cnvs.py` - Process CNV calls from cn.mops/FREEC
- `combine_cnmops_cnvpytor_cnv_calls.py` - Multi-tool fusion
- `bicseq2_post_processing.py`, `annotate_FREEC_segments.py` - Post-processing

**Entry Points:** 10
- `run_jalign`, `analyze_cnv_breakpoint_reads`, `process_cnvs`, `run_cnvpytor`, `combine_cnmops_cnvpytor_cnv_calls`, `combine_cnv_vcfs`, `filter_dup_cnmmops_cnv_calls`, `annotate_vcf_with_regions`, `plot_cnv_results`, `annotate_FREEC_segments`

### Other Modules

**ugbio_comparison** (11 files):
- Truvari-based SV comparison
- Produces concordance analysis
- Entry points: `run_comparison_pipeline`, `sv_comparison_pipeline`

**ugbio_srsnv** (20 files):
- Single-read SNV detection, training, inference with SHAP
- Entry points: `srsnv_training`, `srsnv_inference`, `srsnv_report`

**ugbio_mrd** (6 files):
- Minimal residual disease detection
- Entry points: `generate_synthetic_signatures`, `generate_report`

**ugbio_omics** (25 files):
- AWS HealthOmics cost analysis, performance monitoring
- Entry points: `compare_cromwell_omics`, `get_omics_logs`, `performance`, etc.

**ugbio_methylation, ugbio_single_cell, ugbio_ppmseq, ugbio_cloud_utils, ugbio_freec**:
- Specialized analysis domains
- See pyproject.toml for entry points

## Architectural Patterns

### 1. SimplePipeline Wrapper

All shell command execution uses `exec_utils.print_and_execute()`:
```python
from ugbio_core.exec_utils import print_and_execute
print_and_execute("bcftools view -h file.vcf")
```

### 2. VCF Processing Pipeline

All VCF operations flow through `vcftools.get_vcf_df()`:
```python
from ugbio_core.vcfbed.vcftools import get_vcf_df
df = get_vcf_df("file.vcf", sample_id=0, chromosome="chr1")
```

### 3. HDF5 Data Format

Intermediate results stored as HDF5 files:
```python
from ugbio_core.h5_utils import read_hdf
df = read_hdf("file.h5", columns_subset=["feature1", "label"])
```

### 4. ML Model Training

Training pattern used in `filtering/train_models_pipeline.py`:
1. Load training data (HDF5)
2. Train model + transformer
3. Evaluate on test set
4. Save as pickle/dill

### 5. Centralized Logging

All modules use:
```python
from ugbio_core.logger import logger
logger.info("message")
logger.debug("debug info")
```

**Note:** Currently refining DEBUG level control (BIOIN-2620).

### 6. Entry Point Scripts

Each module defines entry points in pyproject.toml:
```toml
[project.scripts]
run_tests = "pytest:main"
train_models_pipeline = "ugbio_filtering.train_models_pipeline:main"
```

All implement pattern: `parse_args()` → `run(argv)`

### 7. Optional Dependencies

Lightweight installation possible:
```bash
pip install ugbio_core[vcfbed]  # VCF processing only
pip install ugbio_core[ml]      # ML features
pip install ugbio_core[reports] # Report generation
```

## Testing

### Test Organization

99 test files organized by module:
- **core:** 26 test files (unit + system)
- **filtering:** 15 test files
- **cnv:** 17 test files
- **omics:** 10 test files
- **featuremap:** 8 test files
- **srsnv:** 8 test files
- **comparison:** 5 test files
- **Others:** 10 test files combined

### Test Resources (Git LFS)

Large test files tracked via Git LFS:
- JALIGN CRAM files
- XGBoost model JSON files
- Pileup files
- VCF/BAM test data

Excluded from pre-commit checks to preserve integrity.

### Running Tests

```bash
# Module-specific (preferred)
uv run pytest src/core/tests/

# All modules
uv run pytest --durations=0 src/ --ignore src/cnv/cnmops

# Docker-based
docker run --rm -v .:/workdir <image> run_tests /workdir/src/<module>
```

## Code Style & Quality

### Ruff Configuration (.ruff.toml)

- **Line Length:** 120 characters
- **Target Python:** 3.11
- **String Quotes:** Double quotes (enforced)
- **Import Sorting:** isort integration
- **Auto-fix:** Enabled

### Linting Rules

**Enabled:** E, F, W, B, PD, NPY, C4, C90, I, UP, ASYNC, S, N, A, COM, PIE, FBT, PL

**Disabled for Tests:** S101 (assertions), PLR (pylint rules), B (bugbear), ERA, S (security)

**Pylint Max Args:** 10 per function

### Pre-commit Hooks

- Ruff linting + formatting
- Trailing whitespace, end-of-file fixers
- YAML/JSON validation
- Large file checks (excludes test resources)

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## CI/CD Pipeline

### GitHub Actions (.github/workflows/ci.yml)

1. **Pre-commit Check** - Ruff linting/formatting on all files
2. **Security Scans** - Trivy for Docker config and filesystem
3. **Parallel Tests** - All 13 modules tested matrix-style
4. **Docker Building** - Multi-stage builds for each module

### Docker Architecture

**Base Image:** `ugbio_base:1.7.0`
- Python 3.11-bullseye
- Pre-installed: bcftools, samtools, bedtools, bedops

**Module Images:**
- Multi-stage build (Build + Runtime)
- Runtime: FROM ugbio_base + module-specific tools

**Registry:**
- Public: Docker Hub

## Important Patterns & Conventions

### String Formatting
```python
# YES - Double quotes (enforced)
message = "Hello world"

# NO - Single quotes
message = 'Hello world'
```

### Docstrings
```python
# Triple double quotes
def my_function():
    """
    Function description.

    Returns
    -------
    result_type
        Description
    """
```

### Entry Point Pattern
```python
def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # Add arguments
    return ap.parse_args(argv)

def run(argv: list[str]):
    args = parse_args(argv[1:])
    # Implementation

if __name__ == "__main__":
    import sys
    run(sys.argv)
```

### Module Usage
```python
# Logger
from ugbio_core.logger import logger
logger.info("Processing...")

# VCF
from ugbio_core.vcfbed.vcftools import get_vcf_df
df = get_vcf_df("file.vcf")

# Execution
from ugbio_core.exec_utils import print_and_execute
print_and_execute("bcftools view file.vcf")
```

## Key Files

| File | Purpose |
|------|---------|
| `src/core/ugbio_core/vcfbed/vcftools.py` | Core VCF→DataFrame engine |
| `src/core/ugbio_core/exec_utils.py` | Shell execution wrapper |
| `src/core/ugbio_core/logger.py` | Centralized logging |
| `src/filtering/ugbio_filtering/train_models_pipeline.py` | Main ML training pipeline |
| `src/filtering/ugbio_filtering/transformers.py` | Feature transformers |
| `src/cnv/ugbio_cnv/jalign.py` | JALIGN alignment |
| `src/cnv/ugbio_cnv/process_cnvs.py` | CNV processing and VCF generation |
| `src/cnv/ugbio_cnv/cnv_bed_format_utils.py` | CNV VCF writing utilities |
| `src/cnv/ugbio_cnv/combine_cnv_vcf_utils.py` | CNV VCF merging and aggregation |
| `src/cnv/ugbio_cnv/cnv_vcf_consts.py` | CNV VCF field definitions |
| `src/featuremap/ugbio_featuremap/featuremap_to_dataframe.py` | Feature extraction |
| `pyproject.toml` | Root workspace config |
| `.ruff.toml` | Linting rules |
| `.pre-commit-config.yaml` | Pre-commit hooks |
| `.github/workflows/ci.yml` | CI/CD pipeline |

## Troubleshooting

### ModuleNotFoundError (Most Common)

**Problem:** `ModuleNotFoundError: No module named 'ugbio_core'` or similar

**Solution:** Activate the virtual environment first:
```bash
cd /path/to/BioinfoResearch/VariantCalling/ugbio_utils
source .venv/bin/activate
python your_script.py
```

**Alternative:** Use `uv run` instead:
```bash
uv run python your_script.py
```

### UV Hardlink Issues
```bash
rm -rf .venv
uv sync --all-extras --all-packages
```

### Python Interpreter in VSCode
- Check: `.venv/bin/python` is selected
- Dev containers: Python 3.11 from image

### Git LFS Test Resources
```bash
git lfs install
git lfs pull
```


## Related Documentation

- **Parent Project:** `/BioinfoResearch/CLAUDE.md` - Overall research toolkit structure
- **VariantCalling:** `../CLAUDE.md` - Production pipeline documentation
- **Module READMEs:** `src/<module>/README.<module>.md` - Module-specific guides

## Quick Reference

### Commands
```bash
# ALWAYS activate virtual environment first
source .venv/bin/activate

# Setup (rare - only if packages missing)
uv sync --all-extras --all-packages

# Test
uv run pytest src/core/tests/
# OR with activated environment:
pytest src/core/tests/

# Lint
uv run pre-commit run --all-files

# Run pipeline (requires activated environment)
python -m ugbio_filtering.train_models_pipeline --train_dfs file.h5 --test_dfs file2.h5 --output_file_prefix out

# Alternative: Use uv run (no activation needed)
uv run python -m ugbio_filtering.train_models_pipeline --train_dfs file.h5 --test_dfs file2.h5 --output_file_prefix out
```

### Module Imports
```python
from ugbio_core.logger import logger
from ugbio_core.vcfbed.vcftools import get_vcf_df
from ugbio_core.exec_utils import print_and_execute
from ugbio_filtering import train_models_pipeline
```

### Entry Points
```bash
train_models_pipeline --train_dfs train.h5 --test_dfs test.h5 --output_file_prefix model
filter_variants_pipeline --input_vcf calls.vcf --model_file model.pkl --output_vcf filtered.vcf
run_jalign --input_bam reads.bam --ref_genome ref.fa --output_dir results
```
