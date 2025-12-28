# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UGBio Utils is a bioinformatics pipeline toolkit organized as a UV workspace with modular components. Each module in `src/` is a separate Python package with its own Docker container for cloud deployment. The repository is public on GitHub, with Docker images published to both AWS ECR (internal) and Docker Hub (public).

**The project uses pre-commit hooks** for code quality checks (Ruff linting/formatting). These run automatically on `git commit`.

## Architecture

### Workspace Structure
- **UV Workspace**: Monorepo with multiple packages under `src/`, managed by UV package manager
- **Core Module**: `ugbio_core` provides VCF processing (`vcfbed/`), statistics utilities, and shared infrastructure. All other modules depend on this.
- **Pipeline Modules**: comparison (Truvari SV analysis), featuremap (VCF→Parquet conversion), filtering, CNV calling, methylation analysis, MRD, ppmseq, srsnv, single_cell, freec, omics
- **Cloud Integration**: AWS HealthOmics workflow management, Cromwell cost comparison

### Module Pattern
Each module follows a consistent structure:
```
src/<module>/
├── Dockerfile                    # Multi-stage build from ugbio_base
├── pyproject.toml               # UV workspace member config
├── ugbio_<module>/              # Python source code
├── tests/                       # pytest test suite
└── README.<module>.md           # Module documentation
```

### Docker Architecture
- **`ugbio_base`**: Foundation image (Python 3.11-bullseye) with bcftools, samtools, bedtools, bedops
- **Module Images**: Multi-stage builds that first build wheels, then install on ugbio_base with additional tools (e.g., GATK, Truvari)

## Development Commands

### Environment Setup
```bash
# Install UV first (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install full workspace with all packages
uv sync --all-extras --all-packages

# Install specific module only
uv sync --package ugbio-<module>

# Install pre-commit hooks (Ruff formatting/linting)
uv run pre-commit install
```

### Testing
```bash
# Run tests for a specific module (preferred in dev container)
uv run pytest src/<module>/tests/

# Run tests for core functionality
uv run pytest src/core/tests/

# Run all tests (excluding cnmops)
uv run pytest --durations=0 src/ --ignore src/cnv/cnmops

# Docker-based testing
docker run --rm -v .:/workdir <image> run_tests /workdir/src/<module>
```

### Linting & Formatting
```bash
# Pre-commit runs automatically on commit, or manually:
uv run pre-commit run --all-files

# Ruff is configured via .ruff.toml (line length 120, Python 3.11 target)
```

### Dev Containers
```bash
# AWS ECR login (required before pulling images)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 337532070941.dkr.ecr.us-east-1.amazonaws.com

# Open in VSCode: F1 → "Dev Containers: Open Folder in Container"
# Choose specific module container (CNV/comparison/etc.)
```

## Coding Conventions

### Python Code Style

- **String Quotes**: Always use double quotes for strings
- **Docstrings**: Use triple double quotes for docstrings and multi-line strings
- **Line Length**: 120 characters (enforced by Ruff)
- **Python Version**: Target Python 3.11
- **Linting**: Ruff configuration in [.ruff.toml](.ruff.toml) includes:
  - pycodestyle (E, W)
  - pyflakes (F)
  - flake8-bugbear (B)
  - pandas-vet (PD)
  - NumPy-specific rules (NPY)
  - isort (I) for import sorting
  - pylint (PL) with max 10 arguments per function

### Testing Conventions

- **Framework**: Use `pytest` for all tests
- **Test Location**: Place tests in `tests/` folder within each module
- **Mocking**: Mock external bioinformatics tools (bcftools, samtools, truvari, etc.) using `@patch`

```python
from unittest.mock import patch

@patch("subprocess.run")
@patch("ugbio_core.vcfbed.vcftools.get_vcf_df")
def test_my_function(mock_vcf_df, mock_run):
    # Test implementation
    pass
```

- **Entry Point**: Each module must have `run_tests = "pytest:main"` in `pyproject.toml`
- **Test Execution**: Run tests in dev containers/dockers where bioinformatics tools are available

## Key Technical Patterns

### Pipeline Execution

- **SimplePipeline**: Core framework in `ugbio_core` for executing shell commands
- **Parallel Processing**: Region-based chunking for large genomics files (300Mbp default chunks)
- **Truvari Integration**: SV comparison uses `--passonly` flags (conditional via `ignore_filter` parameter)

### Dependency Management
- Modules declare workspace dependencies in `pyproject.toml`:
```toml
[tool.uv.sources]
ugbio_core = {workspace = true}
```
- Each module must include `run_tests = "pytest:main"` script for CI

### AWS/Cloud Integration
```python
from ugbio_omics import compare_cromwell_omics, get_omics_log
# Cost comparison between Cromwell and AWS HealthOmics
# Performance monitoring via CloudWatch logs
```

## Critical Code Locations

- **Core VCF Logic**: [src/core/ugbio_core/vcfbed/vcftools.py](src/core/ugbio_core/vcfbed/vcftools.py)
- **Pipeline Framework**: [src/core/ugbio_core/exec_utils.py](src/core/ugbio_core/exec_utils.py), [src/core/ugbio_core/vcf_utils.py](src/core/ugbio_core/vcf_utils.py)
- **SV Comparison**: [src/comparison/ugbio_comparison/sv_comparison_pipeline.py](src/comparison/ugbio_comparison/sv_comparison_pipeline.py)
- **Parallel VCF Processing**: [src/featuremap/ugbio_featuremap/featuremap_to_dataframe.py](src/featuremap/ugbio_featuremap/featuremap_to_dataframe.py)

## Adding New Modules

When creating a new module in `src/`:

1. **Create standard structure**: Dockerfile, pyproject.toml, README.<module>.md, ugbio_<module>/, tests/
2. **pyproject.toml requirements**:
   - Include `[build-system]` with setuptools backend
   - Add `run_tests = "pytest:main"` script
   - Declare workspace dependencies via `[tool.uv.sources]`
3. **Dockerfile**: Build from `ugbio_base` using multi-stage pattern (see existing modules)
4. **Put common code in ugbio_core**: Check if functionality already exists; if new code is reusable, add it to core

## CI/CD

- **ci.yml**: Runs pre-commit (Ruff), Trivy security scans, and pytest for all modules
- **build-ugbio-base-docker.yml**: Builds foundational `ugbio_base` image
- **build-ugbio-member-docker.yml**: Builds module-specific Docker images
- **docker-build-push.yml**: Reusable workflow for scanning, building, and pushing images

## Common Gotchas

- If `uv` has hardlink issues, delete `.venv` and run `uv sync` again
- Ensure correct Python interpreter in VSCode (.venv/bin/python)
- Dev container `devcontainer.json` must not mount non-existent directories
- WDL validation: Use `miniwdl check "${file}"` (available as VS Code task)
- Version managed across workspace: see root `pyproject.toml` for current version (1.16.3-0)
