# UGBio Utils - AI Agent Instructions

## Architecture Overview

This is a bioinformatics pipeline toolkit organized as a UV workspace with modular components. Each module in `src/` is a separate Python package with its own Docker container for cloud deployment.
The GitHub repository is publicly shared. All members' Docker images are published internally to UG private AWS ECR, and also to public Docker Hub.

### Core Components

- **`ugbio_core`**: Foundation package providing VCF processing (`vcfbed/`), statistics utilities, and shared infrastructure. All other modules depend on this.
- **Pipelines**: comparison (Truvari SV analysis), featuremap (VCF→Parquet conversion), filtering, CNV calling, methylation analysis
- **Cloud Integration**: AWS HealthOmics workflow management, Cromwell cost comparison, distributed processing

### Module Structure Pattern
```
src/<module>/
├── Dockerfile                    # Multi-stage build from ugbio_base
├── pyproject.toml               # UV workspace member config
├── ugbio_<module>/              # Python source code
├── tests/                       # pytest test suite
└── README.<module>.md           # Module documentation
```

## Development Workflows

### Environment Setup
```bash
# Use UV for all dependency management
uv sync --all-extras --all-packages  # Full workspace
uv sync --package ugbio-<module>     # Single module

# Dev containers preferred for testing (includes bioinformatics tools)
# F1 → "Dev Containers: Open Folder in Container"
# Choose specific module container (CNV/comparison/etc.)
```

### Testing Strategy
```bash
# In dev container or with proper Python env:
uv run pytest src/<module>/tests/   # Module tests
uv run pytest src/core/tests/       # Core functionality

# Docker-based testing:
docker run --rm -v .:/workdir <image> run_tests /workdir/src/<module>
```

### Build Process
- **Local Development**: UV handles Python dependencies and virtual environments
- **Production**: Multi-stage Docker builds using `ugbio_base` image containing bcftools, samtools, GATK
- **Dependencies**: Modules declare workspace dependencies via `[tool.uv.sources]`

## Project-Specific Conventions

### Pipeline Architecture
- **SimplePipeline**: Core execution framework for shell commands
- **Truvari Integration**: SV comparison using `--passonly` flags (conditional via `ignore_filter`)
- **Parallel Processing**: Region-based chunking for large genomics files (300Mbp default)

### Testing Requirements
- Tests use `pytest` framework.
```python
# Mock external tools (bcftools, truvari, etc.)
@patch("subprocess.run")
@patch("ugbio_core.vcfbed.vcftools.get_vcf_df")

```

### AWS/Cloud Patterns
```python
# HealthOmics workflow analysis
from ugbio_omics import compare_cromwell_omics, get_omics_log
# Cost comparison between Cromwell and AWS HealthOmics
# Performance monitoring via CloudWatch logs
```

## Critical File Locations

- **Core VCF Logic**: `src/core/ugbio_core/vcfbed/vcftools.py`
- **Pipeline Framework**: `src/core/ugbio_core/exec_utils.py`, `src/core/ugbio_core/vcf_utils.py`
- **SV Comparison**: `src/comparison/ugbio_comparison/sv_comparison_pipeline.py`
- **Parallel VCF Processing**: `src/featuremap/ugbio_featuremap/featuremap_to_dataframe.py`

## Key Tools & Commands

- **WDL Validation**: `miniwdl check "${file}"` (via VS Code tasks)
- **Pre-commit**: `uv run pre-commit install` (Ruff formatting/linting)
- **Container Login**: `aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 337532070941.dkr.ecr.us-east-1.amazonaws.com`

When adding new modules, follow the established pattern: create pyproject.toml with `run_tests = "pytest:main"` script, add workspace dependency declarations, and ensure Dockerfile builds from appropriate base image.

## GitHub Actions Review
- **ci.yml**: Runs continuous integration checks, likely including linting, testing, and static analysis for all modules.
- **build-ugbio-base-docker.yml**: Builds the `ugbio_base` Docker image, which serves as the foundation for all module containers.
- **build-ugbio-member-docker.yml**: Builds Docker images for each member/module in the workspace, supporting modular deployment.
- **docker-build-push.yml**: Common inherited workflow that automates scanning, building and pushing Docker images to a container registry (e.g., AWS ECR). Cn be called by other internal/external workflows.
- **publish.yml**: Handles publishing of packages or images, possibly to PyPI or a Docker registry.
- **set-dev-version.yml**: Automates version bumping or tagging for development releases.

These workflows ensure code quality, automate builds, and streamline deployment for both Python and Docker-based components. For details or customization, review the YAML files in `.github/workflows/`.
