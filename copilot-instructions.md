# Copilot Instructions for ugbio-utils

## Project Overview
`ugbio-utils` is a modular toolkit for bioinformatics pipelines, organized as a UV workspace. Each module ("member") in `src/` is a standalone Python package, typically with its own Docker container for cloud deployment. The project supports workflows for VCF processing, CNV calling, methylation analysis, and more, with a shared core for common utilities.
The github repository is publicly shared. All members' Docker images are published internally to UG private AWS ECR, and also to public Docker Hub.

## Virtual Environment Setup
- **Dependency Management:** Uses [uv](https://docs.astral.sh/uv/) for Python package management and virtual environments.
- **Install uv:**
  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Open a new terminal after installation
  ```
- **Sync All Packages:**
  ```sh
  uv sync --all-extras --all-packages
  # Creates a single virtual environment under .venv/
  ```
- **Sync a Specific Member:**
  ```sh
  uv sync --package ugbio-<member>
  ```

## Members (Modules) Structure
Each member is located under `src/<member>/` and should include:
- `Dockerfile` (multi-stage, based on `ugbio_base`)
- `pyproject.toml` (with `[build-system]`, `[project.scripts]`, and `[tool.uv.sources]` if needed)
- `ugbio_<member>/` (Python source code)
- `tests/` (pytest test suite)
- `README.<member>.md` (module documentation)

## Test Structure
- Tests are located in each member's `tests/` folder.
- Every change in code should be accompanied by relevant tests.
- Tests use `pytest` framework.
- tests resources (e.g., test data files) should be placed in `tests/test_resources/` and are managed via git-lfs.
- Run tests inside the dev container or with the correct Python environment:
  ```sh
  uv run pytest src/<member>/tests/
  ```
- Or via Docker:
  ```sh
  docker run --rm -v .:/workdir <docker image> run_tests /workdir/src/<member>
  ```

## Pre-commit Hooks
- Install pre-commit tools:
  ```sh
  uv run pre-commit install
  ```
- Hooks (e.g., Ruff) will run automatically on each commit.
- VSCode users: install the Ruff extension for best experience.

## Dev Containers
- Recommended for development: open the repo in a dev container for all tools pre-installed.
- Each member can have its own `.devcontainer/<member>/devcontainer.json`.
- To open in VSCode: F1 → "Dev Containers: Open Folder in Container..."

## Adding a New Member
1. Create a new folder under `src/` with the required structure.
2. Base Dockerfile on `ugbio_base`.
3. In `pyproject.toml`, include `[build-system]`, `[project.scripts]`, and `[tool.uv.sources]` as needed.
4. Optionally, add a devcontainer config under `.devcontainer/<member>/`.

## General Guidelines
- Use `ugbio_core` for shared utilities; add new common code there.
- Follow the established module structure and conventions.
- For cloud workflows, see AWS HealthOmics and Cromwell integration patterns in the codebase.

## Troubleshooting
- If `uv` or the environment breaks, delete `.venv` and run `uv sync` again.
- Ensure the correct Python interpreter is selected in VSCode (should point to `.venv`).
- Check that all mount points in `devcontainer.json` exist on your system.

---
For more details, see the main `README.md` and module-specific documentation in each `src/<member>/README.<member>.md`.

## GitHub Actions Review

This repository includes several GitHub Actions workflows under `.github/workflows/` to automate CI/CD, Docker builds, and publishing:

- **ci.yml**: Runs continuous integration checks, likely including linting, testing, and static analysis for all modules.
- **build-ugbio-base-docker.yml**: Builds the `ugbio_base` Docker image, which serves as the foundation for all module containers.
- **build-ugbio-member-docker.yml**: Builds Docker images for each member/module in the workspace, supporting modular deployment.
- **docker-build-push.yml**: Common inherited workflow that automates scanning, building and pushing Docker images to a container registry (e.g., AWS ECR). Cn be called by other internal/external workflows.
- **publish.yml**: Handles publishing of packages or images, possibly to PyPI or a Docker registry.
- **set-dev-version.yml**: Automates version bumping or tagging for development releases.

These workflows ensure code quality, automate builds, and streamline deployment for both Python and Docker-based components. For details or customization, review the YAML files in `.github/workflows/`.
