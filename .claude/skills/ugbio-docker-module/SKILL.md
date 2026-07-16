---
name: ugbio-docker-module
description: Use when creating a new module Dockerfile, updating an existing one, or remediating CVE/Trivy/Inspector findings on a ugbio module image.
---

# Ugbio Docker Module

Guidance for writing and updating Dockerfiles for ugbio_utils module images (the per-module images under `src/<module>/`), including how to respond to CVE/Trivy/Inspector findings.

## When to Use This Skill

Activate this skill when:

- Creating a new module Dockerfile.
- Updating an existing module's Dockerfile.
- Responding to a Trivy/ECR Inspector CRITICAL or HIGH finding on a ugbio module image.

## The Two Base Images

`build-ugbio-base-docker.yml` publishes two images from the **same** base Dockerfile:

- **`ugbio_base`** — the runtime image. `python:3.11-slim-bookworm` plus bcftools, samtools, bedtools, bedops pre-installed.
- **`ugbio_base_build`** — the `builder` target of that same Dockerfile. Adds gcc, g++, make, python3-dev, libncurses-dev, pip on top, for compiling wheels/native deps.

For the full deep-dive on why this split exists and how the CVE remediation was scoped, see `docs/agent_docs/DATA-9903-docker-cve-remediation-summary.md`.

## Multi-Stage Is Always the Right Shape

Regardless of which base images a module uses, the Dockerfile should be multi-stage: a build stage with compilers/pip/source, and a runtime stage that only copies in the built `/opt/venv`. This is what actually removes gcc, pip, dev-headers, and source from the CVE-scanned final image — it is not tied to using `ugbio_base` specifically. A Dockerfile that skips this split will carry build tooling into the scanned runtime image no matter which base it starts from.

## Whether to Base On ugbio_base / ugbio_base_build — Decide, Don't Default

This is a decision per module, not a default to blindly apply:

- Use `BASE_IMAGE`/`BASE_BUILD_IMAGE` (resolving to `ugbio_base`/`ugbio_base_build`) **only when the module actually needs the tools `ugbio_base` provides** (bcftools, samtools, bedtools, bedops), or wants to inherit its patched/slim Debian layer for consistency with the rest of the fleet.
- If the module doesn't need those bioinformatics CLI tools, it's fine — and simpler — to build multi-stage against a plain `python:3.11-slim-bookworm` (or similar) in both stages instead, exactly like **pypgx** does. Don't force `ugbio_base` on a module just for the sake of uniformity.

### Standard Pattern When ugbio_base IS the Right Fit

```dockerfile
ARG BASE_IMAGE
ARG BASE_BUILD_IMAGE

FROM $BASE_BUILD_IMAGE AS build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY ./src/ ./src/
RUN pip install build
RUN python -m build --outdir /tmp/ugbio --wheel ./src/core
RUN python -m build --outdir /tmp/ugbio --wheel ./src/<module>
RUN pip install /tmp/ugbio/*.whl

FROM $BASE_IMAGE
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN groupadd ugbio && useradd -m ugbio -g ugbio
USER ugbio
```

`BASE_IMAGE`/`BASE_BUILD_IMAGE` are injected by `build-workspace-docker.yml` (it reads the `ARG ... default` from the Dockerfile if present, else falls back to `DEFAULT_BASE_IMAGE`/`DEFAULT_BASE_BUILD_IMAGE`) — no need to hardcode a registry path.

### Still Multi-Stage, Just Not on ugbio_base

When the module has its own strict/conflicting version constraints, it's still multi-stage, just self-contained:

- **pypgx**: hardcodes `python:3.11-slim-bookworm` in both stages (no `BASE_IMAGE`/`BASE_BUILD_IMAGE`), pins `numpy<2` + `pandas<2.0` to avoid a proven ABI mismatch crash. Doesn't need bcftools/samtools from `ugbio_base` for its own build step, and pypgx/pypgx-bundle (third-party GitHub clones) have their own strict version/ABI constraints independent of `ugbio_base`'s Python version.
- **omics**, **single_cell**, **cloud_utils** are also self-contained (`python:3.11-bookworm`/`slim-bookworm`), mostly because they don't need `ugbio_base`'s bio CLI tools either — this is a legitimate, not accidental, choice per module.

### Exception Still Pinned to an Old Base Entirely

This is not a "which base" choice, it's a "can't upgrade yet" blocker:

- **cnv**: pinned to `ugbio_base:1.7.0` (bullseye) hardcoded, no `BASE_BUILD_IMAGE` — blocked by a `libparasail8` ABI incompatibility on bookworm. See `src/cnv/DATA-9903-jalign-bookworm-migration.md` before attempting to migrate this.

## DO

- DO use multi-stage with `ugbio_base_build` for the build stage, `ugbio_base` for runtime — this is what actually removes gcc/pip/dev-headers/source from the CVE-scanned final image.
- DO create and `USER` into the non-root `ugbio` user in the runtime stage.
- DO purge any tool installed only for building (git/wget/unzip/build-essential) in the *same RUN layer* it was used, once no longer needed — but never purge something the runtime actually calls (e.g. don't purge `bcftools`, only purge `git`/`wget`/`unzip`/compilers).
- DO run `apt-get upgrade -y` if the Dockerfile does its own apt installs, to pick up patched system packages.
- DO check `docker/patch-gatk-jars.sh` if the module bundles GATK — it patches the fat JAR's netty/zookeeper to fix CVEs GATK itself hasn't shipped a fix for yet.
- DO pin any version that's had a proven runtime incompatibility (numpy/pandas ABI, etc.) with a comment explaining *why*, not just what.

## DON'T

- DON'T install `libcurl4-gnutls-dev`, `default-libmysqlclient-dev`, `libglib2.0-dev`, or other `-dev` headers in the runtime stage — these belong in the builder stage only, and are a major source of CRITICAL CVEs when they leak into runtime.
- DON'T use the full `python:3.11-bookworm`/`bullseye` image for anything — always `-slim-` variants; the full image's persistent build-deps layer pulls in unused mariadb/glib2.0 client libraries.
- DON'T add ImageMagick, JupyterLab, or other heavy transitive deps without checking if a lighter alternative exists (e.g. `ipykernel` instead of the `jupyter` metapackage for papermill).
- DON'T assume a CVE fix requires waiting on upstream — check if it's fixable by pinning a patched transitive dependency directly (see GATK jar patching) or removing an unused package outright.
- DON'T skip re-scanning after a change — CVE counts are the actual acceptance criteria, not just "the build succeeded."

## CVE Scanning in the Build Pipeline

Two distinct scans run, with different timing and different consequences:

- **Trivy config + fs scans** (`docker-build-push.yml`, before image build) — `exit-code: '1'`, severity `CRITICAL,HIGH` — **these block the build** if they find something.
- **ECR/Inspector2 scan** (`.github/utils/ecr_scan_results.py`, step "Check ECR scan results") — runs **after** the image is built and pushed, at the end of `docker-build-push.yml`. It polls AWS Inspector2 for the pushed image's findings, writes a JSON findings file plus a GitHub step summary table, but is `continue-on-error: true` and never fails the build itself — it's a report, not a gate. This is the source of the CRITICAL counts tracked in `docs/agent_docs/DATA-9903-docker-cve-remediation-summary.md`.

**Practical flow:** push your Dockerfile change → trigger `build-ugbio-member-docker.yml` for the module → check the run's step summary for the ECR scan table → compare CRITICAL count before/after.

## Maintenance Note

This skill documents the Docker structure as of DATA-9903 (multi-stage refactor). If the base-image split, multi-stage pattern, or CVE-scan pipeline changes, update this skill in the same PR.
