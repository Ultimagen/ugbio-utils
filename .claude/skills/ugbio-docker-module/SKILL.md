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
- DO install GATK with `docker/install-gatk.sh` (COPY + RUN) if the module bundles GATK — it installs the Ultima GATK build with fixed netty (official releases still bundle a vulnerable netty) into /opt/gatk.
- DO pin any version that's had a proven runtime incompatibility (numpy/pandas ABI, etc.) with a comment explaining *why*, not just what.

## DON'T

- DON'T install `libcurl4-gnutls-dev`, `default-libmysqlclient-dev`, `libglib2.0-dev`, or other `-dev` headers in the runtime stage — these belong in the builder stage only, and are a major source of CRITICAL CVEs when they leak into runtime.
- DON'T use the full `python:3.11-bookworm`/`bullseye` image for anything — always `-slim-` variants; the full image's persistent build-deps layer pulls in unused mariadb/glib2.0 client libraries.
- DON'T add ImageMagick, JupyterLab, or other heavy transitive deps without checking if a lighter alternative exists (e.g. `ipykernel` instead of the `jupyter` metapackage for papermill).
- DON'T assume a CVE fix requires waiting on upstream — check if it's fixable by pinning a patched transitive dependency directly (see GATK jar patching) or removing an unused package outright.
- DON'T skip re-scanning after a change — CVE counts are the actual acceptance criteria, not just "the build succeeded."

## CVE Scanning in the Build Pipeline

Every image is built and scanned by one composite action, `.github/actions/docker-build-scan`, used both by CI (`run-workspace-tests`, nothing pushed) and by `docker-build-push.yml` (which pushes only if the action passed). CI, releases and the base images use `policy: block`; dev member builds (`build-ugbio-member-docker.yml`) use `warn`. In CI a blocking finding fails the module job before its tests run. The action runs, before any push:

- **Trivy Dockerfile config scan** of the built Dockerfile: reports CRITICAL/HIGH; fails on CRITICAL with `policy: block`.
- **Trivy image scan** (vulnerabilities + secrets), gated by `trivy_json_to_summary.py`: with `policy: block`, a CRITICAL vulnerability with a fix, or a CRITICAL in a language package, fails. A HIGH/CRITICAL secret fails under any policy.
- **Build-token check**: the image must not contain the token passed as the `github_token` build secret (raw or as the base64 git `extraheader`).
- **Syft + Grype binaries report** (`binary_findings.py`, report only): CVEs in binaries Trivy can't see (e.g. CPython in `/usr/local`) and executables that belong to no package (source-built tools).

`ci.yml` also runs a Trivy filesystem scan of the repo (lockfile + secrets).

**Practical flow:** push your Dockerfile change → the PR's CI builds and scans every module → check the job summary (Trivy table, binaries section) → compare CRITICAL count before/after. CI uploads no artifact; for the full JSON results (`trivy-results.json`, `sbom.json`, `grype-results.json`), run `build-ugbio-member-docker.yml` for the module and download its `trivy-image-findings-*` artifact. That artifact is not uploaded when a secret was found (the results embed the image config).

## Maintenance Note

This skill documents the Docker structure as of DATA-9903 (multi-stage refactor). If the base-image split, multi-stage pattern, or CVE-scan pipeline changes, update this skill in the same PR.
