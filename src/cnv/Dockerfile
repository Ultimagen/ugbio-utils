# BASE_IMAGE is injected by build-workspace-docker.yml (defaults to DEFAULT_BASE_IMAGE)
ARG BASE_IMAGE
ARG BASE_BUILD_IMAGE
ARG OUT_DIR=/tmp/ugbio
FROM $BASE_BUILD_IMAGE AS build

ARG OUT_DIR
ARG MODULE_DIR=cnv
RUN mkdir -p $OUT_DIR

# Create virtual environment for pip install
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./src/ ./src/

# build module
RUN pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

# Install wheels — C extensions compile here, not in runtime.
# pywfa and pyabpoa (truvari, used by merge_cnv_sv) are built without -march=native, so the image runs on hosts
# without AVX-512 (a native build exits with SIGILL there)
RUN PYWFA_MARCH=x86-64-v3 pip install --no-binary pywfa "pywfa>=0.6.0"
RUN AVX2=1 pip install ${OUT_DIR}/*.whl
RUN pip install pytest

##################################################################
FROM $BASE_IMAGE

RUN groupadd ugbio && useradd -m ugbio -g ugbio

# Copy pre-built virtual environment from build stage (no compilation in runtime)
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

USER ugbio

WORKDIR /home/ugbio
