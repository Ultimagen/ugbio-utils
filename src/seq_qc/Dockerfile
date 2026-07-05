# BASE_IMAGE is injected by build-workspace-docker.yml (defaults to DEFAULT_BASE_IMAGE)
ARG BASE_IMAGE
ARG OUT_DIR=/tmp/ugbio
FROM python:3.11-bookworm AS build

ARG OUT_DIR
ARG MODULE_DIR=seq_qc
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

#build ugbio_core + ugbio_cloud_utils + module
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core
RUN python -m build --outdir $OUT_DIR --wheel ./src/cloud_utils
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

##################################################################
FROM ${BASE_IMAGE}

ARG OUT_DIR

RUN groupadd ugbio && useradd -m ugbio -g ugbio

COPY --from=build ${OUT_DIR} ${OUT_DIR}
RUN pip install ${OUT_DIR}/*.whl && rm -rf ${OUT_DIR}

USER ugbio

WORKDIR /home/ugbio
