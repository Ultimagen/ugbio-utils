ARG BASE_IMAGE=us-central1-docker.pkg.dev/ultima-data-307918/ultimagen/ugbio_base:latest
FROM python:3.10-bullseye as build

ARG OUT_DIR=/tmp/ugbio
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

#build ugbio_core
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core

# ##################################################################
FROM $BASE_IMAGE

ARG OUT_DIR=/tmp/ugbio

COPY --from=build-base $OUT_DIR $OUT_DIR

RUN pip install ${OUT_DIR}/*.whl 