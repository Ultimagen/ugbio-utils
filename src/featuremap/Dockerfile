ARG BASE_IMAGE=ugbio_base
FROM python:3.11-bullseye AS build

ARG OUT_DIR=/tmp/ugbio
ARG MODULE_DIR=featuremap
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

#build dependened + module
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core
RUN python -m build --outdir $OUT_DIR --wheel ./src/ppmseq
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR
#install htslib
RUN apt-get update && \
    apt-get install -y htslib && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

##################################################################
FROM $BASE_IMAGE

ARG OUT_DIR=/tmp/ugbio

COPY --from=build $OUT_DIR $OUT_DIR

RUN pip install ${OUT_DIR}/*.whl
