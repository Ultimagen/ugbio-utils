ARG BASE_IMAGE=ugbio_base
ARG OUT_DIR=/tmp/ugbio
FROM python:3.11-bullseye AS build

ARG OUT_DIR
ARG MODULE_DIR=srsnv
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

#build all depnded + module
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core
RUN python -m build --outdir $OUT_DIR --wheel ./src/ppmseq
RUN python -m build --outdir $OUT_DIR --wheel ./src/featuremap
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

##################################################################
FROM $BASE_IMAGE

ARG OUT_DIR

RUN groupadd ugbio && useradd -m ugbio -g ugbio

COPY --from=build ${OUT_DIR} ${OUT_DIR}
RUN pip install ${OUT_DIR}/*.whl && rm -rf ${OUT_DIR}

USER ugbio

WORKDIR /home/ugbio
