ARG BASE_IMAGE=ugbio_base
FROM python:3.11-bullseye AS build

ARG OUT_DIR=/tmp/ugbio
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

ARG OUT_DIR=/tmp/ugbio

# Create a non-root user and group
RUN groupadd --system ugbio && useradd --system --create-home --gid ugbio ugbio

# Set ownership for writable directories
RUN chown -R ugbio:ugbio /app /usr/local/bin

USER ugbio

COPY --from=build $OUT_DIR $OUT_DIR

RUN pip install ${OUT_DIR}/*.whl
