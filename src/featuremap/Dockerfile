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

##################################################################
FROM $BASE_IMAGE

# Create a non-root user and group
RUN groupadd --system ugbio && useradd --system --create-home --gid ugbio ugbio

# Set ownership for writable directories
RUN chown -R ugbio:ugbio /opt /usr/local/bin

USER ugbio

ARG OUT_DIR=/tmp/ugbio

# Copy whl and install module
COPY --from=build $OUT_DIR $OUT_DIR

ENV PATH="/home/.local/bin:${PATH}"

RUN pip install ${OUT_DIR}/*.whl
