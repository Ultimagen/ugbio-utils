ARG BASE_IMAGE=337532070941.dkr.ecr.us-east-1.amazonaws.com/ugbio_base:1.5.0
FROM python:3.11-bullseye AS build

ARG OUT_DIR=/tmp/ugbio
ARG MODULE_DIR=ppmseq
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

#build ugbio_core + module
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

##################################################################
FROM ${BASE_IMAGE}

ARG OUT_DIR=/tmp/ugbio

# Create a non-root user and group
RUN groupadd --system ugbio && useradd --system --create-home --gid ugbio ugbio -d /home/ugbio

# Set ownership for writable directories
RUN chown -R ugbio:ugbio /opt /usr/local/bin

USER ugbio

WORKDIR /home/ugbio

ARG OUT_DIR=/tmp/ugbio

# Copy whl and install module
COPY --from=build $OUT_DIR $OUT_DIR

ENV PATH="/home/ugbio/.local/bin:${PATH}"

RUN pip install --user ${OUT_DIR}/*.whl
