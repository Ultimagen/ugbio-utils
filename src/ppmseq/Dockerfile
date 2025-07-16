ARG BASE_IMAGE=ugbio_base
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

COPY --from=build $OUT_DIR $OUT_DIR

# Create a non-root user and group
RUN groupadd --system ugbio && useradd --system --create-home --gid ugbio ugbio

# Copy files before changing ownership
COPY --from=build $OUT_DIR $OUT_DIR

# If you need /workdir/src, create it or copy it first
COPY ./src/ /workdir/src/

# Set ownership for writable directories
RUN chown -R ugbio:ugbio /usr/local/bin /workdir/src/

RUN echo "umask 002" >> /etc/profile

USER ugbio

ENV PATH="/home/ugbio/.local/bin:${PATH}"

RUN pip install ${OUT_DIR}/*.whl
