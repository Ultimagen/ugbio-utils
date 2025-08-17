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

ARG USER_ID=1000
ARG GROUP_ID=$USER_ID
# Create a non-root user and group
RUN groupadd -g ${GROUP_ID} ugbio && useradd -m -u ${USER_ID} -g ${GROUP_ID} ugbio -d /home/ugbio

# Set ownership for writable directories
RUN chown -R ugbio:ugbio /opt /usr/local/bin

USER ugbio

WORKDIR /home/ugbio

ARG OUT_DIR=/tmp/ugbio

# Copy whl and install module
COPY --from=build $OUT_DIR $OUT_DIR

ENV PATH="/home/ugbio/.local/bin:${PATH}"

RUN pip install --user ${OUT_DIR}/*.whl
