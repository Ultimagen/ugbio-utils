FROM python:3.11-bullseye AS build

ARG OUT_DIR=/tmp/ugbio
ARG MODULE_DIR=cloud_utils
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

# build module
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

##################################################################
FROM python:3.11-slim

ARG OUT_DIR=/tmp/ugbio

# Create a non-root user and group
RUN groupadd ugbio && useradd -m ugbio -d /home/ugbio -g ugbio

# Set ownership for writable directories
RUN chown -R ugbio:ugbio /opt /usr/local/bin

USER ugbio

WORKDIR /home/ugbio

ARG OUT_DIR=/tmp/ugbio

COPY --from=build $OUT_DIR $OUT_DIR

ENV PATH="/home/ugbio/.local/bin:${PATH}"

RUN pip install --user ${OUT_DIR}/*.whl

# for runnig CI (can be removed if pytest is installed as part of the venv)
RUN pip install --user pytest
