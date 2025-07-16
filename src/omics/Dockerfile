FROM python:3.11-bullseye AS build

ARG OUT_DIR=/tmp/ugbio
ARG MODULE_DIR=omics
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

# build module
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

##################################################################
FROM python:3.11-slim

ARG OUT_DIR=/tmp/ugbio

# Create a non-root user and group
RUN groupadd --system ugbio && useradd --system --create-home --gid ugbio ugbio

# Set ownership for writable directories
RUN chown -R ugbio:ugbio /opt /usr/local/bin

USER ugbio

ARG OUT_DIR=/tmp/ugbio

# Copy whl and install module
COPY --from=build $OUT_DIR $OUT_DIR

RUN WHEEL_FILE=$(find ${OUT_DIR} -name "*.whl") && \
    pip install "${WHEEL_FILE}[db]"

# for runnig CI (can be removed if pytest is installed as part of the venv)
RUN pip install pytest
