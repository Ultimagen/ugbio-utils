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

# Install module
COPY --from=build $OUT_DIR $OUT_DIR

RUN pip install ${OUT_DIR}/*.whl

# for runnig CI (can be removed if pytest is installed as part of the venv)
RUN pip install pytest
