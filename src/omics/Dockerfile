ARG OUT_DIR=/tmp/ugbio
FROM python:3.11-bookworm AS build

ARG OUT_DIR
ARG MODULE_DIR=omics
RUN mkdir -p $OUT_DIR

# Create virtual environment for pip install
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./src/ ./src/

# build module
RUN pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

# Install wheels in build stage
RUN WHEEL_FILE=$(find ${OUT_DIR} -name "*.whl") && \
    pip install "${WHEEL_FILE}[db]"
RUN pip install pytest

##################################################################
FROM python:3.11-slim-bookworm

RUN groupadd ugbio && useradd -m ugbio -g ugbio

# Copy pre-built virtual environment from build stage
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

USER ugbio

WORKDIR /home/ugbio
