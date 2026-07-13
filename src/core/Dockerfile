# BASE_IMAGE is injected by build-workspace-docker.yml (defaults to DEFAULT_BASE_IMAGE)
ARG BASE_IMAGE
ARG OUT_DIR=/tmp/ugbio
FROM python:3.11-bookworm AS build

ARG OUT_DIR
RUN mkdir -p $OUT_DIR

# Install build dependencies for C extensions (pyBigWig, pysam, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libcurl4-gnutls-dev \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for pip install
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./src/ ./src/

#build ugbio_core
RUN pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core

# Install wheels with all extras — C extensions compile here, not in runtime
RUN WHEEL_FILE=$(find ${OUT_DIR} -name "ugbio_core*.whl") && \
    pip install "${WHEEL_FILE}[vcfbed,reports,ml]"

# ##################################################################
FROM $BASE_IMAGE

# Download Picard
ARG PICARD_VERSION=3.4.0
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends --reinstall ca-certificates openjdk-17-jre-headless \
    && apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get autoclean
RUN mkdir -p /opt/picard && \
    wget -O /opt/picard/picard.jar https://github.com/broadinstitute/picard/releases/download/${PICARD_VERSION}/picard.jar && \
    chmod +x /opt/picard/picard.jar
    # Create a shell script to run Picard easily
RUN echo '#!/bin/sh\njava -jar /opt/picard/picard.jar "$@"' > /usr/local/bin/picard && \
    chmod +x /usr/local/bin/picard

RUN groupadd ugbio && useradd -m ugbio -g ugbio

# Copy pre-built virtual environment from build stage (no compilation in runtime)
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

USER ugbio

WORKDIR /home/ugbio

COPY --from=build ./src/core/ugbio_core/reports ./src/core/ugbio_core/reports
