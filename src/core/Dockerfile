ARG BASE_IMAGE=ugbio_base
FROM python:3.11-bullseye AS build

ARG OUT_DIR=/tmp/ugbio
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

#build ugbio_core
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core

# ##################################################################
FROM $BASE_IMAGE

# Download Picard
ARG PICARD_VERSION=3.3.0
RUN apt-get update && apt-get install -y --reinstall ca-certificates openjdk-17-jre-headless \
    && apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get autoclean
RUN mkdir -p /opt/picard && \
    wget -O /opt/picard/picard.jar https://github.com/broadinstitute/picard/releases/download/${PICARD_VERSION}/picard.jar && \
    chmod +x /opt/picard/picard.jar
    # Create a shell script to run Picard easily
RUN echo '#!/bin/sh\njava -jar /opt/picard/picard.jar "$@"' > /usr/local/bin/picard && \
    chmod +x /usr/local/bin/picard

ARG OUT_DIR=/tmp/ugbio

COPY --from=build $OUT_DIR $OUT_DIR

RUN WHEEL_FILE=$(find ${OUT_DIR} -name "*.whl") && \
    pip install "${WHEEL_FILE}[vcfbed]" \
