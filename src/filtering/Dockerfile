ARG BASE_IMAGE=ugbio_base
FROM python:3.11-bullseye AS build

ARG OUT_DIR=/tmp/ugbio
ARG MODULE_DIR=filtering
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

# build depended + module
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core
RUN python -m build --outdir $OUT_DIR --wheel ./src/comparison
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

##################################################################
FROM $BASE_IMAGE

RUN apt-get update && apt-get install --no-install-recommends -y \
    openjdk-17-jre-headless `# for gatk` \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && apt-get autoclean

# Download and install gatk
ARG GATK_VERSION=4.6.0.0
RUN wget https://github.com/broadinstitute/gatk/releases/download/${GATK_VERSION}/gatk-${GATK_VERSION}.zip && \
    unzip gatk-${GATK_VERSION}.zip && \
    mv gatk-${GATK_VERSION} /opt/gatk && \
    ln -s /opt/gatk/gatk /usr/local/bin/gatk && \
    rm gatk-${GATK_VERSION}.zip

# Download Picard
ARG PICARD_VERSION=3.3.0
RUN mkdir -p /opt/picard && \
    wget -O /opt/picard/picard.jar https://github.com/broadinstitute/picard/releases/download/${PICARD_VERSION}/picard.jar && \
    chmod +x /opt/picard/picard.jar
    # Create a shell script to run Picard easily
RUN echo '#!/bin/sh\njava -jar /opt/picard/picard.jar "$@"' > /usr/local/bin/picard && \
    chmod +x /usr/local/bin/picard

# Download RTG-tools
ARG RTG_VERSION=3.12.1
RUN curl -SLO https://github.com/RealTimeGenomics/rtg-tools/releases/download/${RTG_VERSION}/rtg-tools-${RTG_VERSION}-linux-x64.zip && \
    unzip -d /opt rtg-tools-${RTG_VERSION}-linux-x64.zip && \
    rm rtg-tools-${RTG_VERSION}-linux-x64.zip && \
    ln -s /opt/rtg-tools-${RTG_VERSION} /opt/rtg && \
    echo "RTG_TALKBACK=true  # Crash logs\nRTG_USAGE=false # Simple usage logging\n" >/opt/rtg/rtg.cfg && \
    ln -s /opt/rtg/rtg /usr/local/bin/rtg


RUN groupadd ugbio && useradd -m ugbio -g ugbio
RUN chown -R ugbio:ugbio /opt /usr/local/bin

USER ugbio

WORKDIR /home/ugbio

ARG OUT_DIR=/tmp/ugbio

# Copy whl and install module
COPY --from=build $OUT_DIR $OUT_DIR

ENV PATH="/home/ugbio/.local/bin:${PATH}"

RUN pip install --user ${OUT_DIR}/*.whl
