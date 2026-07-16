# BASE_IMAGE is injected by build-workspace-docker.yml (defaults to DEFAULT_BASE_IMAGE)
ARG BASE_IMAGE
ARG BASE_BUILD_IMAGE
ARG OUT_DIR=/tmp/ugbio
FROM $BASE_BUILD_IMAGE AS build

ARG OUT_DIR
ARG MODULE_DIR=comparison
RUN mkdir -p $OUT_DIR

# Create virtual environment for pip install
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./src/ ./src/

# build module
RUN pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

# Install wheels — C extensions compile here, not in runtime
RUN pip install ${OUT_DIR}/*.whl
RUN pip install pytest

##################################################################
FROM $BASE_IMAGE

RUN apt-get update && apt-get upgrade -y && apt-get install --no-install-recommends -y \
    openjdk-17-jre-headless `# for gatk` \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && apt-get autoclean

# Download and install gatk
ARG GATK_VERSION=4.6.2.0
RUN wget https://github.com/broadinstitute/gatk/releases/download/${GATK_VERSION}/gatk-${GATK_VERSION}.zip && \
    unzip gatk-${GATK_VERSION}.zip && \
    mv gatk-${GATK_VERSION} /opt/gatk && \
    ln -s /opt/gatk/gatk /usr/local/bin/gatk && \
    rm gatk-${GATK_VERSION}.zip

# Patch GATK fat JAR: upgrade bundled netty and zookeeper to fix CVEs
COPY docker/patch-gatk-jars.sh /tmp/patch-gatk-jars.sh
RUN chmod +x /tmp/patch-gatk-jars.sh && /tmp/patch-gatk-jars.sh && rm /tmp/patch-gatk-jars.sh

# Download Picard
ARG PICARD_VERSION=3.4.0
RUN apt-get update && apt-get install -y --no-install-recommends --reinstall ca-certificates openjdk-17-jre-headless \
    && apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get autoclean
RUN mkdir -p /opt/picard && \
    wget -O /opt/picard/picard.jar https://github.com/broadinstitute/picard/releases/download/${PICARD_VERSION}/picard.jar && \
    chmod +x /opt/picard/picard.jar
    # Create a shell script to run Picard easily
RUN echo '#!/bin/sh\njava -jar /opt/picard/picard.jar "$@"' > /usr/local/bin/picard && \
    chmod +x /usr/local/bin/picard

# Download RTG-tools
ARG RTG_VERSION=3.12.1
RUN wget https://github.com/RealTimeGenomics/rtg-tools/releases/download/${RTG_VERSION}/rtg-tools-${RTG_VERSION}-linux-x64.zip && \
    unzip -d /opt rtg-tools-${RTG_VERSION}-linux-x64.zip && \
    rm rtg-tools-${RTG_VERSION}-linux-x64.zip && \
    ln -s /opt/rtg-tools-${RTG_VERSION} /opt/rtg && \
    echo "RTG_TALKBACK=true  # Crash logs\nRTG_USAGE=false # Simple usage logging\n" >/opt/rtg/rtg.cfg && \
    ln -s /opt/rtg/rtg /usr/local/bin/rtg

RUN groupadd ugbio && useradd -m ugbio -g ugbio

# Copy pre-built virtual environment from build stage (no compilation in runtime)
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

USER ugbio

WORKDIR /home/ugbio
