# BASE_IMAGE is injected by build-workspace-docker.yml (defaults to DEFAULT_BASE_IMAGE)
ARG BASE_IMAGE
ARG OUT_DIR=/tmp/ugbio
FROM python:3.11-bookworm AS build

ARG OUT_DIR
ARG MODULE_DIR=mrd
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

#build all depended + module
RUN pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core
RUN python -m build --outdir $OUT_DIR --wheel ./src/ppmseq
RUN python -m build --outdir $OUT_DIR --wheel ./src/featuremap
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

# Install wheels — C extensions compile here, not in runtime
RUN pip install ${OUT_DIR}/*.whl

##################################################################
FROM ${BASE_IMAGE}

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

RUN groupadd ugbio && useradd -m ugbio -g ugbio

# Copy pre-built virtual environment from build stage (no compilation in runtime)
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

USER ugbio

WORKDIR /home/ugbio

COPY --from=build ./src/mrd/ugbio_mrd/reports ./src/mrd/ugbio_mrd/reports
