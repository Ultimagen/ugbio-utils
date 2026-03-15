ARG OUT_DIR=/tmp/ugbio
FROM python:3.11-bullseye AS build

ARG OUT_DIR
ARG MODULE_DIR=srsnv
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

#build all depended + module
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core
RUN python -m build --outdir $OUT_DIR --wheel ./src/ppmseq
RUN python -m build --outdir $OUT_DIR --wheel ./src/featuremap
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

##################################################################
FROM nvcr.io/nvidia/tensorrt:25.03-py3

ARG OUT_DIR=/tmp/ugbio
ARG DEBIAN_FRONTEND=noninteractive
ARG BCFTOOLS_VERSION=1.20
ARG SAMTOOLS_VERSION=1.20
ARG BEDTOOLS_VERSION=2.31.1

ENV PATH="/opt/tensorrt/bin:${PATH}"

RUN apt-get update && apt-get install --no-install-recommends -y \
    wget \
    libgomp1 \
    unzip \
    automake \
    make \
    libcurl4-gnutls-dev \
    git \
    gawk \
    jq \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libc6-dev \
    gcc \
    libncurses-dev \
    procps \
    bzip2 \
    tabix \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/samtools/bcftools/releases/download/${BCFTOOLS_VERSION}/bcftools-${BCFTOOLS_VERSION}.tar.bz2 && \
    tar -vxjf bcftools-${BCFTOOLS_VERSION}.tar.bz2 && \
    rm bcftools-${BCFTOOLS_VERSION}.tar.bz2 && \
    cd bcftools-${BCFTOOLS_VERSION} && \
    make && \
    make install && \
    cd .. && \
    rm -rf bcftools-${BCFTOOLS_VERSION}

RUN wget https://github.com/samtools/samtools/releases/download/${SAMTOOLS_VERSION}/samtools-${SAMTOOLS_VERSION}.tar.bz2 && \
    tar -vxjf samtools-${SAMTOOLS_VERSION}.tar.bz2 && \
    rm samtools-${SAMTOOLS_VERSION}.tar.bz2 && \
    cd samtools-${SAMTOOLS_VERSION} && \
    make && \
    make install && \
    cd .. && \
    rm -rf samtools-${SAMTOOLS_VERSION}

RUN wget https://github.com/arq5x/bedtools2/releases/download/v${BEDTOOLS_VERSION}/bedtools-${BEDTOOLS_VERSION}.tar.gz && \
    tar -zxvf bedtools-${BEDTOOLS_VERSION}.tar.gz && \
    cd bedtools2 && \
    make && \
    make install && \
    cd .. && \
    rm -rf bedtools-${BEDTOOLS_VERSION}.tar.gz bedtools2

RUN groupadd ugbio && useradd -m ugbio -g ugbio

COPY --from=build ${OUT_DIR} ${OUT_DIR}
RUN SRSNV_WHL=$(ls ${OUT_DIR}/ugbio_srsnv*.whl) && \
    pip install "${SRSNV_WHL}[trt]" ${OUT_DIR}/*.whl && \
    rm -rf ${OUT_DIR}

USER ugbio

WORKDIR /home/ugbio
