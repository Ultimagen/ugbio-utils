FROM python:3.11-bullseye AS ugbio-base

ARG DEBIAN_FRONTEND=noninteractive
ARG BCFTOOLS_INSTALL_DIR=/opt/bcftools
ARG BCFTOOLS_VERSION=1.20

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    wget \
    libgomp1 \
    unzip \
    automake \
    make \
    libcurl4-gnutls-dev \
    git \
    `# the following packages required for pyBigWig python package used by ugbio_core` \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libc6-dev \
    gcc \
    procps `# for monitoring script memory usage` \
    sysstat `# for monitoring script IO metrics` \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && apt-get autoclean

# install bcftools
RUN wget https://github.com/samtools/bcftools/releases/download/${BCFTOOLS_VERSION}/bcftools-${BCFTOOLS_VERSION}.tar.bz2 && \
    tar -vxjf bcftools-${BCFTOOLS_VERSION}.tar.bz2 && \
    rm bcftools-${BCFTOOLS_VERSION}.tar.bz2 && \
    cd bcftools-${BCFTOOLS_VERSION} && \
    make && \
    make install

# Install bedtools
ARG BEDTOOLS_VERSION=2.31.0

RUN wget https://github.com/arq5x/bedtools2/releases/download/v${BEDTOOLS_VERSION}/bedtools-${BEDTOOLS_VERSION}.tar.gz && \
    tar -zxvf bedtools-${BEDTOOLS_VERSION}.tar.gz && \
    cd bedtools2 && \
    make && \
    make install && \
    cd .. && \
    rm -rf bedtools-${BEDTOOLS_VERSION}.tar.gz bedtools2

# Install bedops
RUN git clone https://github.com/bedops/bedops.git && \
    cd bedops && \
    make && \
    make install && \
    cp bin/* /usr/local/bin && \
    cd .. && \
    rm -rf bedops

# Include licenses
COPY ["licenses/Ultima UGVC - EULA.pdf", "."]