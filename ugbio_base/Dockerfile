# =============================================================================
# Stage 1: Build bioinformatics tools from source
# =============================================================================
FROM python:3.11-slim-bookworm AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG BCFTOOLS_VERSION=1.20
ARG SAMTOOLS_VERSION=1.20
ARG BEDTOOLS_VERSION=2.31.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    automake \
    make \
    gcc \
    g++ \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libcurl4-gnutls-dev \
    libncurses-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# All tools install to /opt/tools via prefix= so we can copy them cleanly
RUN mkdir -p /opt/tools/bin /opt/tools/libexec

# Build bcftools
RUN wget https://github.com/samtools/bcftools/releases/download/${BCFTOOLS_VERSION}/bcftools-${BCFTOOLS_VERSION}.tar.bz2 && \
    tar -vxjf bcftools-${BCFTOOLS_VERSION}.tar.bz2 && \
    rm bcftools-${BCFTOOLS_VERSION}.tar.bz2 && \
    cd bcftools-${BCFTOOLS_VERSION} && \
    make -j$(nproc) && \
    make install prefix=/opt/tools && \
    cd .. && \
    rm -rf bcftools-${BCFTOOLS_VERSION}

# Build samtools
RUN wget https://github.com/samtools/samtools/releases/download/${SAMTOOLS_VERSION}/samtools-${SAMTOOLS_VERSION}.tar.bz2 && \
    tar -vxjf samtools-${SAMTOOLS_VERSION}.tar.bz2 && \
    rm samtools-${SAMTOOLS_VERSION}.tar.bz2 && \
    cd samtools-${SAMTOOLS_VERSION} && \
    make -j$(nproc) && \
    make install prefix=/opt/tools && \
    cd .. && \
    rm -rf samtools-${SAMTOOLS_VERSION}

# Build bedtools
RUN wget https://github.com/arq5x/bedtools2/releases/download/v${BEDTOOLS_VERSION}/bedtools-${BEDTOOLS_VERSION}.tar.gz && \
    tar -zxvf bedtools-${BEDTOOLS_VERSION}.tar.gz && \
    cd bedtools2 && \
    make -j$(nproc) && \
    make install prefix=/opt/tools && \
    cd .. && \
    rm -rf bedtools-${BEDTOOLS_VERSION}.tar.gz bedtools2

# Build bedops
RUN git clone https://github.com/bedops/bedops.git && \
    cd bedops && \
    make -j$(nproc) && \
    make install && \
    cp bin/* /opt/tools/bin/ && \
    cd .. && \
    rm -rf bedops

# =============================================================================
# Stage 2: Runtime image — only binaries and runtime shared libraries
# =============================================================================
FROM python:3.11-slim-bookworm AS ugbio-base

ARG USERNAME=app
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ARG DEBIAN_FRONTEND=noninteractive

# Runtime-only packages: no compilers, no -dev headers
RUN apt-get update && apt-get upgrade -y && apt-get install --no-install-recommends -y \
    wget \
    libgomp1 \
    unzip \
    git \
    gawk \
    jq \
    `# shared libs needed at runtime by bcftools/samtools/pysam/pyBigWig` \
    zlib1g \
    libbz2-1.0 \
    liblzma5 \
    libcurl3-gnutls \
    libncurses6 \
    libncursesw6 \
    procps \
    sysstat \
    sudo \
    bzip2 \
    tabix \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && apt-get autoclean

# Copy compiled tool binaries from builder (bcftools, samtools, bedtools, bedops)
COPY --from=builder /opt/tools/bin/ /usr/local/bin/
COPY --from=builder /opt/tools/libexec/ /usr/local/libexec/

# Include licenses
COPY ["licenses/Ultima UGVC - EULA.pdf", "."]

# Add user and grant sudo privileges
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
