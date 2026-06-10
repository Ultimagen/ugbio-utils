# You can build with default version:
# docker build -f Dockerfile.giraffe_with_picard -t giraffe:v1.74.0 .
# Or override to any other VG/Giraffe release:
# docker build -f Dockerfile.giraffe_with_picard -t giraffe:v1.75.0 --build-arg VG_VERSION=1.75.0 .

FROM --platform=linux/amd64 ubuntu:24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG VG_VERSION=1.74.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    make \
    build-essential \
    zlib1g-dev \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Build KMC
RUN git clone --depth 1 --recurse-submodules https://github.com/refresh-bio/kmc.git /tmp/kmc \
    && cd /tmp/kmc \
    && make -j "$(nproc)" kmc kmc_tools \
    && install -m 0755 /tmp/kmc/bin/kmc /usr/local/bin/kmc \
    && install -m 0755 /tmp/kmc/bin/kmc_tools /usr/local/bin/kmc_tools \
    && rm -rf /tmp/kmc

# Build minimap2
RUN git clone --depth 1 https://github.com/lh3/minimap2 /tmp/minimap2 \
    && cd /tmp/minimap2 \
    && make -j "$(nproc)" \
    && install -m 0755 /tmp/minimap2/minimap2 /usr/local/bin/minimap2 \
    && rm -rf /tmp/minimap2

# Download seqkit and vg binaries
RUN set -eux; \
    curl -fsSL https://github.com/shenwei356/seqkit/releases/download/v2.12.0/seqkit_linux_amd64.tar.gz -o /tmp/seqkit_linux_amd64.tar.gz; \
    tar -xzf /tmp/seqkit_linux_amd64.tar.gz -C /tmp; \
    install -m 0755 /tmp/seqkit /usr/local/bin/seqkit; \
    rm -f /tmp/seqkit_linux_amd64.tar.gz /tmp/seqkit

RUN set -eux; \
    curl -fsSL https://github.com/vgteam/vg/releases/download/v${VG_VERSION}/vg -o /usr/local/bin/vg; \
    chmod 0755 /usr/local/bin/vg

# Download GATK jar used as picard.jar
RUN mkdir -p /usr/gitc /opt \
    && curl -fsSL https://github.com/broadinstitute/gatk/releases/download/4.2.6.1/gatk-4.2.6.1.zip -o /tmp/gatk.zip \
    && unzip -q /tmp/gatk.zip -d /opt \
    && ln -s /opt/gatk-4.2.6.1/gatk-package-4.2.6.1-local.jar /usr/gitc/picard.jar \
    && rm -f /tmp/gatk.zip


FROM --platform=linux/amd64 ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    default-jre-headless \
    samtools \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/gitc

COPY --from=builder /usr/local/bin/kmc /usr/local/bin/kmc
COPY --from=builder /usr/local/bin/kmc_tools /usr/local/bin/kmc_tools
COPY --from=builder /usr/local/bin/minimap2 /usr/local/bin/minimap2
COPY --from=builder /usr/local/bin/seqkit /usr/local/bin/seqkit
COPY --from=builder /usr/local/bin/vg /usr/local/bin/vg
COPY --from=builder /opt/gatk-4.2.6.1/gatk-package-4.2.6.1-local.jar /opt/gatk-4.2.6.1/gatk-package-4.2.6.1-local.jar

# Keep compatibility with existing workflows expecting /usr/gitc/picard.jar
RUN ln -sf /opt/gatk-4.2.6.1/gatk-package-4.2.6.1-local.jar /usr/gitc/picard.jar

# Create a script to run tests from docker CI
RUN echo '#!/bin/sh\necho "No tests to run"' > /usr/local/bin/run_tests \
&& chmod +x /usr/local/bin/run_tests

# Don't use root user
RUN groupadd ugbio && useradd -m ugbio -g ugbio
USER ugbio
WORKDIR /home/ugbio
