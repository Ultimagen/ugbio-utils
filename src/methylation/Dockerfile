ARG BASE_BUILD_IMAGE
FROM $BASE_BUILD_IMAGE AS build

ARG OUT_DIR=/tmp/ugbio
ARG MODULE_DIR=methylation
RUN mkdir -p $OUT_DIR

# Extra build deps for libBigWig/MethylDackel (beyond what BASE_BUILD_IMAGE provides)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhts-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for pip install
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./src/ ./src/

# build module
RUN pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

# Install wheels — C extensions compile here
RUN pip install ${OUT_DIR}/*.whl
RUN pip install pytest

# Build libBigWig & MethylDackel
WORKDIR /app
RUN git clone https://github.com/dpryan79/libBigWig.git && \
    cd libBigWig && \
    make && \
    make install && \
    cp libBigWig.a /usr/local/lib/

RUN git clone https://github.com/Ultimagen/MethylDackel.git && \
    cd MethylDackel && \
    make LIBBIGWIG="/usr/local/lib/libBigWig.a" && \
    mkdir -p /app/bin && \
    make install prefix=/app/bin/

##################################################################
FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

# Runtime-only dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install --no-install-recommends -y \
    openjdk-17-jre-headless \
    wget \
    unzip \
    sudo \
    zlib1g \
    libbz2-1.0 \
    liblzma5 \
    libcurl3-gnutls \
    libhts3 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && apt-get autoclean

# Copy MethylDackel binary from build stage
COPY --from=build /app/bin/ /app/bin/

# Install GATK (Ultima build of 4.7.0.0 with fixed netty, see docker/install-gatk.sh)
COPY docker/install-gatk.sh /tmp/install-gatk.sh
RUN bash /tmp/install-gatk.sh && rm /tmp/install-gatk.sh

# Copy pre-built virtual environment from build stage
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:/app/bin:$PATH"

COPY --from=build /src/methylation/ugbio_methylation/resources/meth_hg38-chr20_Lambda_pUC19.interval_list /app/interval_list/meth_hg38-chr20_Lambda_pUC19.interval_list

# Avoid matplotlib warnings
RUN mkdir -p /app/matplotlib_config && \
    chmod 777 /app/matplotlib_config
ENV MPLCONFIGDIR=/app/matplotlib_config

# Add user and grant sudo privileges
ARG USERNAME=app
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USERNAME}
