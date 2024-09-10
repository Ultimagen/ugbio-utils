FROM python:3.10-bullseye as build-base

ARG OUT_DIR=/tmp/ugbio
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

#build ugbio_core
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core

# ##################################################################
FROM ubuntu:22.04 as ugbio-base

ARG PYTHON_VERSION=3.10
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
python${PYTHON_VERSION} \
python${PYTHON_VERSION}-dev \
python${PYTHON_VERSION}-venv \
python3-pip \
&& apt-get clean && \
 rm -rf /var/lib/apt/lists/* && apt-get autoclean

# Create symbolic links for python
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
# install bcftools
RUN wget https://github.com/samtools/bcftools/releases/download/${BCFTOOLS_VERSION}/bcftools-${BCFTOOLS_VERSION}.tar.bz2 && \
 tar -vxjf bcftools-${BCFTOOLS_VERSION}.tar.bz2 && \
 rm bcftools-${BCFTOOLS_VERSION}.tar.bz2 && \
 cd bcftools-${BCFTOOLS_VERSION} && \
 make && \
 make install

#install required python packages
RUN pip install tables==3.9.2

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

ARG OUT_DIR=/tmp/ugbio

COPY --from=build-base $OUT_DIR $OUT_DIR

RUN pip install ${OUT_DIR}/*.whl 