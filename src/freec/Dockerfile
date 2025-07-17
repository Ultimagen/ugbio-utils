ARG BASE_IMAGE=ugbio_base
FROM python:3.11-bullseye AS build

ARG OUT_DIR=/tmp/ugbio
ARG MODULE_DIR=freec
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

# build module
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

##################################################################
FROM $BASE_IMAGE

RUN apt-get update && apt-get install --no-install-recommends -y \
    openjdk-17-jre-headless `# for gatk` \
    git \
    wget \
    unzip \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && apt-get autoclean

# Download and install gatk
ARG GATK_VERSION=4.6.1.0
RUN wget https://github.com/broadinstitute/gatk/releases/download/${GATK_VERSION}/gatk-${GATK_VERSION}.zip && \
    unzip gatk-${GATK_VERSION}.zip && \
    mv gatk-${GATK_VERSION} /opt/gatk && \
    ln -s /opt/gatk/gatk /usr/local/bin/gatk && \
    rm gatk-${GATK_VERSION}.zip

# install samtools
ARG SAMTOOLS_VERSION=1.20
WORKDIR /usr/local/bin
RUN wget https://github.com/samtools/samtools/releases/download/${SAMTOOLS_VERSION}/samtools-${SAMTOOLS_VERSION}.tar.bz2
RUN tar -xjf /usr/local/bin/samtools-${SAMTOOLS_VERSION}.tar.bz2 -C /usr/local/bin/
RUN cd /usr/local/bin/samtools-${SAMTOOLS_VERSION}/ && ./configure
RUN cd /usr/local/bin/samtools-${SAMTOOLS_VERSION}/ && make
RUN cd /usr/local/bin/samtools-${SAMTOOLS_VERSION}/ && make install

# Download and install bigWigToBedGraph
RUN wget http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64.v385/bigWigToBedGraph -O /usr/local/bin/bigWigToBedGraph \
    && chmod +x /usr/local/bin/bigWigToBedGraph

## clone FREEC (branch: master):
ARG FREEC_BRANCH=master
WORKDIR /usr/local/bin
RUN git clone --branch ${FREEC_BRANCH} https://github.com/Ultimagen/FREEC FREEC
RUN cp FREEC/src/freec /freec
RUN ln -s /freec /usr/bin/freec

ARG OUT_DIR=/tmp/ugbio

# Create a non-root user and group
RUN groupadd --system ugbio && useradd --system --create-home --gid ugbio ugbio -d /home/ugbio

# Set ownership for writable directories
RUN chown -R ugbio:ugbio /opt /usr/local/bin

USER ugbio

WORKDIR /home/ugbio

ARG OUT_DIR=/tmp/ugbio

# Copy whl and install module
COPY --from=build $OUT_DIR $OUT_DIR

ENV PATH="/home/ugbio/.local/bin:${PATH}"

RUN pip install --user ${OUT_DIR}/*.whl

# for runnig CI (can be removed if pytest is installed as part of the venv)
RUN pip install --user pytest
