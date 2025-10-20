ARG BASE_IMAGE=ugbio_base
ARG OUT_DIR=/tmp/ugbio
FROM python:3.11-bullseye AS build

ARG OUT_DIR
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
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && apt-get autoclean

# Download and install gatk
ARG GATK_VERSION=4.6.1.0
RUN wget https://github.com/broadinstitute/gatk/releases/download/${GATK_VERSION}/gatk-${GATK_VERSION}.zip && \
    unzip gatk-${GATK_VERSION}.zip && \
    mv gatk-${GATK_VERSION} /opt/gatk && \
    ln -s /opt/gatk/gatk /usr/local/bin/gatk && \
    rm gatk-${GATK_VERSION}.zip

# Download and install bigWigToBedGraph
RUN wget http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64.v385/bigWigToBedGraph -O /usr/local/bin/bigWigToBedGraph \
    && chmod +x /usr/local/bin/bigWigToBedGraph

## clone FREEC (branch: master):
ARG FREEC_BRANCH=master
WORKDIR /usr/local/bin
RUN git clone --branch ${FREEC_BRANCH} https://github.com/Ultimagen/FREEC FREEC
RUN cp FREEC/src/freec /freec
RUN ln -s /freec /usr/bin/freec

ARG OUT_DIR

RUN groupadd ugbio && useradd -m ugbio -g ugbio

COPY --from=build ${OUT_DIR} ${OUT_DIR}
RUN pip install ${OUT_DIR}/*.whl && rm -rf ${OUT_DIR}

USER ugbio

WORKDIR /home/ugbio

# for runnig CI (can be removed if pytest is installed as part of the venv)
RUN pip install --user pytest
