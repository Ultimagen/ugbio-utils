# BASE_IMAGE is injected by build-workspace-docker.yml (defaults to DEFAULT_BASE_IMAGE)
ARG BASE_IMAGE
ARG OUT_DIR=/tmp/ugbio
FROM python:3.11-bullseye AS build

ARG OUT_DIR
ARG MODULE_DIR=mrd
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

#build all depended + module
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core
RUN python -m build --outdir $OUT_DIR --wheel ./src/ppmseq
RUN python -m build --outdir $OUT_DIR --wheel ./src/featuremap
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

##################################################################
FROM ${BASE_IMAGE}

# Bake the image tag in so generate_report can display it in the report.
# Pass --build-arg IMAGE_TAG=<tag> from CI (docker-build-push.yml).
ARG IMAGE_TAG=unknown
ENV UGBIO_IMAGE_TAG=${IMAGE_TAG}

RUN apt-get update && apt-get install --no-install-recommends -y \
    openjdk-17-jre-headless `# for gatk` \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && apt-get autoclean

ARG PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive
ARG BCFTOOLS_INSTALL_DIR=/opt/bcftools
ARG BCFTOOLS_VERSION=1.20

# Download and install gatk
ARG GATK_VERSION=4.5.0.0
RUN wget https://github.com/broadinstitute/gatk/releases/download/${GATK_VERSION}/gatk-${GATK_VERSION}.zip && \
    unzip gatk-${GATK_VERSION}.zip && \
    mv gatk-${GATK_VERSION} /opt/gatk && \
    ln -s /opt/gatk/gatk /usr/local/bin/gatk && \
    rm gatk-${GATK_VERSION}.zip

ARG OUT_DIR

RUN groupadd ugbio && useradd -m ugbio -g ugbio

COPY --from=build ${OUT_DIR} ${OUT_DIR}
RUN pip install ${OUT_DIR}/*.whl && rm -rf ${OUT_DIR}

USER ugbio

WORKDIR /home/ugbio
