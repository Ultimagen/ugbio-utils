# BASE_IMAGE is injected by build-workspace-docker.yml (defaults to DEFAULT_BASE_IMAGE)
ARG BASE_IMAGE
ARG BASE_BUILD_IMAGE
ARG OUT_DIR=/tmp/ugbio
FROM $BASE_BUILD_IMAGE AS build

ARG OUT_DIR
ARG MODULE_DIR=mrd
RUN mkdir -p $OUT_DIR

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

# Bake the image tag in so generate_report can display it in the report.
# Pass --build-arg IMAGE_TAG=<tag> from CI (docker-build-push.yml).
ARG IMAGE_TAG=unknown
ENV UGBIO_IMAGE_TAG=${IMAGE_TAG}

RUN apt-get update && apt-get upgrade -y && apt-get install --no-install-recommends -y \
    openjdk-17-jre-headless `# for gatk` \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && apt-get autoclean

# Install GATK (Ultima build of 4.7.0.0 with fixed netty, see docker/install-gatk.sh)
COPY docker/install-gatk.sh /tmp/install-gatk.sh
RUN bash /tmp/install-gatk.sh && rm /tmp/install-gatk.sh

RUN groupadd ugbio && useradd -m ugbio -g ugbio

# Copy pre-built virtual environment from build stage (no compilation in runtime)
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

USER ugbio

WORKDIR /home/ugbio
