ARG BASE_IMAGE=ugbio_base
FROM python:3.11-bullseye AS build

ARG OUT_DIR=/tmp/ugbio
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

#build ugbio_core
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core

# ##################################################################
FROM $BASE_IMAGE

ARG OUT_DIR=/tmp/ugbio

COPY --from=build $OUT_DIR $OUT_DIR

RUN WHEEL_FILE=$(find ${OUT_DIR} -name "*.whl") && \
    pip install "${WHEEL_FILE}[vcfbed]"

# set the environment variables
ENV picard_version=3.2.0

# install picard
WORKDIR /usr/local/bin
RUN wget https://github.com/broadinstitute/picard/releases/download/3.2.0/picard.jar
RUN chmod +x /usr/local/bin/picard.jar
RUN alias picard="java -jar /usr/local/bin/picard.jar"
