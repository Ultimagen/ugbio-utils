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

# install picard
WORKDIR /usr/local/bin
RUN wget https://github.com/broadinstitute/picard/releases/download/${picard_version}/picard-tools-${picard_version}.zip
RUN unzip picard-tools-${picard_version}.zip
RUN mkdir -p /usr/local/bin/picard
RUN mv /usr/local/bin/picard-tools-${picard_version}/* /usr/local/bin/picard/
RUN chmod 0644 /usr/local/bin/picard/SamToFastq.jar
RUN rm -rf /usr/local/bin/picard-tools-${picard_version}/
