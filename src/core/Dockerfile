# ARG BASE_IMAGE=ugbio_base
ARG BASE_IMAGE=337532070941.dkr.ecr.us-east-1.amazonaws.com/ugbio_base:1.3.0
FROM python:3.11-bullseye AS build

RUN add-apt-repository ppa:webupd8team/java && \
 apt-get update -y && apt-get install -y\
 openjdk-8-jre && \
 apt-get clean && \
 rm -rf /var/lib/apt/lists/* && \
 apt-get autoclean

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
RUN wget https://github.com/broadinstitute/picard/releases/download/${picard_version}/picard.jar
RUN chmod +x /usr/local/bin/picard.jar
RUN echo 'alias picard="java -jar /usr/local/bin/picard.jar"' >> ~/.bashrc
