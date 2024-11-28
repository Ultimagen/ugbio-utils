ARG BASE_IMAGE=ugbio_base
FROM python:3.11-bullseye AS build

# RUN add-apt-repository ppa:webupd8team/java && \
#  apt-get update -y && apt-get install -y\
#  openjdk-8-jre && \
#  apt-get clean && \
#  rm -rf /var/lib/apt/lists/* && \
#  apt-get autoclean

# # Install OpenJDK 8 from AdoptOpenJDK
# RUN apt-get update --fix-missing \
#  && apt-get install -y wget gnupg \
#  && wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public | apt-key add - \
#  && add-apt-repository --yes https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/ \
#  && apt-get update \
#  && apt-get install -y adoptopenjdk-8-hotspot \
#  && rm -rf /var/lib/apt/lists/* \
#  && apt-get clean

#RUN apt-get update && apt-get install -y curl zip unzip
#RUN curl -s 'https://get.sdkman.io' | bash
#RUN /bin/bash -c "source $HOME/.sdkman/bin/sdkman-init.sh; sdk version; sdk install java 8.0.302-open"
# RUN source "$HOME/.sdkman/bin/sdkman-init.sh"
# ENV PATH=$PATH:/root/.sdkman/candidates/java/current/

ARG OUT_DIR=/tmp/ugbio
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

#build ugbio_core
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core

# ##################################################################
FROM $BASE_IMAGE
# Add OpenJDK and download Picard
ARG PICARD_VERSION=3.3.0
ARG PICARD_INSTALL_DIR=/opt/picard

RUN apt-get update && apt-get install -y \
    openjdk-11-jre-headless \
    && apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get autoclean && \
    mkdir -p ${PICARD_INSTALL_DIR} && \
    wget -O ${PICARD_INSTALL_DIR}/picard.jar https://github.com/broadinstitute/picard/releases/download/${PICARD_VERSION}/picard.jar

ARG OUT_DIR=/tmp/ugbio

COPY --from=build $OUT_DIR $OUT_DIR

RUN WHEEL_FILE=$(find ${OUT_DIR} -name "*.whl") && \
    pip install "${WHEEL_FILE}[vcfbed]"

## set the environment variables
#ENV picard_version=3.2.0
#
## install picard
#WORKDIR /usr/local/bin
#RUN wget https://github.com/broadinstitute/picard/releases/download/${picard_version}/picard.jar
#RUN chmod +x /usr/local/bin/picard.jar
#RUN echo 'alias picard="java -jar /usr/local/bin/picard.jar"' >> ~/.bashrc
