FROM python:3.10-bullseye as build

ARG OUT_DIR=/tmp/ugbio
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

#build ugbio_core
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core

##################################################################
FROM ubuntu:22.04

ARG PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
 wget \
 ca-certificates \
 perl \
 bzip2 \
 autoconf \
 automake \
 make \
 gcc \
 git \
 zlib1g-dev \
 libbz2-dev \
 liblzma-dev \
 libcurl4-gnutls-dev \
 libssl-dev \
 libperl-dev \
 libgsl0-dev \
python${PYTHON_VERSION} \
python${PYTHON_VERSION}-dev \
python${PYTHON_VERSION}-venv \
python3-pip \
&& apt-get clean && \
 rm -rf /var/lib/apt/lists/* && apt-get autoclean

# Create symbolic links for python
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

#install required python packages
RUN pip install tables==3.9.2

ARG OUT_DIR=/tmp/ugbio

COPY --from=build $OUT_DIR $OUT_DIR

RUN pip install ${OUT_DIR}/*.whl 
