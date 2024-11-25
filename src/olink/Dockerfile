
FROM python:3.11-bullseye AS build

ARG OUT_DIR=/tmp/ugbio
ARG MODULE_DIR=olink_test
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

#build ugbio_core + module
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

##################################################################
FROM python:3.11-slim

ARG OUT_DIR=/tmp/ugbio
ARG USERNAME=app
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Install apt-get dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    wget \
    libgomp1 \
    unzip \
    bzip2 \
    make \
    gcc \
    g++ \
    libbz2-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    liblzma-dev\
    libcurl4-openssl-dev\
    curl \
    procps `# for monitoring script memory usage` \
    sysstat `# for monitoring script IO metrics` \
    sudo `# install sudo` \
    ; \
    apt-get autoclean && rm -rf /var/lib/apt/lists/*


# Install module and dependencies (like core)
COPY --from=build $OUT_DIR $OUT_DIR

RUN pip install ${OUT_DIR}/*.whl

# Add user and grant sudo privileges
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# install htslib
RUN wget https://github.com/samtools/htslib/releases/download/1.9/htslib-1.9.tar.bz2 &&\
    tar -vxjf htslib-1.9.tar.bz2 &&\
    cd htslib-1.9 &&\
    make

# Install samtools
ARG SAMTOOLS_VERSION=1.17

RUN wget https://github.com/samtools/samtools/releases/download/${SAMTOOLS_VERSION}/samtools-${SAMTOOLS_VERSION}.tar.bz2 && \
    tar -vxjf samtools-${SAMTOOLS_VERSION}.tar.bz2 && \
    cd samtools-${SAMTOOLS_VERSION} && \
    make -j $(nproc) && \
    make install && \
    cd .. && \
    rm -rf samtools-${SAMTOOLS_VERSION}.tar.bz2 samtools-${SAMTOOLS_VERSION}