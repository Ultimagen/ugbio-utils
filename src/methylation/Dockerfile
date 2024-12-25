FROM python:3.11-bullseye AS build

ARG OUT_DIR=/tmp/ugbio
ARG MODULE_DIR=methylation
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

# build module
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

##################################################################
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    openjdk-17-jre-headless `# for gatk` \
    build-essential `# for make`\
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libhts-dev \
    git \
    wget \
    unzip \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && apt-get autoclean

#Install libBigWig & MethylDackel
WORKDIR /app
RUN git clone https://github.com/dpryan79/libBigWig.git && \
    cd libBigWig && \
    make && \
    make install && \
    cp libBigWig.a /usr/local/lib/

RUN git clone https://github.com/dpryan79/MethylDackel.git && \
    cd MethylDackel && \
    make LIBBIGWIG="/usr/local/lib/libBigWig.a" && \
    mkdir -p /app/bin && \
    make install prefix=/app/bin/
ENV PATH="/app/bin:$PATH"

# Download and install gatk
ARG GATK_VERSION=4.6.0.0
RUN wget https://github.com/broadinstitute/gatk/releases/download/${GATK_VERSION}/gatk-${GATK_VERSION}.zip && \
    unzip gatk-${GATK_VERSION}.zip && \
    mv gatk-${GATK_VERSION} /opt/gatk && \
    ln -s /opt/gatk/gatk /usr/local/bin/gatk && \
    rm gatk-${GATK_VERSION}.zip

ARG OUT_DIR=/tmp/ugbio

# Install module
COPY --from=build $OUT_DIR $OUT_DIR

RUN pip install ${OUT_DIR}/*.whl

COPY --from=build ./src/methylation/ugbio_methylation/reports /src/methylation/ugbio_methylation/reports
COPY --from=build ./src/methylation/ugbio_methylation/resources/meth_hg38-chr20_Lambda_pUC19.interval_list /app/interval_list/meth_hg38-chr20_Lambda_pUC19.interval_list

# for runnig CI (can be removed if pytest is installed as part of the venv)
RUN pip install pytest
