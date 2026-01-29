ARG BASE_IMAGE=ugbio_base
ARG OUT_DIR=/tmp/ugbio
FROM python:3.11-bullseye AS build

ARG OUT_DIR
ARG MODULE_DIR=cnv
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

#build ugbio_core + module
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

##################################################################
FROM $BASE_IMAGE

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    g++ \
    gfortran \
    procps \
    curl \
    libcurl4-openssl-dev \
    libxml2-dev \
    libssl-dev \
    bzip2 \
    xz-utils \
    libreadline-dev \
    libpcre2-dev \
    tar \
    bzip2 \
    libncurses5-dev \
    libncursesw5-dev \
    libopenblas-dev \
    libstdc++-10-dev \
    libparasail3 libparasail-dev \
    libhdf5-dev \
    tzdata \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && apt-get autoclean


# Download and install R
# Install R from CRAN's pre-compiled Debian packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    dirmngr \
    gnupg \
    apt-transport-https \
    ca-certificates \
    software-properties-common && \
    gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-key '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7' && \
    gpg --armor --export '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7' | tee /etc/apt/trusted.gpg.d/cran_debian_key.asc && \
    echo "deb https://cloud.r-project.org/bin/linux/debian bullseye-cran40/" | tee /etc/apt/sources.list.d/cran.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    r-base=4.5.1-1~bullseyecran.0 \
    r-base-dev=4.5.1-1~bullseyecran.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install samtools
ARG SAMTOOLS_VERSION=1.17

RUN wget https://github.com/samtools/samtools/releases/download/${SAMTOOLS_VERSION}/samtools-${SAMTOOLS_VERSION}.tar.bz2 && \
    tar -vxjf samtools-${SAMTOOLS_VERSION}.tar.bz2 && \
    cd samtools-${SAMTOOLS_VERSION} && \
    make -j $(nproc) && \
    make install && \
    cd .. && \
    rm -rf samtools-${SAMTOOLS_VERSION}.tar.bz2 samtools-${SAMTOOLS_VERSION}

## install ug-cn.mops (branch: ug.master):
RUN cd /opt && \
    if [ -d "cn.mops/.git" ]; then \
        cd cn.mops && git pull; \
    else \
        git clone --branch ug.master https://github.com/Ultimagen/cn.mops.git cn.mops; \
    fi


# Install R packages using apt-get where available (pre-compiled binaries from Debian)
# This is much faster than compiling from source
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    r-cran-cpp11 \
    r-cran-crayon \
    r-cran-formatr \
    r-cran-jsonlite \
    r-cran-magrittr \
    r-cran-r6 \
    r-cran-rcurl \
    r-cran-snow && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install remaining R packages that aren't available as Debian packages
# Using BiocManager for Bioconductor packages and install.packages for CRAN packages
RUN R -e "install.packages(c('argparse', 'BiocManager', 'bitops', 'codetools', 'findpython', 'futile.logger', 'futile.options', 'lambda.r'), repos='https://cloud.r-project.org/', Ncpus=4)"

# Install Bioconductor packages - using Ncpus for parallel compilation
RUN R -e "BiocManager::install(c('BiocGenerics', 'BiocParallel', 'Biostrings', 'Biobase', 'GenomeInfoDb', 'GenomicRanges', 'IRanges', 'S4Vectors', 'XVector', 'rhdf5', 'rhdf5filters', 'Rhtslib', 'Rsamtools', 'rtracklayer'), ask=FALSE, update=FALSE, Ncpus=4)"

# Remove unnecessary files
RUN rm -rf /tmp/* /var/tmp/* /usr/share/doc /usr/share/man /usr/share/info

# Install cn.mops packages from local
RUN R CMD INSTALL --preclean --no-multiarch --with-keep.source /opt/cn.mops/

# Download para_jalign binary from private jalign repository
ARG JALIGN_VERSION=1.4.1
RUN --mount=type=secret,id=github_token \
    if [ ! -f /run/secrets/github_token ]; then \
        echo "ERROR: Authentication required to download dependencies" && exit 1; \
    fi && \
    GITHUB_TOKEN=$(cat /run/secrets/github_token) && \
    ASSET_NAME="para_jalign-${JALIGN_VERSION}-debian-bullseye-x64" && \
    echo "Looking for asset: ${ASSET_NAME}" && \
    curl -sL \
        -H "Accept: application/vnd.github+json" \
        -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "https://api.github.com/repos/Ultimagen/jalign/releases/tags/${JALIGN_VERSION}" \
        -o /tmp/release_data.json && \
    ASSET_URL=$(cat /tmp/release_data.json | \
        jq -r ".assets[] | select(.name == \"${ASSET_NAME}\") | .url" | head -1) && \
    echo "Asset URL: ${ASSET_URL}" && \
    if [ -z "$ASSET_URL" ] || [ "$ASSET_URL" = "null" ]; then \
        echo "ERROR: Could not find asset '${ASSET_NAME}' in release ${JALIGN_VERSION}" && \
        echo "Available assets:" && \
        cat /tmp/release_data.json | jq -r '.assets[].name' && \
        exit 1; \
    fi && \
    curl -sLf \
        -H "Accept: application/octet-stream" \
        -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "${ASSET_URL}" \
        -o /opt/para_jalign && \
    chmod +x /opt/para_jalign && \
    rm /tmp/release_data.json

ARG OUT_DIR

RUN groupadd ugbio && useradd -m ugbio -g ugbio

COPY --from=build ${OUT_DIR} ${OUT_DIR}
RUN pip install ${OUT_DIR}/*.whl && rm -rf ${OUT_DIR}

#download cnvpytor additional necessary files (described here: https://github.com/abyzovlab/CNVpytor?tab=readme-ov-file#install-using-pip)
ENV PYTOR_DATA_DIR=/usr/share/cnvpytor_data
RUN mkdir -p $PYTOR_DATA_DIR
RUN cnvpytor -download

USER ugbio

WORKDIR /home/ugbio

COPY --from=build ./src/cnv/cnmops ./src/cnv/cnmops
