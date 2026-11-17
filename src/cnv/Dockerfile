# BASE_IMAGE is injected by build-workspace-docker.yml (defaults to DEFAULT_BASE_IMAGE)
ARG BASE_IMAGE
ARG OUT_DIR=/tmp/ugbio

FROM python:3.11-bookworm AS build

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

# Install basic dependencies; upgrade first to pick up security fixes missing from the base image
# (e.g. ugbio_base:1.8.6 ships perl 5.36.0-7+deb12u3, CRITICAL CVE-2026-13221/-42496/-8376, fixed in deb12u4)
RUN apt-get update && apt-get upgrade -y && \
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
    libstdc++-12-dev \
    libhdf5-dev \
    libparasail8 \
    tzdata \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && apt-get autoclean


# Install R from Posit's prebuilt Debian 12 package (installs to /opt/R/<version>), pinned by SHA-256.
# CRAN's own Debian repository is not used: its signing key expired on 2026-11-15, so apt rejects it.
ARG R_VERSION=4.5.3
ARG R_DEB_SHA256=66bfa95bc41608867cdff09dd2798acf5d6b39fe1b6c756b387cc25a1700bf6d
RUN curl -fsSL -o /tmp/r.deb https://cdn.posit.co/r/debian-12/pkgs/r-${R_VERSION}_1_amd64.deb && \
    echo "${R_DEB_SHA256}  /tmp/r.deb" | sha256sum -c - && \
    apt-get update && apt-get install -y --no-install-recommends /tmp/r.deb && \
    ln -s /opt/R/${R_VERSION}/bin/R /usr/local/bin/R && \
    ln -s /opt/R/${R_VERSION}/bin/Rscript /usr/local/bin/Rscript && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/r.deb

# Install samtools
ARG SAMTOOLS_VERSION=1.17

RUN wget https://github.com/samtools/samtools/releases/download/${SAMTOOLS_VERSION}/samtools-${SAMTOOLS_VERSION}.tar.bz2 && \
    tar -vxjf samtools-${SAMTOOLS_VERSION}.tar.bz2 && \
    cd samtools-${SAMTOOLS_VERSION} && \
    make -j $(nproc) && \
    make install && \
    cd .. && \
    rm -rf samtools-${SAMTOOLS_VERSION}.tar.bz2 samtools-${SAMTOOLS_VERSION}

# Install Picard (IntervalFile converts interval_list to BED with it)
ARG PICARD_VERSION=3.4.0
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends --reinstall ca-certificates openjdk-17-jre-headless \
    && apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get autoclean
RUN mkdir -p /opt/picard && \
    wget -O /opt/picard/picard.jar https://github.com/broadinstitute/picard/releases/download/${PICARD_VERSION}/picard.jar && \
    chmod +x /opt/picard/picard.jar
RUN echo '#!/bin/sh\njava -jar /opt/picard/picard.jar "$@"' > /usr/local/bin/picard && \
    chmod +x /usr/local/bin/picard

## install ug-cn.mops (branch: ug.master):
RUN cd /opt && \
    if [ -d "cn.mops/.git" ]; then \
        cd cn.mops && git pull; \
    else \
        git clone --branch ug.master https://github.com/Ultimagen/cn.mops.git cn.mops; \
    fi


# Install CRAN packages (not from Debian's r-cran-* packages: those depend on Debian's own r-base-core)
# Using BiocManager for Bioconductor packages and install.packages for CRAN packages
RUN R -e "install.packages(c('cpp11', 'crayon', 'formatR', 'jsonlite', 'magrittr', 'R6', 'RCurl', 'snow', 'argparse', 'BiocManager', 'bitops', 'codetools', 'findpython', 'futile.logger', 'futile.options', 'lambda.r'), repos='https://cloud.r-project.org/', Ncpus=4)"

# Install Bioconductor packages - using Ncpus for parallel compilation
RUN R -e "BiocManager::install(c('BiocGenerics', 'BiocParallel', 'Biostrings', 'Biobase', 'GenomeInfoDb', 'GenomicRanges', 'IRanges', 'S4Vectors', 'XVector', 'rhdf5', 'rhdf5filters', 'Rhtslib', 'Rsamtools', 'GenomicAlignments', 'BSgenome', 'VariantAnnotation'), ask=FALSE, Ncpus=4)"

# Remove unnecessary files
RUN rm -rf /tmp/* /var/tmp/* /usr/share/doc /usr/share/man /usr/share/info

# Install cn.mops packages from local
RUN R CMD INSTALL --preclean --no-multiarch --with-keep.source /opt/cn.mops/

# Download para_jalign binary from private jalign repository.
# The bookworm build links the distro libparasail8 (parasail 2.6) and uses its striped kernels,
# since the parasail 2.6 diag kernels return corrupted scores/cigars (DATA-9903)
ARG JALIGN_VERSION=1.4.2
RUN --mount=type=secret,id=github_token \
    if [ ! -f /run/secrets/github_token ]; then \
        echo "ERROR: Authentication required to download dependencies" && exit 1; \
    fi && \
    GITHUB_TOKEN=$(cat /run/secrets/github_token) && \
    ASSET_NAME="para_jalign-${JALIGN_VERSION}-debian-bookworm-x64" && \
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

# Build pywfa and pyabpoa without -march=native, so the image runs on hosts without AVX-512
RUN PYWFA_MARCH=x86-64-v3 pip install --no-binary pywfa "pywfa>=0.6.0"
RUN AVX2=1 pip install ${OUT_DIR}/*.whl && rm -rf ${OUT_DIR}

#download cnvpytor additional necessary files (described here: https://github.com/abyzovlab/CNVpytor?tab=readme-ov-file#install-using-pip)
ENV PYTOR_DATA_DIR=/usr/share/cnvpytor_data
RUN mkdir -p $PYTOR_DATA_DIR
RUN cnvpytor -download

USER ugbio

WORKDIR /home/ugbio

COPY --from=build ./src/cnv/cnmops ./src/cnv/cnmops
