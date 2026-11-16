# BASE_IMAGE is injected by build-workspace-docker.yml (defaults to DEFAULT_BASE_IMAGE)
ARG BASE_IMAGE
ARG BASE_BUILD_IMAGE
ARG OUT_DIR=/tmp/ugbio
# R from Posit's prebuilt Debian 12 package (installs to /opt/R/<version>), pinned by SHA-256. CRAN's own Debian
# repository is not used: its signing key expired on 2026-11-15, so apt rejects it.
ARG R_VERSION=4.5.3
ARG R_DEB_SHA256=66bfa95bc41608867cdff09dd2798acf5d6b39fe1b6c756b387cc25a1700bf6d
FROM $BASE_BUILD_IMAGE AS build

ARG OUT_DIR
ARG R_VERSION
ARG R_DEB_SHA256
ARG MODULE_DIR=cnv
RUN mkdir -p $OUT_DIR

# Create virtual environment for pip install
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./src/ ./src/

# build module
RUN pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

# Install wheels — C extensions compile here, not in runtime.
# pywfa and pyabpoa (truvari, used by merge_cnv_sv) are built without -march=native, so the image runs on hosts
# without AVX-512 (a native build exits with SIGILL there)
RUN PYWFA_MARCH=x86-64-v3 pip install --no-binary pywfa "pywfa>=0.6.0"
RUN AVX2=1 pip install ${OUT_DIR}/*.whl
RUN pip install pytest

# Optional cn.mops path (src/cnv/cnmops): R and the R packages its scripts load, compiled here into /opt/rlib
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://cdn.posit.co/r/debian-12/pkgs/r-${R_VERSION}_1_amd64.deb', '/tmp/r.deb')" && \
    echo "${R_DEB_SHA256}  /tmp/r.deb" | sha256sum -c - && \
    apt-get update && apt-get install -y --no-install-recommends /tmp/r.deb libssl-dev libxml2-dev && \
    ln -s /opt/R/${R_VERSION}/bin/R /usr/local/bin/R && ln -s /opt/R/${R_VERSION}/bin/Rscript /usr/local/bin/Rscript && \
    rm -rf /var/lib/apt/lists/* /tmp/r.deb
RUN mkdir -p /opt/rlib && \
    Rscript -e "install.packages(c('argparse', 'magrittr', 'BiocManager'), lib='/opt/rlib', repos='https://cloud.r-project.org', Ncpus=4)" && \
    R_LIBS=/opt/rlib Rscript -e "BiocManager::install(c('GenomicRanges', 'rhdf5', 'rtracklayer', 'Rsamtools', 'Biobase', 'GenomeInfoDb'), lib='/opt/rlib', ask=FALSE, update=FALSE, Ncpus=4)" && \
    R_LIBS=/opt/rlib Rscript -e "stopifnot(all(sapply(c('argparse', 'magrittr', 'GenomicRanges', 'rhdf5', 'rtracklayer', 'Rsamtools', 'Biobase', 'GenomeInfoDb'), requireNamespace)))"
# Picard (estimate_ploidy: IntervalFile converts interval_list to BED with it); Java is installed in the runtime stage
ARG PICARD_VERSION=3.4.0
RUN mkdir -p /opt/picard && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/broadinstitute/picard/releases/download/${PICARD_VERSION}/picard.jar', '/opt/picard/picard.jar')"
RUN git clone --depth 1 --branch ug.master https://github.com/Ultimagen/cn.mops.git /tmp/cn.mops && \
    R_LIBS=/opt/rlib R CMD INSTALL --preclean --no-multiarch -l /opt/rlib /tmp/cn.mops && \
    rm -rf /tmp/cn.mops

##################################################################
FROM $BASE_IMAGE
ARG R_VERSION

# R runtime only for the cn.mops scripts: R is copied from the build stage, with the shared libraries it and the
# R packages load (no compilers or -dev headers); cairo / pango for the PNG of normalize_reads_count.R
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    libopenblas0-pthread libgfortran5 libgomp1 libreadline8 libpcre2-8-0 libcurl4 libicu72 libdeflate0 libbz2-1.0 \
    liblzma5 libzstd1 zlib1g libxml2 libssl3 libpng16-16 libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libtiff6 \
    libx11-6 libxt6 fonts-dejavu-core \
    openjdk-17-jre-headless && \
    rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/R /opt/R
RUN ln -s /opt/R/${R_VERSION}/bin/R /usr/local/bin/R && ln -s /opt/R/${R_VERSION}/bin/Rscript /usr/local/bin/Rscript

RUN printf '#!/bin/sh\nexec java -jar /opt/picard/picard.jar "$@"\n' > /usr/local/bin/picard && chmod +x /usr/local/bin/picard
RUN groupadd ugbio && useradd -m ugbio -g ugbio

# Copy pre-built virtual environment and R library from build stage (no compilation in runtime)
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/rlib /opt/rlib
COPY --from=build /opt/picard /opt/picard
ENV PATH="/opt/venv/bin:$PATH"
ENV R_LIBS_SITE=/opt/rlib

USER ugbio

WORKDIR /home/ugbio

COPY --from=build ./src/cnv/cnmops ./src/cnv/cnmops
