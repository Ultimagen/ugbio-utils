# BASE_IMAGE is injected by build-workspace-docker.yml (defaults to DEFAULT_BASE_IMAGE)
ARG BASE_IMAGE
ARG BASE_BUILD_IMAGE
ARG OUT_DIR=/tmp/ugbio
# R from CRAN's Debian packages: the same version in both stages, so the R library built here runs in the runtime
ARG R_VERSION=4.5.3-1~bookwormcran.0
FROM $BASE_BUILD_IMAGE AS build

ARG OUT_DIR
ARG R_VERSION
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
RUN apt-get update && apt-get install -y --no-install-recommends gnupg dirmngr ca-certificates && \
    gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-key '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7' && \
    gpg --armor --export '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7' > /etc/apt/trusted.gpg.d/cran_debian_key.asc && \
    echo "deb https://cloud.r-project.org/bin/linux/debian bookworm-cran40/" > /etc/apt/sources.list.d/cran.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    r-base-core=${R_VERSION} r-base-dev=${R_VERSION} gfortran libssl-dev libxml2-dev && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/rlib && \
    Rscript -e "install.packages(c('argparse', 'magrittr', 'BiocManager'), lib='/opt/rlib', repos='https://cloud.r-project.org', Ncpus=4)" && \
    R_LIBS=/opt/rlib Rscript -e "BiocManager::install(c('GenomicRanges', 'rhdf5', 'rtracklayer', 'Rsamtools', 'Biobase', 'GenomeInfoDb'), lib='/opt/rlib', ask=FALSE, update=FALSE, Ncpus=4)" && \
    R_LIBS=/opt/rlib Rscript -e "stopifnot(all(sapply(c('argparse', 'magrittr', 'GenomicRanges', 'rhdf5', 'rtracklayer', 'Rsamtools', 'Biobase', 'GenomeInfoDb'), requireNamespace)))"
RUN git clone --depth 1 --branch ug.master https://github.com/Ultimagen/cn.mops.git /tmp/cn.mops && \
    R_LIBS=/opt/rlib R CMD INSTALL --preclean --no-multiarch -l /opt/rlib /tmp/cn.mops && \
    rm -rf /tmp/cn.mops

##################################################################
FROM $BASE_IMAGE
ARG R_VERSION

# R runtime only (no compilers or -dev headers) for the cn.mops scripts; gnupg is removed once the key is in
RUN apt-get update && apt-get install -y --no-install-recommends gnupg dirmngr ca-certificates && \
    gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-key '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7' && \
    gpg --armor --export '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7' > /etc/apt/trusted.gpg.d/cran_debian_key.asc && \
    echo "deb https://cloud.r-project.org/bin/linux/debian bookworm-cran40/" > /etc/apt/sources.list.d/cran.list && \
    apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends r-base-core=${R_VERSION} libxml2 && \
    apt-get purge -y gnupg dirmngr && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /root/.gnupg

RUN groupadd ugbio && useradd -m ugbio -g ugbio

# Copy pre-built virtual environment and R library from build stage (no compilation in runtime)
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/rlib /opt/rlib
ENV PATH="/opt/venv/bin:$PATH"
ENV R_LIBS_SITE=/opt/rlib

USER ugbio

WORKDIR /home/ugbio

COPY --from=build ./src/cnv/cnmops ./src/cnv/cnmops
