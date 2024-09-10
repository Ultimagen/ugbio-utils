FROM python:3.10-bullseye as build

ARG OUT_DIR=/tmp/ugbio
ARG MODULE_DIR=cnv
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

#build ugbio_core + module
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

##################################################################
FROM 337532070941.dkr.ecr.us-east-1.amazonaws.com/ugbio-base

ENV TZ=Etc/UTC

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y \
    g++ \
    gfortran \
    procps \
    libcurl4-openssl-dev \
    libxml2-dev \
    libssl-dev \
    bzip2 \
    xz-utils \
    libreadline-dev \
    libpcre2-dev \
    make \
    tar \
    git \
    bzip2 \
    libncurses5-dev \
    libncursesw5-dev \
    libopenblas-dev \
    libstdc++-10-dev \
    tzdata \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and install R
RUN wget https://cloud.r-project.org/src/base/R-4/R-4.3.1.tar.gz && \
    tar -xzf R-4.3.1.tar.gz && \
    cd R-4.3.1 && \
    ./configure --with-x=no --enable-R-shlib && \
    make && \
    make install && \
    cd .. && \
    rm -rf R-4.3.1 R-4.3.1.tar.gz

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
   git clone --branch ug.master https://github.com/Ultimagen/cn.mops.git cn.mops && \
   cd cn.mops

# Install additional R dependencies
RUN R -e "install.packages(c('argparse', 'BiocManager'), repos='https://cran.r-project.org')"

# Install Bioconductor packages one by one, ensuring dependencies are met
RUN R -e "BiocManager::install(c('BiocGenerics', 'BiocParallel', 'Biostrings', 'Biobase'))"
RUN R -e "BiocManager::install(c('GenomeInfoDb', 'GenomicRanges', 'IRanges', 'S4Vectors', 'XVector', 'zlibbioc'))"
RUN R -e "BiocManager::install(c('rhdf5', 'rhdf5filters', 'rhdf5lib', 'Rhtslib', 'Rsamtools'))"
RUN R -e "BiocManager::install('exomeCopy')"

# Install other necessary R packages
RUN R -e "install.packages(c('bitops', 'codetools', 'cpp11', 'crayon', 'findpython', 'formatR', 'futile.logger', 'futile.options', 'jsonlite', 'lambda.r', 'magrittr', 'R6', 'RCurl', 'snow'), repos='https://cran.r-project.org')"

# Remove unnecessary files
RUN rm -rf /tmp/* /var/tmp/* /usr/share/doc /usr/share/man /usr/share/info

# Install cn.mops packages from local
RUN R CMD INSTALL --preclean --no-multiarch --with-keep.source /opt/cn.mops/

ARG OUT_DIR=/tmp/ugbio

COPY --from=build $OUT_DIR $OUT_DIR

RUN pip install ${OUT_DIR}/*.whl

COPY --from=build ./src/cnv/cnmops ./src/cnv/cnmops