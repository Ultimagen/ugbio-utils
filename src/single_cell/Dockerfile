
FROM python:3.11-bullseye as build

ARG OUT_DIR=/tmp/ugbio
ARG MODULE_DIR=single_cell
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

#build ugbio_core + module
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

##################################################################
FROM python:3.11-slim

ARG OUT_DIR=/tmp/ugbio

# Install apt-get dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    wget \
    libgomp1 \
    unzip \
    procps `# for monitoring script memory usage` \
    sysstat `# for monitoring script IO metrics` \
    ; \
    apt-get autoclean && rm -rf /var/lib/apt/lists/*

# Install STAR
RUN wget -O- https://github.com/alexdobin/STAR/archive/2.7.10a.tar.gz \
    | tar xfz - STAR-2.7.10a/bin/Linux_x86_64/STAR \
    && mv STAR-2.7.10a/bin/Linux_x86_64/STAR /usr/local/bin \
    && rm -rf STAR-2.7.10a

# Install module and dependencies (like core)
COPY --from=build $OUT_DIR $OUT_DIR

RUN pip install ${OUT_DIR}/*.whl

# Include licenses
COPY ["licenses/star LICENSE.txt", "."]
COPY ["licenses/Ultima UGVC - EULA.pdf", "."]