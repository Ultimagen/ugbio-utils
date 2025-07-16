
FROM python:3.11-bullseye AS build

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
ARG USERNAME=app
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Install apt-get dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    wget \
    libgomp1 \
    unzip \
    procps `# for monitoring script memory usage` \
    sysstat `# for monitoring script IO metrics` \
    sudo `# install sudo` \
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

# Add user and grant sudo privileges
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USERNAME}
