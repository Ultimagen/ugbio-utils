FROM python:3.11-bookworm AS build

ARG OUT_DIR=/tmp/ugbio
ARG MODULE_DIR=single_cell
RUN mkdir -p $OUT_DIR

# Create virtual environment for pip install
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./src/ ./src/

#build ugbio_core + module
RUN pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/core
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

# Install wheels in build stage
RUN pip install ${OUT_DIR}/*.whl

##################################################################
FROM python:3.11-slim-bookworm

ARG USERNAME=app
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Install runtime-only dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    wget \
    libgomp1 \
    unzip \
    procps \
    sysstat \
    sudo \
    ; \
    apt-get autoclean && rm -rf /var/lib/apt/lists/*

# Install STAR
RUN wget -O- https://github.com/alexdobin/STAR/archive/2.7.10a.tar.gz \
    | tar xfz - STAR-2.7.10a/bin/Linux_x86_64/STAR \
    && mv STAR-2.7.10a/bin/Linux_x86_64/STAR /usr/local/bin \
    && rm -rf STAR-2.7.10a

# Copy pre-built virtual environment from build stage
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Include licenses
COPY ["licenses/star LICENSE.txt", "."]
COPY ["licenses/Ultima UGVC - EULA.pdf", "."]

# Add user and grant sudo privileges
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USERNAME}
