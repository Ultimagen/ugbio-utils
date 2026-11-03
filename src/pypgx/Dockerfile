# Pinned by digest: patched Python (CVE-2026-6100 fixed in 3.11.16) and a fresh build cache key
FROM python:3.11.17-slim-bookworm@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf AS build

ARG PYPGX_BRANCH=0.27.0
ARG PYPGX_BUNDLE_BRANCH=0.27.0

# Install build dependencies (git/wget/unzip stay in this stage only)
RUN apt-get update && \
    apt-get install --no-install-recommends -y git build-essential wget unzip && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment for pip install
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Pin numpy<2 and pandas<2.0 together to avoid ABI mismatch
# (pandas 1.5.x C extensions expect numpy 1.x dtype layout)
RUN pip install --no-cache-dir "numpy<2" "pandas<2.0"

# Install pypgx with numpy<2 constraint to prevent resolver upgrading numpy
RUN git clone https://github.com/sbslee/pypgx /opt/pypgx && \
    cd /opt/pypgx && \
    git checkout ${PYPGX_BRANCH} && \
    pip install --no-cache-dir "numpy<2" . && \
    rm -rf /opt/pypgx

# pypgx-bundle at the matching version, from the release archive (no .git history)
RUN wget -q -O /tmp/pypgx-bundle.tar.gz \
        https://github.com/sbslee/pypgx-bundle/archive/refs/tags/${PYPGX_BUNDLE_BRANCH}.tar.gz && \
    mkdir -p /opt/pypgx-bundle && \
    tar -xzf /tmp/pypgx-bundle.tar.gz -C /opt/pypgx-bundle --strip-components=1 && \
    rm /tmp/pypgx-bundle.tar.gz

# Install GATK (Ultima build of 4.7.0.0 with fixed netty, see docker/install-gatk.sh) into /opt/gatk
COPY docker/install-gatk.sh /tmp/install-gatk.sh
RUN bash /tmp/install-gatk.sh && rm /tmp/install-gatk.sh

##################################################################
FROM python:3.11.17-slim-bookworm@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install --no-install-recommends -y default-jre-headless bcftools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy pre-built virtual environment, pypgx-bundle and GATK from the build stage
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/pypgx-bundle /opt/pypgx-bundle
COPY --from=build /opt/gatk /opt/gatk
RUN ln -s /opt/gatk/gatk /usr/local/bin/gatk
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables for pypgx bundle
ENV PYPGX_BUNDLE=/opt/pypgx-bundle

# Create a script to run tests from docker CI
RUN echo '#!/bin/sh\necho "No tests to run"' > /usr/local/bin/run_tests \
&& chmod +x /usr/local/bin/run_tests

# Don't use root user
RUN groupadd ugbio && useradd -m ugbio -g ugbio
USER ugbio
WORKDIR /home/ugbio
