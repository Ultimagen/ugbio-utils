FROM python:3.11-slim AS build

ARG PYPGX_BRANCH=0.26.0

# Install build dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment for pip install
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Pin numpy<2 and pandas<2.0 together to avoid ABI mismatch
# (pandas 1.5.x C extensions expect numpy 1.x dtype layout)
RUN pip install "numpy<2" "pandas<2.0"

# Install pypgx with numpy<2 constraint to prevent resolver upgrading numpy
RUN git clone https://github.com/sbslee/pypgx /opt/pypgx && \
    cd /opt/pypgx && \
    git checkout ${PYPGX_BRANCH} && \
    pip install "numpy<2" . && \
    rm -rf /opt/pypgx

##################################################################
FROM python:3.11-slim

ARG PYPGX_BUNDLE_BRANCH=0.26.0

RUN apt-get update && \
    apt-get install --no-install-recommends -y default-jre-headless git wget unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone the pypgx-bundle repository at the corresponding version
RUN git clone --branch ${PYPGX_BUNDLE_BRANCH} --depth 1 https://github.com/sbslee/pypgx-bundle /opt/pypgx-bundle

# Install GATK
ENV GATK_VERSION=4.6.2.0
RUN wget -q https://github.com/broadinstitute/gatk/releases/download/${GATK_VERSION}/gatk-${GATK_VERSION}.zip -O /tmp/gatk.zip && \
    unzip -q /tmp/gatk.zip -d /opt && \
    mv /opt/gatk-${GATK_VERSION} /opt/gatk && \
    ln -s /opt/gatk/gatk /usr/local/bin/gatk && \
    rm /tmp/gatk.zip

# Patch GATK fat JAR: upgrade bundled netty and zookeeper to fix CVEs
COPY docker/patch-gatk-jars.sh /tmp/patch-gatk-jars.sh
RUN chmod +x /tmp/patch-gatk-jars.sh && /tmp/patch-gatk-jars.sh && rm /tmp/patch-gatk-jars.sh

# Remove unnecessary packages
RUN apt-get purge -y git wget unzip && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy pre-built virtual environment from build stage
COPY --from=build /opt/venv /opt/venv
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
