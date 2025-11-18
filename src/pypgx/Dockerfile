FROM python:3.12-slim

ARG PYPGX_BRANCH=0.26.0-dev
ARG PYPGX_BUNDLE_BRANCH=0.25.0

# Install dependencies and create venv
RUN apt-get update && \
    apt-get install --no-install-recommends -y default-jre-headless git build-essential wget unzip

# Install pypgx
RUN git clone https://github.com/sbslee/pypgx /opt/pypgx && \
    cd /opt/pypgx && \
    git switch ${PYPGX_BRANCH} && \
    pip install . && \
    rm -rf /opt/pypgx

# Clone the pypgx-bundle repository at the corresponding version
RUN git clone --branch ${PYPGX_BUNDLE_BRANCH} --depth 1 https://github.com/sbslee/pypgx-bundle /opt/pypgx-bundle

# Install GATK
ENV GATK_VERSION=4.5.0.0
RUN wget -q https://github.com/broadinstitute/gatk/releases/download/${GATK_VERSION}/gatk-${GATK_VERSION}.zip -O /tmp/gatk.zip && \
    unzip -q /tmp/gatk.zip -d /opt && \
    mv /opt/gatk-${GATK_VERSION} /opt/gatk && \
    ln -s /opt/gatk/gatk /usr/local/bin/gatk && \
    rm /tmp/gatk.zip

# Remove unnecessary packages and recreate python3 symlink
RUN apt-get purge -y git wget unzip && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables for pypgx bundle
ENV PYPGX_BUNDLE=/opt/pypgx-bundle

# Create a script to run tests from docker CI
RUN echo '#!/bin/sh\necho "No tests to run"' > /usr/local/bin/run_tests \
&& chmod +x /usr/local/bin/run_tests

# Don't use root user
RUN groupadd ugbio && useradd -m ugbio -g ugbio
USER ugbio
WORKDIR /home/ugbio
