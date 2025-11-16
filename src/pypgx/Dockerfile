FROM ubuntu:24.04

# Install dependencies and create venv
RUN apt-get update && \
    apt-get install --no-install-recommends -y python3.12 python3.12-venv python3.12-dev python3-pip openjdk-17-jdk git build-essential wget unzip && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# Create venv and install pip
RUN python3 -m venv --without-pip /opt/venv && \
    python3 -m pip install --target=/opt/venv/lib/python3.12/site-packages pip setuptools wheel && \
    echo '#!/bin/bash\n/opt/venv/bin/python -m pip "$@"' > /opt/venv/bin/pip && \
    chmod +x /opt/venv/bin/pip

# Install pypgx
RUN git clone https://github.com/sbslee/pypgx /opt/pypgx && \
    cd /opt/pypgx && \
    git switch 0.26.0-dev && \
    /opt/venv/bin/pip install . && \
    rm -rf /opt/pypgx

# Clone the pypgx-bundle repository at the corresponding version
RUN git clone --branch 0.25.0 --depth 1 https://github.com/sbslee/pypgx-bundle /opt/pypgx-bundle

# Install GATK
ENV GATK_VERSION=4.5.0.0
RUN wget -q https://github.com/broadinstitute/gatk/releases/download/${GATK_VERSION}/gatk-${GATK_VERSION}.zip -O /tmp/gatk.zip && \
    unzip -q /tmp/gatk.zip -d /opt && \
    mv /opt/gatk-${GATK_VERSION} /opt/gatk && \
    ln -s /opt/gatk/gatk /usr/local/bin/gatk && \
    rm /tmp/gatk.zip

# Remove unnecessary packages and recreate python3 symlink
RUN apt-get purge -y git python3.12-dev python3-pip wget unzip && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# Set environment variables for venv and pypgx bundle
ENV PATH="/opt/venv/bin:$PATH"
ENV PYPGX_BUNDLE=/opt/pypgx-bundle

# Create a script to run tests from docker CI
RUN echo '#!/bin/sh\necho "No tests to run"' > /usr/local/bin/run_tests \
&& chmod +x /usr/local/bin/run_tests

# Don't use root user
RUN groupadd ugbio && useradd -m ugbio -g ugbio
USER ugbio
WORKDIR /home/ugbio
