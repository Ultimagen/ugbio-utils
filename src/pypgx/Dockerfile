FROM ubuntu:24.04

# Install dependencies and create venv
RUN apt-get update && \
    apt-get install --no-install-recommends -y python3.12 python3.12-venv python3.12-dev python3-pip openjdk-17-jdk git build-essential && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# Create venv and install pip properly
RUN python3 -m venv --without-pip /opt/venv && \
    python3 -m pip install --target=/opt/venv/lib/python3.12/site-packages pip setuptools wheel && \
    echo '#!/bin/bash\n/opt/venv/bin/python -m pip "$@"' > /opt/venv/bin/pip && \
    chmod +x /opt/venv/bin/pip && \
    /opt/venv/bin/pip install pypgx==0.25.0

# Clone the pypgx-bundle repository at version 0.25.0, then remove unnecessary packages
RUN git clone --branch 0.25.0 --depth 1 https://github.com/sbslee/pypgx-bundle /opt/pypgx-bundle && \
    apt-get purge -y git build-essential python3.12-dev python3-pip && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables for venv and pypgx bundle
ENV PATH="/opt/venv/bin:$PATH"
ENV PYPGX_BUNDLE=/opt/pypgx-bundle

# Don't use root user
RUN groupadd ugbio && useradd -m ugbio -g ugbio
USER ugbio
WORKDIR /home/ugbio
