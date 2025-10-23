FROM ubuntu:24.04

# Install dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y python3.12 python3.12-venv python3.12-dev python3.12-ensurepip python3-pip openjdk-17-jdk git build-essential && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/python -m ensurepip && \
    apt-get purge -y python3-pip && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip in the venv and install pypgx==0.25.0
RUN /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install pypgx==0.25.0

# Clone the pypgx-bundle repository at version 0.25.0, then remove git, build-essential, and python3.12-dev
RUN git clone --branch 0.25.0 --depth 1 https://github.com/sbslee/pypgx-bundle /opt/pypgx-bundle && \
    apt-get purge -y git build-essential python3.12-dev && \
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
