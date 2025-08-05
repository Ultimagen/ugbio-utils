FROM python:3.11-bullseye

# Install required packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    wget \
    libgomp1 \
    unzip \
    automake \
    make \
    libcurl4-gnutls-dev \
    git \
    `# the following packages required for pyBigWig python package used by ugbio_core` \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libc6-dev \
    gcc \
    procps `# for monitoring script memory usage` \
    sysstat `# for monitoring script IO metrics` \
    sudo \
    bzip2 `# instead of full htslib installation`\
    tabix `#instead of full htslib installation`\
    openjdk-17-jre-headless `# for gatk` \
    git \
    wget \
    unzip \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && apt-get autoclean

# Download and install gatk
ARG GATK_VERSION=4.6.1.0
RUN wget https://github.com/broadinstitute/gatk/releases/download/${GATK_VERSION}/gatk-${GATK_VERSION}.zip && \
    unzip gatk-${GATK_VERSION}.zip && \
    mv gatk-${GATK_VERSION} /opt/gatk && \
    ln -s /opt/gatk/gatk /usr/local/bin/gatk && \
    rm gatk-${GATK_VERSION}.zip

# install samtools
ARG SAMTOOLS_VERSION=1.20
WORKDIR /usr/local/bin
RUN wget https://github.com/samtools/samtools/releases/download/${SAMTOOLS_VERSION}/samtools-${SAMTOOLS_VERSION}.tar.bz2
RUN tar -xjf /usr/local/bin/samtools-${SAMTOOLS_VERSION}.tar.bz2 -C /usr/local/bin/
RUN cd /usr/local/bin/samtools-${SAMTOOLS_VERSION}/ && ./configure
RUN cd /usr/local/bin/samtools-${SAMTOOLS_VERSION}/ && make
RUN cd /usr/local/bin/samtools-${SAMTOOLS_VERSION}/ && make install

RUN wget https://github.com/samtools/samtools/releases/download/${SAMTOOLS_VERSION}/samtools-${SAMTOOLS_VERSION}.tar.bz2 && \
    tar -vxjf samtools-${SAMTOOLS_VERSION}.tar.bz2 && \
    rm samtools-${SAMTOOLS_VERSION}.tar.bz2 && \
    cd samtools-${SAMTOOLS_VERSION} && \
    make && \
    make install && \
    cd .. && \
    rm -rf samtools-${SAMTOOLS_VERSION}


# install bcftools
ARG BCFTOOLS_VERSION=1.20
RUN wget https://github.com/samtools/bcftools/releases/download/${BCFTOOLS_VERSION}/bcftools-${BCFTOOLS_VERSION}.tar.bz2 && \
    tar -vxjf bcftools-${BCFTOOLS_VERSION}.tar.bz2 && \
    rm bcftools-${BCFTOOLS_VERSION}.tar.bz2 && \
    cd bcftools-${BCFTOOLS_VERSION} && \
    make && \
    make install && \
    cd .. && \
    rm -rf bcftools-${BCFTOOLS_VERSION}

# Create a non-root user
RUN groupadd -r ugbio && useradd -r -g ugbio -m -d /home/ugbio ugbio

# Set ownership for writable directories
RUN chown -R ugbio:ugbio /opt /usr/local/bin

# Set the working directory
WORKDIR /home/ugbio

# Switch to non-root user
USER ugbio

# Set default command
CMD ["/bin/bash"]
