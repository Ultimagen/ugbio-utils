FROM ubuntu:22.04

# Install required packages
RUN apt-get update && apt-get install --no-install-recommends -y \
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
