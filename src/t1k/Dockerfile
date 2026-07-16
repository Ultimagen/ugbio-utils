FROM ubuntu:24.04

# Install dependencies
# Note: libdeflate-dev, liblzma-dev, libbz2-dev are required by htslib for compression support
# autoconf and automake are required to build bundled htslib
RUN apt update && apt install --no-install-recommends -y \
    git \
    ca-certificates \
    make \
    build-essential \
    autoconf \
    automake \
    libtool \
    libdeflate-dev \
    liblzma-dev \
    libbz2-dev \
    zlib1g-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    perl \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*


# Clone T1K (using v1.0.9, closest available release to validation version)
RUN git clone https://github.com/mourisl/T1K.git /T1K \
    && cd /T1K \
    && git checkout v1.0.9

# Build T1K with CRAM support
# htslib=1 tells T1K to build bundled htslib-1.15.1 (includes CRAM support)
# First we need to download and build htslib with submodules, then build T1K
RUN cd /T1K \
    && git clone --depth 1 --branch 1.15.1 --recurse-submodules https://github.com/samtools/htslib.git htslib-1.15.1 \
    && cd htslib-1.15.1 \
    && autoreconf -i \
    && ./configure \
    && make -j $(nproc) \
    && cd .. \
    && make htslib=1 LINKFLAGS="-lpthread -lz -ldeflate -llzma -lbz2 -lcurl -lssl -lcrypto" -j $(nproc)


# Install htslib shared library
RUN cp /T1K/htslib-1.15.1/libhts.so.3 /usr/local/lib/ \
      && ldconfig

# Symlink executables to PATH
RUN ln -s /T1K/run-t1k /usr/local/bin/run-t1k \
    && ln -s /T1K/t1k-build.pl /usr/local/bin/t1k-build.pl \
    && ln -s /T1K/bam-extractor /usr/local/bin/bam-extractor

# Clean up build dependencies (keep runtime libraries)
RUN apt --yes remove git make build-essential autoconf automake libtool && apt --yes autoremove && apt clean

# Create a script to run tests from docker CI
RUN echo '#!/bin/sh\necho "No tests to run"' > /usr/local/bin/run_tests \
            && chmod +x /usr/local/bin/run_tests

# Non-root user (following terra_pipeline conventions)
RUN groupadd ugbio && useradd -m ugbio -g ugbio
USER ugbio
WORKDIR /home/ugbio
