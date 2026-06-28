FROM ubuntu:24.04

# Install dependencies
# Note: libdeflate-dev, liblzma-dev, libbz2-dev are required by htslib for compression support
# autoconf and automake are required to build bundled htslib
RUN apt update && apt install -y \
    git \
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
    && apt clean

# Clone T1K (using v1.0.9, closest available release to validation version)
RUN git clone https://github.com/mourisl/T1K.git /T1K \
    && cd /T1K \
    && git checkout v1.0.9

# Build T1K with CRAM support
# htslib=1 tells T1K to build bundled htslib-1.15.1 (includes CRAM support)
# The Makefile will automatically build htslib if libhts.a doesn't exist
# We add compression library flags to LINKFLAGS for proper linking
RUN cd /T1K \
    && make htslib=1 LINKFLAGS="-lpthread -lz -ldeflate -llzma -lbz2 -lcurl -lssl -lcrypto" -j $(nproc)

# Symlink executables to PATH
RUN ln -s /T1K/run-t1k /usr/local/bin/run-t1k \
    && ln -s /T1K/t1k-build.pl /usr/local/bin/t1k-build.pl \
    && ln -s /T1K/bam-extractor /usr/local/bin/bam-extractor

# Clean up build dependencies (keep runtime libraries)
RUN apt --yes remove git make build-essential autoconf automake libtool && apt --yes autoremove && apt clean

# Non-root user (following terra_pipeline conventions)
RUN groupadd ugbio && useradd -m ugbio -g ugbio
USER ugbio
WORKDIR /home/ugbio
