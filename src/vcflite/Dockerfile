FROM rust:1.80.1-bookworm AS builder

WORKDIR /usr/src/myapp

COPY ./src/vcflite/ .

RUN cargo install --path .

#################################
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        libsqlite3-0 \
        tabix `# for vcf indexing` \
        file `# for monitoring` \
        sysstat `# for IO check in monitoring log` \
        procps `# for memory check in monitoring log` \
    && \
    rm -rf /var/lib/apt/lists/*

# Create a script to run tests from dockerfrom the CI
RUN echo '#!/bin/sh\necho "No tests to run :)"' > /usr/local/bin/run_tests && \
chmod +x /usr/local/bin/run_tests

COPY --from=builder /usr/local/cargo/bin/vcflite /usr/local/bin/vcflite

COPY ./src/vcflite/LICENSE.txt .

CMD ["vcflite"]
