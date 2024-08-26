FROM rust:1.80.1-bookworm AS builder

WORKDIR /usr/src/myapp

COPY . .
RUN cargo install --path .

#################################
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y \
        libsqlite3-0 \
        tabix `# for vcf indexing` \
        file `# for monitoring` \
    && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/cargo/bin/vcflite /usr/local/bin/vcflite

CMD ["vcflite"]
