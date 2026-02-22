# cloud/Dockerfile — multi-stage build for poker-solver-trainer
#
# Build:  docker build -f cloud/Dockerfile -t poker-solver-trainer .
# Run:    docker run --rm poker-solver-trainer train -c /config/training.yaml

# ── Stage 1: Build ──────────────────────────────────────────────────────────
FROM rust:1.85-bookworm AS builder

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

# Build only the trainer binary in release mode
RUN cargo build -p poker-solver-trainer --release

# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/target/release/poker-solver-trainer /usr/local/bin/

ENTRYPOINT ["poker-solver-trainer"]
