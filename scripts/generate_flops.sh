#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_DIR="$REPO_ROOT/datasets"

mkdir -p "$DATASET_DIR"

echo "Building trainer (release)..."
cargo build -p poker-solver-trainer --release --quiet

echo "Generating canonical flops (JSON)..."
cargo run -p poker-solver-trainer --release --quiet -- flops --format json --output "$DATASET_DIR/flops.json"

echo "Generating canonical flops (CSV)..."
cargo run -p poker-solver-trainer --release --quiet -- flops --format csv --output "$DATASET_DIR/flops.csv"

echo "Done. Files written to $DATASET_DIR/"
ls -lh "$DATASET_DIR"/flops.*
