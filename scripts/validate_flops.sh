#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

REFERENCE="$REPO_ROOT/datasets/1755_flops.csv"
GENERATED="$REPO_ROOT/datasets/flops.csv"

if [ ! -f "$REFERENCE" ]; then
    echo "Error: $REFERENCE not found"
    exit 1
fi

if [ ! -f "$GENERATED" ]; then
    echo "Error: $GENERATED not found. Run ./scripts/generate_flops.sh first."
    exit 1
fi

echo "Building validator..."
cargo build -p poker-solver-core --release --quiet --example validate_flops

echo "Validating..."
cargo run -p poker-solver-core --release --quiet --example validate_flops -- "$REFERENCE" "$GENERATED"
