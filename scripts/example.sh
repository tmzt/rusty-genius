#!/bin/bash
set -e

# Sandbox configuration
export TMPDIR="$(pwd)/target/tmp"
export GENIUS_HOME="$(pwd)/target/tmp/config"
export GENIUS_CACHE="$(pwd)/target/tmp/cache"
export CARGO_INCREMENTAL=0

mkdir -p "$TMPDIR"
mkdir -p "$GENIUS_HOME"
mkdir -p "$GENIUS_CACHE"

echo "Running basic_chat example in sandbox..."
echo "GENIUS_HOME: $GENIUS_HOME"
echo "GENIUS_CACHE: $GENIUS_CACHE"

# Run the example with real-engine and metal features
cargo run --example basic_chat --features "real-engine metal"
