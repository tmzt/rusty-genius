#!/bin/bash
set -e

# Fix for rustc EPERM in some environments: Ensure TMPDIR is local and writable
export TMPDIR="$(pwd)/target/tmp"
mkdir -p "$TMPDIR"

# Stub the cache to skip download
export GENIUS_HOME="$TMPDIR/genius_home"
export GENIUS_CACHE="$GENIUS_HOME/cache"
mkdir -p "$GENIUS_CACHE"

# Pre-create the expected model file so facecrab thinks it's cached
touch "$GENIUS_CACHE/qwen2.5-3b-instruct-q4_k_m.gguf"

echo "Running tests with Stubbed (Pinky) Backend..."
# Disable incremental to avoid EPERM on dep-graph
export CARGO_INCREMENTAL=0
# Run cargo test for brain-teaser with default features (stub)
# Ensure we use --test-threads=1 if needed, but default is fine for now
cargo test --locked -p rusty-genius-teaser -- --nocapture
