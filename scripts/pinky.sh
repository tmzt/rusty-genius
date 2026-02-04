#!/bin/bash
set -e

# Fix for permissions: Ensure TMPDIR is local and writable
export TMPDIR="/tmp/rusty-genius-tmp"
mkdir -p "$TMPDIR"
export CARGO_INCREMENTAL=0
# Run cargo test for brain-teaser with default features (stub)
# Ensure we use --test-threads=1 if needed, but default is fine for now
cargo test --locked -p rusty-genius-teaser -- --nocapture
