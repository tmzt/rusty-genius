#!/bin/bash
# Helper script to run Cargo commands WITHOUT --locked.
# Use this only when you need to update Cargo.lock (e.g. adding dependencies).
#
# Usage: ./scripts/fix.sh [command] [args...]
# Example: ./scripts/fix.sh build

set -e

CMD="${1:-check}"
shift

# Directive #2: Sandbox Isolation
export LOCAL_TEMP_DIR="$(pwd)/target/tmp"
mkdir -p "$LOCAL_TEMP_DIR"
export TMPDIR="$LOCAL_TEMP_DIR"
export TEMP="$LOCAL_TEMP_DIR"
export TMP="$LOCAL_TEMP_DIR"

echo "Running: cargo $CMD $@ (unlocked)"
cargo "$CMD" "$@"
