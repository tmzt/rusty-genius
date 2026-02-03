#!/bin/bash
# Helper script to run Cargo commands WITHOUT --locked.
# Use this only when you need to update Cargo.lock (e.g. adding dependencies).
#
# Usage: ./scripts/fix.sh [command] [args...]
# Example: ./scripts/fix.sh build

set -e

CMD="${1:-check}"
shift

echo "Running: cargo $CMD $@ (unlocked)"
cargo "$CMD" "$@"
