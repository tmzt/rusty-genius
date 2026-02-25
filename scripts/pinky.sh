#!/bin/bash
#
# Runs the brainteaser test suite against the "Pinky" stub engine.
# This does NOT require the `real-engine` feature.
#

set -e
echo "🧠 Running tests with Pinky (Stub Engine)..."
cargo test -p rusty-genius-teaser -- --nocapture
echo "✅ Pinky tests passed!"
