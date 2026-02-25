#!/bin/bash
#
# Runs the brainteaser test suite against the "Brain" real engine.
# This REQUIRES the `real-engine` feature and a compatible model.
#

set -e
echo "🤖 Running tests with Brain (Real Engine)..."
cargo test -p rusty-genius-teaser --features real-engine -- --nocapture
echo "✅ Brain tests passed!"
