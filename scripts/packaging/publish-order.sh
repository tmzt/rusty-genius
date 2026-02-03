#!/bin/bash

# Script to publish rusty-genius crates in the correct dependency order.
# This ensures that higher-level crates can find the specific versions
# of their internal dependencies on crates.io.

set -e

echo "🚀 Starting publication sequence for rusty-genius v0.1.2..."

# # 1. CORE (Shared Vocabulary)
# echo "📦 Publishing rusty-genius-core..."
# cargo publish -p rusty-genius-core

# 2. ASSETS (Facecrab)
echo "📦 Publishing facecrab..."
cargo publish -p facecrab

# 3. ENGINE (Cortex)
echo "📦 Publishing rusty-genius-cortex..."
cargo publish -p rusty-genius-cortex

# 4. ORCHESTRATOR (Brainstem)
echo "📦 Publishing rusty-genius-stem..."
cargo publish -p rusty-genius-stem

# 5. FACADE (Genius)
echo "📦 Publishing rusty-genius..."
cargo publish -p rusty-genius

echo "✅ All crates published successfully!"
