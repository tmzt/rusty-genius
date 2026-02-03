#!/bin/bash
set -e

# Function to check if cmake exists
check_cmake() {
    if command -v cmake &> /dev/null; then
        echo "cmake found at $(command -v cmake)"
        return 0
    fi
    
    # Check common locations
    COMMON_LOCATIONS=(
        "/opt/homebrew/bin/cmake"
        "/usr/local/bin/cmake"
        "/usr/bin/cmake"
    )

    for loc in "${COMMON_LOCATIONS[@]}"; do
        if [ -x "$loc" ]; then
            echo "cmake found at $loc"
            export PATH="$(dirname "$loc"):$PATH"
            return 0
        fi
    done

    echo "Error: cmake not found. Please install cmake (e.g., brew install cmake)"
    return 1
}

# Verify cmake availability
check_cmake || exit 1

# Fix for rustc EPERM in some environments: Ensure TMPDIR is local and writable
export TMPDIR="$(pwd)/target/tmp"
mkdir -p "$TMPDIR"

# NO SANDBOX: Running raw
echo "Running tests cleanly (System Config / Local TMP)..."

echo "Running tests with Metal acceleration..."
# Run cargo test with specific features enabled
export CARGO_INCREMENTAL=0
cargo test --locked -p rusty-genius-brain-teaser --features "real-engine rusty-genius-brain-cortex/metal" -- --nocapture
