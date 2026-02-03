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

echo "Running tests with Metal acceleration..."
# Run cargo test with specific features enabled
cargo test --locked -p rusty-genius-brain-teaser --features "rusty-genius-brain-cortex/real-engine rusty-genius-brain-cortex/metal"
