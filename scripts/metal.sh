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

# Directive #2: Sandbox Isolation
# Create a local temp directory to avoid sandbox restrictions on system /tmp
export LOCAL_TEMP_DIR="$(pwd)/target/tmp"
mkdir -p "$LOCAL_TEMP_DIR"
# Force temp dir usage
# Force temp dir usage
export TMPDIR="$(pwd)/target/tmp"
export TEMP="$(pwd)/target/tmp"
export TMP="$(pwd)/target/tmp"
# Use unique config dir to avoid stale locks
# export GENIUS_HOME="$(pwd)/target/tmp/config_$(date +%s)"

# Prevent duplicate model downloads
export GENIUS_HOME="$(pwd)/target/tmp/config"
export GENIUS_CACHE="$(pwd)/target/tmp/cache"

mkdir -p "$TMPDIR"
mkdir -p "$GENIUS_HOME"
mkdir -p "$GENIUS_CACHE"

echo "Using local temp dir: $TMPDIR"
echo "Using GENIUS_HOME: $GENIUS_HOME"
echo "Using GENIUS_CACHE: $GENIUS_CACHE"

# Verify write access
touch "$GENIUS_HOME/test_write" || { echo "Failed to write to GENIUS_HOME"; exit 1; }
touch "$GENIUS_CACHE/test_write" || { echo "Failed to write to GENIUS_CACHE"; exit 1; }
echo "Write access confirmed."

echo "Running tests with Metal acceleration..."
# Run cargo test with specific features enabled
export CARGO_INCREMENTAL=0
cargo test --locked -p rusty-genius-teaser --features "real-engine rusty-genius-cortex/metal"  #-- --nocapture
