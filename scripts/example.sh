#!/usr/bin/env bash
set -euo pipefail

EXAMPLE="${1:-thinker_tui}"
shift 2>/dev/null || true

# Sandbox configuration
export TMPDIR="$(pwd)/target/tmp"
export GENIUS_HOME="$(pwd)/target/tmp/config"
export GENIUS_CACHE="$(pwd)/target/tmp/cache"

mkdir -p "$TMPDIR"
mkdir -p "$GENIUS_HOME"
mkdir -p "$GENIUS_CACHE"

# Pick the GPU feature based on OS
case "$(uname -s)" in
    Darwin) FEATURE=metal  ;;
    Linux)  FEATURE=cuda   ;;
    *)      FEATURE=real-engine ;;
esac

# Allow override: ./scripts/example.sh thinker_tui --feature vulkan
for arg in "$@"; do
    case "$arg" in
        --feature=*) FEATURE="${arg#--feature=}" ;;
        --feature)   shift_next=1 ;;
        *)
            if [[ "${shift_next:-}" == 1 ]]; then
                FEATURE="$arg"
                unset shift_next
            fi
            ;;
    esac
done

echo "Running example '${EXAMPLE}' with --features ${FEATURE}"
echo "GENIUS_HOME: $GENIUS_HOME"
echo "GENIUS_CACHE: $GENIUS_CACHE"

exec cargo run -p rusty-genius --features "${FEATURE}" --example "${EXAMPLE}"
