#!/usr/bin/env bash
set -euo pipefail

EXAMPLE="${1:-thinker_tui}"
shift 2>/dev/null || true

# Pick the GPU feature based on OS
case "$(uname -s)" in
    Darwin) FEATURE=metal  ;;
    Linux)  FEATURE=cuda   ;;
    *)      FEATURE=real-engine ;;
esac

# Allow override: ./example.sh thinker_tui --feature vulkan
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
exec cargo run -p rusty-genius --features "${FEATURE}" --example "${EXAMPLE}"
