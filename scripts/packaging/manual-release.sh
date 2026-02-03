#!/bin/bash
set -e

# Help / Usage
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: ./scripts/manual-release.sh"
    echo ""
    echo "Description:"
    echo "  Manually triggers the release-plz release command to publish crates and git tags."
    echo ""
    echo "Environment Variables (Optional if logged in locally):"
    echo "  GITHUB_TOKEN          - Token with repo permissions"
    echo "  CARGO_REGISTRY_TOKEN  - Token for crates.io publishing"
    exit 0
fi

# Check for release-plz
if ! command -v release-plz &> /dev/null; then
    echo "Error: release-plz is not installed."
    echo "Please run: cargo install release-plz"
    exit 1
fi

# Note: release-plz relies on local `cargo login` and git credentials if tokens are not provided.
# Ensure you are authenticated with crates.io and GitHub before running.

echo "🚀 Starting manual release..."
echo "release-plz version: $(release-plz --version)"

# Run release
# Matches the behavior of the GitHub Action but runs locally
release-plz release

./update-site-crates.sh
