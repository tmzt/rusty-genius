#!/bin/bash
set -e

# Help / Usage
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: ./scripts/bump-version.sh"
    echo ""
    echo "Description:"
    echo "  Updates the version in Cargo.toml and CHANGELOG.md files based on conventional commits."
    echo "  This does NOT publish to crates.io or git push."
    echo ""
    exit 0
fi

# Check for release-plz
if ! command -v release-plz &> /dev/null; then
    echo "Error: release-plz is not installed."
    echo "Please run: cargo install release-plz"
    exit 1
fi

echo "📦 Calculating next versions and updating manifests..."
# release-plz update modifies files locally
release-plz update

./update-site-crates.sh

echo ""
echo "✅ Versions bumped! Review the changes with 'git diff'."
