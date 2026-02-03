#!/bin/bash
set -e

echo "🔍 Extracting new version..."
# Extract version from [workspace.package] in Cargo.toml
# Assumes 'version = "X.Y.Z"' is the first match
NEW_VERSION=$(grep '^version = ' Cargo.toml | head -n 1 | cut -d '"' -f 2)
echo "   New Version: $NEW_VERSION"

echo "📝 Updating documentation..."

# Update README.md: Badges and Dependency
# Mac sed requires -i ''
sed -i '' "s/crates.io-v[0-9.]*-orange/crates.io-v$NEW_VERSION-orange/g" README.md
sed -i '' "s/version = \"[0-9.]*\"/version = \"$NEW_VERSION\"/g" README.md

# Update site/index.html: Dependency
sed -i '' "s/version = \"[0-9.]*\"/version = \"$NEW_VERSION\"/g" site/index.html

echo "   README.md and site/index.html updated."

git add -u README.md ./site/index.html

git commit -m "(chore) site: update crate versions"

git push
