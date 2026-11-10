#!/usr/bin/env bash
# Upload dist/space (made by scripts/release.sh) to the Hugging Face Space.
#
#   scripts/upload.sh
#
# Needs a write token: `uvx --from huggingface_hub hf auth login`, or HF_TOKEN in the environment.
set -euo pipefail
cd "$(dirname "$0")/.."

SPACE="${SPACE:-unvop/phonon-2-web}"
VERSION="$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"
HF="uvx --from huggingface_hub hf"
[ -d dist/space ] || { echo "run scripts/release.sh first"; exit 1; }

$HF auth whoami >/dev/null || { echo "not logged in: run  $HF auth login"; exit 1; }
$HF repos create "$SPACE" --repo-type space --space-sdk static --exist-ok
$HF upload "$SPACE" dist/space . --repo-type space --commit-message "Phonon-2 Web v$VERSION"

echo
echo "space: https://huggingface.co/spaces/$SPACE"
