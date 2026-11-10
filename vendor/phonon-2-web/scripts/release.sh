#!/usr/bin/env bash
# Build, verify and assemble the static Hugging Face Space in dist/space; upload it with scripts/upload.sh.
#
#   scripts/release.sh
#
# Verification runs when the model is in the local Hugging Face cache and testdata/ref exists (VERIFY=0 skips it):
# the native golden check, then the in-browser check on every backend if a Chromium-based browser is installed.
set -euo pipefail
cd "$(dirname "$0")/.."

SPACE="${SPACE:-unvop/phonon-2-web}"
REPO="${REPO:-unvop/phonon-2-web}"
VERSION="$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"
SUBDOMAIN="$(echo "$SPACE" | tr '/' '-' | tr '[:upper:]_.' '[:lower:]--')"
echo "release v$VERSION: space $SPACE, source github.com/$REPO"

./build-wasm.sh

MODEL="$(ls ~/.cache/huggingface/hub/models--FermionResearch--Phonon-2/snapshots/*/phonon-2.bps.tar.zst 2>/dev/null | head -1 || true)"
if [ "${VERIFY:-1}" = "1" ] && [ -n "$MODEL" ] && [ -f testdata/ref/reference.json ]; then
  cargo build --release -p phonon-cli --features gpu
  ./target/release/phonon-rs check "$MODEL" testdata | tail -1 | grep -q "token-identical" || { echo "native check failed"; exit 1; }
  ./target/release/phonon-rs check "$MODEL" testdata --gpu | tail -1
  if command -v node >/dev/null; then
    PHONON_MODEL="$MODEL" node serve.mjs 8787 >/dev/null 2>&1 &
    SERVER=$!
    trap 'kill $SERVER 2>/dev/null || true' EXIT
    sleep 1
    LOG="$(mktemp)"
    node tools/browser-check.mjs cpu,webgpu,webgl | tee "$LOG" | tail -5 || true
    grep -q "status: .*identical" "$LOG" || echo "warning: browser check did not complete"
  fi
else
  echo "skipping verification (no cached model or reference tensors)"
fi

rm -rf dist/space && mkdir -p dist/space
cp -R web/index.html web/app.js web/style.css web/phonon-worker.js web/autoselect.js web/mic-worklet.js \
      web/coi.js web/coi-sw.js web/samples web/pkg web/pkg-relaxed web/pkg-st LICENSE NOTICE dist/space/
rm -f dist/space/pkg*/.gitignore
sed -e "s|{{SPACE}}|$SPACE|g" -e "s|{{SUBDOMAIN}}|$SUBDOMAIN|g" -e "s|{{REPO}}|$REPO|g" hf/space-README.md > dist/space/README.md
sed -i.bak "s|(Apache-2.0)\.|(Apache-2.0), v$VERSION.|" dist/space/index.html && rm dist/space/index.html.bak
find dist -name .DS_Store -delete

echo
du -sh dist/space
echo "next: scripts/upload.sh"
