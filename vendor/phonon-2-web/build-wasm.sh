#!/usr/bin/env bash
# Build the browser module into web/pkg.
#
#   ./build-wasm.sh            threads + SIMD + WebGPU + WebGL2 (needs a cross-origin-isolated page: see serve.mjs)
#   FEATURES=threads ./build-wasm.sh    CPU only
#
# Threads need nightly Rust (std rebuilt with atomics): rustup toolchain install nightly --component rust-src
set -euo pipefail
cd "$(dirname "$0")"

FEATURES="${FEATURES:-threads,gpu,webgl}"
PROFILE="${PROFILE:-release}"
TARGET=wasm32-unknown-unknown
BINDGEN="${WASM_BINDGEN:-$(command -v wasm-bindgen || echo "$HOME/.cargo/bin/wasm-bindgen")}"

# 4 GiB ceiling: the encoder weights take ~330 MB, the decompressed download streams through, activations are small.
# Shared memory and the TLS exports wasm-bindgen's thread support expects are requested explicitly: recent nightlies
# no longer imply them from +atomics.
LINK="-C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=4294967296"
LINK="$LINK -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base"
export RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 $LINK"

cargo +nightly build -Z build-std=panic_abort,std --target "$TARGET" --profile "$PROFILE" \
  -p phonon-wasm --no-default-features --features "$FEATURES"

bindgen() { # <wasm> <out dir> [extra wasm-opt flags]
  rm -rf "$2"
  "$BINDGEN" --target web --out-dir "$2" "$1"
  if command -v wasm-opt >/dev/null; then
    wasm-opt -O3 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext ${3:-} \
      "$2/phonon_wasm_bg.wasm" -o "$2/phonon_wasm_bg.wasm"
  fi
}
bindgen "target/$TARGET/$PROFILE/phonon_wasm.wasm" web/pkg "--enable-threads --enable-mutable-globals"

# Threads + relaxed SIMD (hardware FMA): about 2x on the CPU encoder where the browser supports it (Chrome 114+).
if [ "${RELAXED:-1}" = "1" ]; then
  RUSTFLAGS="${RUSTFLAGS/+simd128/+simd128,+relaxed-simd}" cargo +nightly build -Z build-std=panic_abort,std --target "$TARGET" \
    --profile "$PROFILE" --target-dir target/relaxed -p phonon-wasm --no-default-features --features "$FEATURES"
  bindgen "target/relaxed/$TARGET/$PROFILE/phonon_wasm.wasm" web/pkg-relaxed "--enable-threads --enable-mutable-globals --enable-relaxed-simd"
fi
# Single-threaded fallback for pages that are not cross-origin isolated (no SharedArrayBuffer): stable Rust, no atomics.
if [ "${SINGLE:-1}" = "1" ]; then
  ST_FEATURES="$(echo "$FEATURES" | tr ',' '\n' | grep -v '^threads$' | paste -sd, -)"
  RUSTFLAGS="-C target-feature=+simd128" cargo build --target "$TARGET" --profile "$PROFILE" --target-dir target/st \
    -p phonon-wasm --no-default-features ${ST_FEATURES:+--features "$ST_FEATURES"}
  bindgen "target/st/$TARGET/$PROFILE/phonon_wasm.wasm" web/pkg-st
fi
ls -la web/pkg*/*.wasm
