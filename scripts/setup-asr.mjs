#!/usr/bin/env node
// setup-asr.mjs - one-time setup of the phonon-2-web ASR engine.
//
// The factory ships with a prebuilt single-threaded WASM engine in
// vendor/phonon-2-web/web/pkg-st (built from github.com/unvop/phonon-2-web with
// `RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target
// wasm32-unknown-unknown -p phonon-wasm --no-default-features` + wasm-bindgen).
//
// This script verifies that build and, if it is missing, tries to rebuild it:
//   1. checks for a local Rust toolchain (stable is enough for the CPU build),
//   2. clones the upstream repo if vendor/phonon-2-web is incomplete,
//   3. builds web/pkg-st and runs a smoke transcription.
// The 164 MB Parakeet model itself is downloaded on first use and cached in .cache/asr.
import { execSync, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const VENDOR = path.join(ROOT, 'vendor', 'phonon-2-web');
const PKG_ST = path.join(VENDOR, 'web', 'pkg-st');

function ok(msg) { console.log(`  \u2713 ${msg}`); }
function warn(msg) { console.log(`  ! ${msg}`); }

function hasCommand(cmd) {
  const probe = spawnSync(cmd, ['--version'], { encoding: 'utf8' });
  return probe.status === 0;
}

async function main() {
  console.log('phonon-2-web ASR setup\n----------------------');
  if (fs.existsSync(path.join(PKG_ST, 'phonon_wasm_bg.wasm'))) {
    ok(`prebuilt engine found: ${path.relative(ROOT, PKG_ST)}`);
    console.log('\nAll set. The 164 MB model downloads once on first use (.cache/asr).');
    return;
  }

  console.log('  prebuilt engine missing - building from source ...');
  if (!fs.existsSync(path.join(VENDOR, 'build-wasm.sh'))) {
    console.log('  cloning github.com/unvop/phonon-2-web ...');
    execSync(`git clone --depth 1 https://github.com/unvop/phonon-2-web "${VENDOR}"`, { stdio: 'inherit' });
  }
  if (!hasCommand('cargo')) {
    warn('Rust is not installed. Install it from https://rustup.rs then re-run: npm run setup:asr');
    console.log('\n  (the build needs only stable Rust for the CPU engine:');
    console.log('   rustup target add wasm32-unknown-unknown');
    console.log('   cargo install wasm-bindgen-cli --version 0.2.129');
    console.log(`   cd vendor/phonon-2-web && RUSTFLAGS="-C target-feature=+simd128" \\`);
    console.log('     cargo build --release --target wasm32-unknown-unknown -p phonon-wasm --no-default-features');
    console.log('   wasm-bindgen --target web --out-dir web/pkg-st target/wasm32-unknown-unknown/release/phonon_wasm.wasm');
    process.exit(1);
  }
  if (!hasCommand('wasm-bindgen')) {
    console.log('  installing wasm-bindgen-cli 0.2.129 (one-time, a few minutes) ...');
    execSync('cargo install wasm-bindgen-cli --version 0.2.129', { stdio: 'inherit' });
  }
  console.log('  adding the wasm32 target ...');
  execSync('rustup target add wasm32-unknown-unknown', { stdio: 'inherit' });
  console.log('  building the CPU engine (1-3 minutes) ...');
  execSync('cargo build --release --target wasm32-unknown-unknown -p phonon-wasm --no-default-features', {
    cwd: VENDOR, stdio: 'inherit',
    env: { ...process.env, RUSTFLAGS: '-C target-feature=+simd128' },
  });
  console.log('  generating JS glue ...');
  execSync('wasm-bindgen --target web --out-dir web/pkg-st target/wasm32-unknown-unknown/release/phonon_wasm.wasm', {
    cwd: VENDOR, stdio: 'inherit',
  });
  if (fs.existsSync(path.join(PKG_ST, 'phonon_wasm_bg.wasm'))) {
    ok('engine built successfully');
    console.log('\nAll set. The 164 MB model downloads once on first use (.cache/asr).');
  } else {
    warn('build finished but pkg-st is missing - check the output above');
    process.exit(1);
  }
}

main();
