#!/usr/bin/env node
// setup-tts.mjs - pre-populate .cache/tts/tiny-voca (plain-JS engine + 8 MB GGUF)
// and remove any legacy phonon-tts checkpoints from .cache/tts.
import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from '../src/utils.js';
import {
  DEFAULT_TTS_MODEL,
  ensureTinyVocaEngine,
  ensureTinyVocaModel,
  pruneLegacyTtsCache,
} from '../src/tts.js';

async function main() {
  console.log('tiny-voca TTS setup\n-------------------');
  pruneLegacyTtsCache();
  const sayPath = await ensureTinyVocaEngine();
  const ggufPath = await ensureTinyVocaModel(process.env.TTS_MODEL || DEFAULT_TTS_MODEL);
  const mb = (fs.statSync(ggufPath).size / 1e6).toFixed(1);
  console.log(`  \u2713 engine: ${path.relative(ROOT, sayPath)}`);
  console.log(`  \u2713 model : ${path.relative(ROOT, ggufPath)} (${mb} MB, default voice M2)`);
}

main().catch((e) => {
  console.error(`error: ${e?.message ?? e}`);
  process.exit(1);
});