// asr.js - speech recognition via phonon-2-web (github.com/unvop/phonon-2-web), replacing
// faster-whisper tiny. The engine is a Rust/WASM port of NVIDIA Parakeet-TDT 0.6B v3 that
// returns token-accurate word timestamps - everything captions need - running fully on-device.
//
// The WASM build lives in vendor/phonon-2-web/web/pkg-st (create it once: `npm run setup:asr`).
// The 164 MB model is downloaded once from the Hugging Face Hub and cached in .cache/asr.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { ROOT } from './utils.js';

const MODEL_URL = 'https://huggingface.co/FermionResearch/Phonon-2/resolve/main/phonon-2.bps.tar.zst';

export function asrAvailable() {
  for (const dir of ['web/pkg-st', 'web/pkg', 'web/pkg-relaxed']) {
    const p = path.join(ROOT, 'vendor', 'phonon-2-web', dir, 'phonon_wasm.js');
    if (fs.existsSync(p)) return path.join(ROOT, 'vendor', 'phonon-2-web', dir);
  }
  return null;
}

let enginePromise = null;

async function loadEngine() {
  const pkgDir = asrAvailable();
  if (!pkgDir) {
    throw new Error(
      'phonon-2-web ASR is not built yet. Run:  npm run setup:asr\n' +
      '(builds the WASM engine once; needs Rust or a prebuilt copy in vendor/phonon-2-web/web/pkg-st)');
  }
  // pathToFileURL (not 'file://' + path) - correct on Windows where a raw `D:\...` path or
  // `file://D:\...` would be mis-parsed by the ESM loader.
  const glue = await import(pathToFileURL(path.join(pkgDir, 'phonon_wasm.js')).href);
  // The web-target glue would try to fetch its .wasm via URL (file:// is not fetchable in Node),
  // so we compile the module explicitly.
  await glue.initSync({ module: await WebAssembly.compile(await fs.promises.readFile(path.join(pkgDir, 'phonon_wasm_bg.wasm'))) });
  const cacheDir = path.join(ROOT, '.cache', 'asr');
  const modelPath = path.join(cacheDir, 'phonon-2.tar.zst');
  if (!fs.existsSync(modelPath)) {
    console.log('[asr] downloading the Phonon-2 model (164 MB, once) ...');
    fs.mkdirSync(cacheDir, { recursive: true });
    const res = await fetch(MODEL_URL);
    if (!res.ok) throw new Error(`model download failed: HTTP ${res.status}`);
    const tmp = modelPath + '.part';
    await fs.promises.writeFile(tmp, Buffer.from(await res.arrayBuffer()));
    fs.renameSync(tmp, modelPath);
  }
  const bytes = new Uint8Array(fs.readFileSync(modelPath));
  const phonon = glue.Phonon.load(bytes, null, (stage, done, total) => {
    if (total) process.stdout.write(`\r[asr] unpacking model ${(done / 1e6).toFixed(0)}/${(total / 1e6).toFixed(0)} MB   `);
  });
  const info = JSON.parse(phonon.info());
  console.log(`\n[asr] phonon-2 ready (${info.weight_bytes ? (info.weight_bytes / 1e6).toFixed(0) + ' MB weights, ' : ''}${info.threads ?? 1} thread(s))`);
  return phonon;
}

// audio: { pcm: Float32Array, sampleRate } -> [{ text, start, end }] word timings.
export async function transcribe({ pcm, sampleRate }) {
  enginePromise ??= loadEngine();
  const phonon = await enginePromise;
  // the engine wants 16 kHz mono float32
  const audio = sampleRate === 16000 ? pcm : resample(pcm, sampleRate, 16000);
  const result = JSON.parse(await phonon.transcribe(audio));
  const words = (result.words ?? [])
    .filter((w) => (w.text ?? '').trim())
    .map((w) => ({ text: String(w.text).trim(), start: Number(w.start), end: Number(w.end) }));
  return words;
}

export function resample(f32, from, to) {
  if (from === to) return f32;
  const ratio = from / to;
  const n = Math.floor(f32.length / ratio);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = i * ratio;
    const i0 = Math.floor(x);
    const i1 = Math.min(i0 + 1, f32.length - 1);
    out[i] = f32[i0] + (f32[i1] - f32[i0]) * (x - i0);
  }
  return out;
}
