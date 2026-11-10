// tts.js - on-device text-to-speech in Node via phonon-tts's own WebAssembly engine.
// The npm package targets browsers (worker + Cache API), so this adapter drives the raw
// wasm-bindgen `Model` directly (same engine, same checkpoints from the Hugging Face Hub).
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { ROOT } from './utils.js';

const require = createRequire(import.meta.url);

let wasmMod = null;
function loadWasm() {
  if (wasmMod) return wasmMod;
  const pkgDir = path.dirname(require.resolve('phonon-tts/package.json'));
  const gluePath = path.join(pkgDir, 'wasm', 'phonon_tts.js');
  const wasmPath = path.join(pkgDir, 'wasm', 'phonon_tts_bg.wasm');
  // The glue is an ES module; import it and compile the wasm explicitly (the browser glue
  // otherwise tries to fetch the .wasm relative to its URL, which Node cannot do).
  // pathToFileURL is REQUIRED on Windows: a raw `D:\...` path passed to import() is parsed
  // as a URL with protocol 'd:' -> "Only URLs with a scheme in: file, data, and node...".
  wasmMod = import(pathToFileURL(gluePath).href).then(async (glue) => {
    const bytes = await fs.promises.readFile(wasmPath);
    const mod = await WebAssembly.compile(bytes);
    glue.initSync({ module: mod });
    return glue;
  }).catch((e) => {
    wasmMod = null; // don't cache the rejected promise; a retry should try again
    throw e;
  });
  return wasmMod;
}

// Resolve an `hf://owner/name[@rev]` checkpoint (or a local folder) to concrete file URLs.
async function resolveModel(spec) {
  let root = spec;
  let localDir = null;
  if (spec.startsWith('hf://')) {
    let repo = spec.slice(5);
    let rev = 'main';
    if (repo.includes('@')) [repo, rev] = repo.split('@', 2);
    root = `https://huggingface.co/${repo}/resolve/${rev}`;
  } else if (fs.existsSync(spec)) {
    localDir = spec;
  }
  const url = (f) => (localDir ? path.join(localDir, f) : `${root}/${f}`);
  return url;
}

async function fetchIfNeeded(url, file, cacheDir, onProgress) {
  if (file && fs.existsSync(file)) return fs.readFileSync(file);
  const cachePath = path.join(cacheDir, encodeURIComponent(url));
  if (fs.existsSync(cachePath)) return fs.readFileSync(cachePath);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed (HTTP ${res.status}): ${url}`);
  const total = Number(res.headers.get('content-length') || 0);
  const chunks = [];
  let got = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress?.(total ? got / total : 0, got, total);
  }
  const buf = Buffer.concat(chunks);
  try {
    fs.mkdirSync(cacheDir, { recursive: true });
    fs.writeFileSync(cachePath, buf);
  } catch { /* cache best-effort */ }
  if (file) fs.writeFileSync(file, buf);
  return buf;
}

const HF31M = 'hf://gradium/phonon-31m-en';

export class PhononTTS {
  // cfg: { tts_model, tts_lang, tts_quant, voice }
  static async load(cfg) {
    const glue = await loadWasm();
    const spec = cfg.tts_model || HF31M;
    // The browser build only ships q8/f32 wasm paths; q4k requests fall back to q8.
    const quant = cfg.tts_quant === 'f32' ? 'f32' : 'q8';
    const cacheDir = path.join(ROOT, '.cache', 'tts', encodeURIComponent(spec).slice(0, 80));
    const url = await resolveModel(spec);
    const pick = async (file) => {
      try { return await fetchIfNeeded(url(file), null, cacheDir); } catch (e) { return null; }
    };
    const [weights, tokenizer, config] = await Promise.all([
      fetchIfNeeded(url(quant === 'f32' ? 'model.safetensors' : 'model.q8.gguf'), null, cacheDir,
        (f, got, total) => {
          if (total) process.stdout.write(`\r[tts] downloading weights ${(got / 1e6).toFixed(0)}/${(total / 1e6).toFixed(0)} MB   `);
        }),
      fetchIfNeeded(url('tokenizer.json'), null, cacheDir),
      fetchIfNeeded(url('config.json'), null, cacheDir),
    ]);
    if (weights.length) process.stdout.write('\n');
    console.log(`[tts] loading phonon model (${quant}, ${cfg.tts_lang}) - first run downloads it, then it is cached`);
    const model = await glue.Model.load(
      new Uint8Array(weights), new Uint8Array(tokenizer), new Uint8Array(config),
      quant, cfg.tts_lang || 'en', 'default', [], 'cpu');
    const tts = new PhononTTS();
    tts.model = model;
    tts.frameSize = model.frame_size();
    tts.sampleRate = model.sample_rate();
    tts.voices = {};
    // The checkpoint always ships default-voice.safetensors; named voices are repo files too.
    {
      const bytes = await pick('default-voice.safetensors');
      if (!bytes) throw new Error(`checkpoint ${spec} has no default-voice.safetensors`);
      const idx = model.add_voice(new Uint8Array(bytes));
      await model.flush();
      tts.voices.default = idx;
    }
    const wanted = cfg.voice && cfg.voice !== 'default' ? cfg.voice : null;
    for (const rel of wanted ? [`voices/${wanted}.safetensors`, `embeddings/${wanted}.safetensors`] : []) {
      const bytes = await pick(rel);
      if (bytes) {
        const idx = model.add_voice(new Uint8Array(bytes));
        await model.flush();
        tts.voices[wanted] = idx;
        break;
      }
    }
    if (wanted && tts.voices[wanted] === undefined) {
      console.log(`[tts] voice '${wanted}' is not in this checkpoint; using the default voice`);
    }
    return tts;
  }

  // Synthesize `text` into a 16-bit mono WAV at `wavPath`. Returns duration in seconds.
  async save(wavPath, text, { voice, temperature = 0.3, seed = 4242424242424242n } = {}) {
    const idx = this.voices[voice] ?? 0;
    const chunks = this.model.start_generation(idx, text, temperature, seed);
    const parts = [];
    try {
      for (;;) {
        const tokens = this.model.next_chunk();
        if (tokens === undefined) break;
        for (;;) {
          const pcm = await this.model.generation_step();
          if (pcm === undefined) break;
          parts.push(Float32Array.from(pcm));
        }
      }
    } finally {
      try { this.model.stop_generation(); } catch { /* after a wasm trap keep going */ }
    }
    const pcm = concatF32(parts);
    writeWav(wavPath, pcm, this.sampleRate);
    return pcm.length / this.sampleRate;
  }
}

function concatF32(parts) {
  const n = parts.reduce((a, b) => a + b.length, 0);
  const out = new Float32Array(n);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

export function writeWav(file, f32, sampleRate) {
  const n = f32.length;
  const wav = Buffer.alloc(44 + n * 2);
  wav.write('RIFF', 0);
  wav.writeUInt32LE(36 + n * 2, 4);
  wav.write('WAVEfmt ', 8);
  wav.writeUInt32LE(16, 16);
  wav.writeUInt16LE(1, 20);          // PCM
  wav.writeUInt16LE(1, 22);          // mono
  wav.writeUInt32LE(sampleRate, 24);
  wav.writeUInt32LE(sampleRate * 2, 28);
  wav.writeUInt16LE(2, 32);
  wav.writeUInt16LE(16, 34);
  wav.write('data', 36);
  wav.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    wav.writeInt16LE(Math.round(s * 32767), 44 + i * 2);
  }
  fs.writeFileSync(file, wav);
}

// Read a 16-bit PCM WAV back as mono Float32 at its native sample rate.
export function readWav(file) {
  const buf = fs.readFileSync(file);
  if (buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error(`not a WAV file: ${file}`);
  }
  let pos = 12;
  let fmt = null;
  let data = null;
  while (pos + 8 <= buf.length) {
    const id = buf.toString('ascii', pos, pos + 4);
    const size = buf.readUInt32LE(pos + 4);
    if (id === 'fmt ') {
      fmt = {
        channels: buf.readUInt16LE(pos + 10),
        sampleRate: buf.readUInt32LE(pos + 12),
        bits: buf.readUInt16LE(pos + 22),
      };
    } else if (id === 'data') {
      data = buf.subarray(pos + 8, pos + 8 + size);
      break;
    }
    pos += 8 + size + (size % 2);
  }
  if (!fmt || !data) throw new Error(`malformed WAV: ${file}`);
  const n = Math.floor(data.length / (fmt.bits / 8));
  const f32 = new Float32Array(n);
  for (let i = 0; i < n; i++) f32[i] = data.readInt16LE(i * 2) / 32768;
  // average down to mono
  if (fmt.channels > 1) {
    const mono = new Float32Array(Math.floor(n / fmt.channels));
    for (let i = 0; i < mono.length; i++) {
      let s = 0;
      for (let c = 0; c < fmt.channels; c++) s += f32[i * fmt.channels + c];
      mono[i] = s / fmt.channels;
    }
    return { pcm: mono, sampleRate: fmt.sampleRate };
  }
  return { pcm: f32, sampleRate: fmt.sampleRate };
}
