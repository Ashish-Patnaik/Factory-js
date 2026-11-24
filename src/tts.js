// tts.js - on-device text-to-speech in Node via n4ze3m/tiny-voca's plain-JS engine (js/say.mjs)
// and single 8 MB GGUF bundle (gguf/tiny-voca-js.gguf). Zero npm dependencies, 24 kHz mono WAV.
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { ROOT } from './utils.js';

export const DEFAULT_TTS_MODEL = 'hf://n4ze3m/tiny-voca';
export const DEFAULT_VOICE = 'M2';
const TINY_VOCA_REPO = 'https://github.com/n4ze3m/tiny-voca.git';
const TINY_VOCA_API_JS = 'https://api.github.com/repos/n4ze3m/tiny-voca/contents/js';
const GGUF_REL_PATH = 'gguf/tiny-voca-js.gguf';
const TTS_ROOT = path.join(ROOT, '.cache', 'tts');
const TINY_VOCA_DIR = path.join(TTS_ROOT, 'tiny-voca');
const ENGINE_DIR = path.join(TINY_VOCA_DIR, 'js');
const SAY_MJS = path.join(ENGINE_DIR, 'say.mjs');
const DEFAULT_GGUF = path.join(TINY_VOCA_DIR, 'tiny-voca-js.gguf');

// Remove stale phonon-tts checkpoints from .cache/tts so CI caches stay tiny (~8 MB).
export function pruneLegacyTtsCache() {
  if (!fs.existsSync(TTS_ROOT)) return;
  for (const entry of fs.readdirSync(TTS_ROOT)) {
    if (entry === 'tiny-voca') continue;
    try {
      fs.rmSync(path.join(TTS_ROOT, entry), { recursive: true, force: true });
      console.log(`[tts] removed legacy TTS cache entry: .cache/tts/${entry}`);
    } catch { /* best effort */ }
  }
}

async function downloadFile(url, destPath, label = 'file') {
  fs.mkdirSync(path.dirname(destPath), { recursive: true });
  const tmpPath = `${destPath}.part`;
  const res = await fetch(url, { redirect: 'follow', signal: AbortSignal.timeout(120000) });
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
    if (total) {
      process.stdout.write(`\r[tts] downloading ${label} ${(got / 1e6).toFixed(1)}/${(total / 1e6).toFixed(1)} MB   `);
    }
  }
  if (total) process.stdout.write('\n');
  fs.writeFileSync(tmpPath, Buffer.concat(chunks));
  fs.renameSync(tmpPath, destPath);
  return destPath;
}

async function fetchJsDirViaGitHubApi(destDir) {
  const headers = { 'User-Agent': 'ai-content-factory', Accept: 'application/vnd.github+json' };
  if (process.env.GITHUB_TOKEN) headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;

  const fetchDir = async (apiUrl, outDir) => {
    const res = await fetch(apiUrl, { headers, signal: AbortSignal.timeout(30000) });
    if (!res.ok) throw new Error(`GitHub API HTTP ${res.status} for ${apiUrl}`);
    const items = await res.json();
    fs.mkdirSync(outDir, { recursive: true });
    for (const item of items) {
      if (item.type === 'dir') {
        await fetchDir(item.url, path.join(outDir, item.name));
      } else if (item.type === 'file' && item.download_url) {
        await downloadFile(item.download_url, path.join(outDir, item.name), item.name);
      }
    }
  };
  await fetchDir(TINY_VOCA_API_JS, destDir);
}

export async function ensureTinyVocaEngine() {
  if (fs.existsSync(SAY_MJS)) return SAY_MJS;

  console.log('[tts] fetching tiny-voca plain-JS runtime into .cache/tts/tiny-voca/js ...');
  const tmpRepo = fs.mkdtempSync(path.join(os.tmpdir(), 'tiny_voca_js_'));
  try {
    const clone = spawnSync(
      'git',
      ['clone', '--depth', '1', '--filter=blob:none', '--sparse', TINY_VOCA_REPO, tmpRepo],
      { encoding: 'utf8', timeout: 60000 },
    );
    const sparse = clone.status === 0
      ? spawnSync('git', ['-C', tmpRepo, 'sparse-checkout', 'set', 'js'], { encoding: 'utf8', timeout: 60000 })
      : { status: 1 };

    const srcJs = path.join(tmpRepo, 'js');
    if (clone.status === 0 && sparse.status === 0 && fs.existsSync(path.join(srcJs, 'say.mjs'))) {
      fs.mkdirSync(TINY_VOCA_DIR, { recursive: true });
      fs.cpSync(srcJs, ENGINE_DIR, { recursive: true });
    } else {
      await fetchJsDirViaGitHubApi(ENGINE_DIR);
    }
  } finally {
    fs.rmSync(tmpRepo, { recursive: true, force: true });
  }

  if (!fs.existsSync(SAY_MJS)) {
    throw new Error(`tiny-voca JS engine missing say.mjs in ${ENGINE_DIR}`);
  }
  return SAY_MJS;
}

export async function ensureTinyVocaModel(spec = DEFAULT_TTS_MODEL) {
  const s = String(spec || DEFAULT_TTS_MODEL).trim();

  // Direct local .gguf path override
  if (s && !s.startsWith('hf://') && !/^https?:\/\//i.test(s) && fs.existsSync(s)) {
    return path.resolve(s);
  }

  if (fs.existsSync(DEFAULT_GGUF) && fs.statSync(DEFAULT_GGUF).size > 1_000_000) {
    return DEFAULT_GGUF;
  }

  let url;
  if (/^https?:\/\//i.test(s)) {
    url = s;
  } else {
    let repo = s.startsWith('hf://') ? s.slice(5) : 'n4ze3m/tiny-voca';
    let rev = 'main';
    if (repo.includes('@')) [repo, rev] = repo.split('@', 2);
    url = `https://huggingface.co/${repo}/resolve/${rev}/${GGUF_REL_PATH}`;
  }

  console.log('[tts] downloading tiny-voca-js.gguf (~8 MB, cached in .cache/tts/tiny-voca) ...');
  await downloadFile(url, DEFAULT_GGUF, 'tiny-voca-js.gguf');
  return DEFAULT_GGUF;
}

function normalizeVoice(v) {
  const raw = String(v ?? '').trim().toUpperCase();
  if (raw === 'M2' || raw === 'F2') return raw;
  if (raw && raw !== 'DEFAULT') {
    console.log(`[tts] voice '${v}' is not in tiny-voca (available: M2, F2); using ${DEFAULT_VOICE}`);
  }
  return DEFAULT_VOICE;
}

function splitForSynthesis(text, maxChars = 1500) {
  let clean = String(text ?? '').replace(/\s+/g, ' ').trim();
  if (!clean) return [];
  if (!/[.!?]$/.test(clean)) clean += '.';
  if (clean.length <= maxChars) return [clean];

  const sentences = clean.match(/[^.!?]+[.!?]+|[^.!?]+$/g) ?? [clean];
  const chunks = [];
  let cur = '';
  for (const s of sentences) {
    const piece = s.trim();
    if (!piece) continue;
    if (cur && (cur.length + 1 + piece.length) > maxChars) {
      chunks.push(cur);
      cur = piece;
    } else {
      cur = cur ? `${cur} ${piece}` : piece;
    }
  }
  if (cur) chunks.push(cur);
  return chunks;
}

function runSayMjs(sayPath, ggufPath, text, voice, outWav) {
  return new Promise((resolve, reject) => {
    const args = [sayPath, text, '--voice', voice, '--file', ggufPath, '--output', outWav];
    const proc = spawn(process.execPath, args, {
      cwd: path.dirname(sayPath),
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    const err = [];
    proc.stderr.on('data', (c) => err.push(c));
    proc.on('error', reject);
    proc.on('close', (code) => {
      if (code === 0 && fs.existsSync(outWav)) return resolve();
      const detail = Buffer.concat(err).toString('utf8').trim().split('\n').slice(-2).join(' | ');
      reject(new Error(`tiny-voca synthesis failed (exit ${code}): ${detail || 'no output WAV created'}`));
    });
  });
}

export class TinyVocaTTS {
  constructor(sayPath, ggufPath) {
    this.sayPath = sayPath;
    this.ggufPath = ggufPath;
    this.sampleRate = 24000;
  }

  static async load(cfg = {}) {
    pruneLegacyTtsCache();
    const [sayPath, ggufPath] = await Promise.all([
      ensureTinyVocaEngine(),
      ensureTinyVocaModel(cfg.tts_model || DEFAULT_TTS_MODEL),
    ]);
    console.log(`[tts] tiny-voca ready (${path.relative(ROOT, ggufPath)}, default voice: ${normalizeVoice(cfg.voice || DEFAULT_VOICE)})`);
    return new TinyVocaTTS(sayPath, ggufPath);
  }

  // Synthesize `text` into a 16-bit mono 24 kHz WAV at `wavPath`. Returns duration in seconds.
  async save(wavPath, text, { voice = DEFAULT_VOICE } = {}) {
    const v = normalizeVoice(voice);
    const chunks = splitForSynthesis(text);
    if (!chunks.length) throw new Error('cannot synthesize empty text');

    fs.mkdirSync(path.dirname(path.resolve(wavPath)) || '.', { recursive: true });

    if (chunks.length === 1) {
      await runSayMjs(this.sayPath, this.ggufPath, chunks[0], v, path.resolve(wavPath));
      const { pcm, sampleRate } = readWav(wavPath);
      this.sampleRate = sampleRate;
      return pcm.length / sampleRate;
    }

    // Long scripts (--long): synthesize per chunk and stitch with a 180 ms pause between chunks
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tvoca_'));
    try {
      const parts = [];
      let sr = 24000;
      for (let i = 0; i < chunks.length; i++) {
        const partWav = path.join(tmpDir, `part_${i}.wav`);
        await runSayMjs(this.sayPath, this.ggufPath, chunks[i], v, partWav);
        const { pcm, sampleRate } = readWav(partWav);
        sr = sampleRate;
        parts.push(pcm);
        if (i < chunks.length - 1) parts.push(new Float32Array(Math.round(sr * 0.18)));
      }
      const combined = concatF32(parts);
      writeWav(wavPath, combined, sr);
      this.sampleRate = sr;
      return combined.length / sr;
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
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