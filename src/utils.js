// utils.js - env loading, errors, URL validation, media probing (mediabunny replaces ffprobe).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Input, ALL_FORMATS, BlobSource, BufferSource, FilePathSource } from 'mediabunny';

export const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

// ---------------------------------------------------------------------------
// .env loading (same files & rules as the Python factory)
// ---------------------------------------------------------------------------
export function loadEnv() {
  for (const fname of ['.env', 'env.txt', 'insta env.txt']) {
    const p = path.join(ROOT, fname);
    if (!fs.existsSync(p)) continue;
    for (let line of fs.readFileSync(p, 'utf8').split(/\r?\n/)) {
      line = line.trim();
      if (!line || line.startsWith('#') || !line.includes('=')) continue;
      if (line.startsWith('export ')) line = line.slice(7).trim();
      else if (line.toLowerCase().startsWith('$env:')) line = line.slice(5).trim();
      const eq = line.indexOf('=');
      const k = line.slice(0, eq).trim();
      let v = line.slice(eq + 1).trim();
      v = v.replace(/^'/, '').replace(/'$/, '').replace(/^"/, '').replace(/"$/, '');
      if (k && !(k in process.env)) process.env[k] = v;
    }
  }
}
loadEnv(); // run immediately when utils is imported anywhere (same as Python)

export function die(msg) {
  console.error(`error: ${msg}`);
  // Repeat on stdout only when stderr is redirected (CI / pipes), so interactive terminals
  // don't show the same error twice.
  if (process.env.CI || !process.stderr.isTTY) console.log(`error: ${msg}`);
  process.exit(1);
}

export function validateUrl(url) {
  if (!url.includes('://')) url = 'https://' + url;
  let p;
  try { p = new URL(url); } catch { die(`invalid URL: ${url}`); }
  if (!['http:', 'https:'].includes(p.protocol) || !p.hostname) die(`invalid URL: ${url}`);
  return url;
}

export function humanSize(n) {
  return `${(n / 1e6).toFixed(1)} MB`;
}

// ---------------------------------------------------------------------------
// Media probing via mediabunny (pure JS: no ffprobe anywhere in this port)
// ---------------------------------------------------------------------------
export async function probeMedia(filePathOrBuffer) {
  const isBuf = Buffer.isBuffer(filePathOrBuffer);
  const source = isBuf ? new BufferSource(filePathOrBuffer) : new FilePathSource(filePathOrBuffer);
  const size = isBuf ? filePathOrBuffer.length : (await fs.promises.stat(filePathOrBuffer)).size;
  const input = new Input({ formats: ALL_FORMATS, source });
  try {
    const [videoTrack, audioTrack, format] = await Promise.all([
      input.getPrimaryVideoTrack().catch(() => null),
      input.getPrimaryAudioTrack().catch(() => null),
      input.getFormat(),
    ]);
    const out = {
      container: format?.name ?? null,
      duration: 0,
      size,
      video: null,
      audio: null,
    };
    const [vd, ad] = await Promise.all([
      videoTrack?.getDurationFromMetadata() ?? null,
      audioTrack?.getDurationFromMetadata() ?? null,
    ]);
    // some muxers store the video track duration in microseconds - normalize implausible values
    const norm = (x) => (x !== null && x > 1e5 ? x / 1e6 : x ?? 0);
    out.duration = Math.max(norm(vd), norm(ad));
    if (videoTrack) {
      const stats = await videoTrack.computePacketStats(Number.MAX_SAFE_INTEGER);
      let fps = stats.averagePacketRate > 0 ? stats.averagePacketRate * 1e6 : 0; // packet rate is per-microsecond
      if (out.duration > 0) fps = stats.packetCount / out.duration; // exact from full enumeration
      out.video = {
        codec: videoTrack.codec,
        width: videoTrack.codedWidth,
        height: videoTrack.codedHeight,
        fps: Math.round(fps * 100) / 100,
        rotation: videoTrack.rotation,
      };
    }
    if (audioTrack) {
      out.audio = {
        codec: audioTrack.codec,
        sampleRate: audioTrack.sampleRate,
        channels: audioTrack.numberOfChannels,
      };
    }
    return out;
  } finally {
    try { await input.cancel?.(); } catch { /* not cancellable mid-read */ }
  }
}

// ffprobe-style one-line spec summary (keeps the log format users know from the Python factory)
export async function probeLine(p) {
  const m = await probeMedia(p);
  const v = m.video, a = m.audio;
  const vDesc = v ? `${v.codec} ${v.width}x${v.height} @${v.fps.toFixed(2)}fps` : 'NO VIDEO';
  const aDesc = a ? `${a.codec}/${a.sampleRate}Hz` : 'MISSING';
  return `${vDesc} | audio ${aDesc} | ${m.duration.toFixed(1)}s, ${humanSize(m.size)}`;
}

// Full replacement of Python utils.probe(): prints the finished file's real specs.
export async function probe(p) {
  let m;
  try { m = await probeMedia(p); } catch { return console.log('(mediabunny could not read the file; skipping verification)'); }
  const v = m.video, a = m.audio;
  console.log(`  file      : ${path.resolve(p)}`);
  console.log(`  codec     : ${v?.codec ?? '?'} in ${m.container}`);
  console.log(`  size      : ${v ? `${v.width}x${v.height}` : '?'}`);
  console.log(`  fps       : ${v ? v.fps.toFixed(2) : '?'}   duration: ${m.duration.toFixed(2)}s   size: ${humanSize(m.size)}`);
  console.log(`  audio     : ${a ? `${a.codec} ${a.sampleRate}Hz ${a.channels}ch` : 'NONE'}`);
}

// Media duration in seconds (replacement of Python utils.duration()).
export async function duration(p) {
  const m = await probeMedia(p);
  if (!m.duration) die(`could not read duration of ${p}`);
  return m.duration;
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Python-style page.evaluate: the snippet string is an arrow-function expression that must be
// auto-invoked (Python Playwright does this; Node Playwright needs the explicit wrap).
export function evFn(src) {
  // eslint-disable-next-line no-new-func
  return new Function(`return (${src})`)();
}

export async function ev(page, src, ...args) {
  return page.evaluate(evFn(src), ...args);
}

// Rejected-promise helper so top-level awaits never crash before die() prints.
export function bail(fn) {
  return Promise.resolve().then(fn).catch((e) => die(e?.message ?? String(e)));
}
