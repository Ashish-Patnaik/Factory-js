// encoder/host.js - Node-side driver for the browser "encoder page".
//
// The host launches a headless Chromium/Chrome/Edge page (a file:// context = secure context,
// so WebCodecs is available), injects mediabunny + @mediabunny/aac-encoder + page.js as blob
// modules, and drives the encode through small RPC calls. Frames captured from the target
// page (JPEG) are batched to the encoder page, composed (captions, avatar, fit/pad) and
// encoded to H.264/AAC MP4 entirely in the browser - no ffmpeg anywhere.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';
import { ROOT } from '../utils.js';

const require = createRequire(import.meta.url);
const PAGE_SRC = fs.readFileSync(path.join(ROOT, 'src', 'encoder', 'page.js'), 'utf8');

// package exports do not expose package.json / bundles subpaths: resolve by walking node_modules
function pkgDir(name) {
  let dir = ROOT;
  for (;;) {
    const cand = path.join(dir, 'node_modules', name);
    if (fs.existsSync(cand)) return cand;
    const parent = path.dirname(dir);
    if (parent === dir) throw new Error(`package not installed: ${name}`);
    dir = parent;
  }
}

// Which browsers to try, in order, for the encoder page. Chrome/Edge ship the proprietary
// codecs natively; the bundled Chromium works too (the WASM AAC extension covers audio).
export async function launchEncoderBrowser({ headed = false, channel = null } = {}) {
  const args = browserArgs();
  const tries = [];
  if (channel) tries.push({ channel });
  for (const c of ['chrome', 'msedge']) {
    if (c !== channel) tries.push({ channel: c });
  }
  tries.push({}); // bundled Chromium as the final fallback
  let lastErr = null;
  for (const opts of tries) {
    try {
      const browser = await chromium.launch({ headless: !headed, args, ...opts });
      return { browser, used: opts.channel ?? 'chromium' };
    } catch (e) {
      lastErr = e; // channel not installed -> try the next one
    }
  }
  throw lastErr ?? new Error('no browser available for the encoder');
}

function browserArgs() {
  const args = [
    '--no-first-run', '--no-default-browser-check', '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
  ];
  if (os.platform() === 'linux' && (process.getuid?.() ?? -1) === 0) args.push('--no-sandbox');
  return args;
}

export class EncoderHost {
  constructor() { this.browser = null; this.page = null; }

  async launch({ headed = false, channel = null } = {}) {
    const { browser, used } = await launchEncoderBrowser({ headed, channel });
    this.browser = browser;
    this.usedChannel = used;
    this.context = await this.browser.newContext();
    // a file:// page = secure context = WebCodecs available; stub html lives in .cache
    const stub = path.join(ROOT, '.cache', 'encoder.html');
    fs.mkdirSync(path.dirname(stub), { recursive: true });
    fs.writeFileSync(stub, '<!doctype html><title>factory encoder</title>');
    this.page = await this.context.newPage();
    this.page.on('pageerror', (e) => console.error('[encoder-page]', String(e?.message ?? e).split('\n')[0]));
    this.page.on('console', (m) => {
      if (m.type() === 'error') console.error('[encoder-page]', m.text().slice(0, 300));
    });
    // pathToFileURL: keeps a file: URL (secure context for WebCodecs) that is also valid on
    // Windows - 'file://' + 'D:\...' is not.
    await this.page.goto(pathToFileURL(stub).href);

    // 1) mediabunny ESM bundle -> blob URL (shared by everything on the page)
    // (the bundles/ subpath is not in package exports; resolve the package dir directly)
    const mbDir = pkgDir('mediabunny');
    const mbSrc = fs.readFileSync(path.join(mbDir, 'dist', 'bundles', 'mediabunny.mjs'), 'utf8');
    this.mbUrl = await this.page.evaluate((src) => URL.createObjectURL(new Blob([src], { type: 'text/javascript' })), mbSrc);
    // 2) @mediabunny/aac-encoder bundle -> blob URL, rewritten to import the same mediabunny
    const aacDir = pkgDir('@mediabunny/aac-encoder');
    const aacSrc = fs.readFileSync(path.join(aacDir, 'dist', 'bundles', 'mediabunny-aac-encoder.mjs'), 'utf8')
      .replace(/(["'])mediabunny\1/g, `"${this.mbUrl}"`);
    this.aacUrl = await this.page.evaluate((src) => URL.createObjectURL(new Blob([src], { type: 'text/javascript' })), aacSrc);
    // 3) page.js -> blob URL with the two import tokens rewritten
    const pageSrc = PAGE_SRC
      .replaceAll("'__MEDIABUNNY_URL__'", JSON.stringify(this.mbUrl))
      .replaceAll("'__AAC_ENCODER_URL__'", JSON.stringify(this.aacUrl));
    this.pageUrl = await this.page.evaluate((src) => URL.createObjectURL(new Blob([src], { type: 'text/javascript' })), pageSrc);
    const ok = await this.page.evaluate(async (u) => {
      const m = await import(u);
      return typeof m.api?.init === 'function';
    }, this.pageUrl);
    if (!ok) throw new Error('encoder page module failed to load');
    return this;
  }

  rpc(method, ...args) {
    return this.page.evaluate(async ({ url, method, args }) => {
      const m = await import(url);
      return m.api[method](...args);
    }, { url: this.pageUrl, method, args });
  }

  // Binary args (Uint8Array/Float32Array) go through evaluate's structured serialization.
  rpcBin(method, ...args) {
    return this.page.evaluate(async ({ url, method, args }) => {
      const m = await import(url);
      return m.api[method](...args);
    }, { url: this.pageUrl, method, args });
  }

  async close() {
    try { await this.browser?.close(); } catch { /* already gone */ }
    this.browser = this.page = null;
  }

  // ---------------- high-level factory helpers ----------------

  async initEncoder(opts) { return this.rpc('init', opts); }

  async setFont(ttfBuffer) { return this.rpcBin('setFont', new Uint8Array(ttfBuffer)); }

  async setCaptions(data) { return this.rpc('setCaptions', data); }

  async setAvatar(gifBuffer, corner, sizePct) {
    return this.rpcBin('setAvatar', new Uint8Array(gifBuffer), corner, sizePct);
  }

  async start() { return this.rpc('start'); }
  async finish() { return this.rpc('finish'); }

  // Send the TTS waveform (Float32 mono at `sampleRate`) in chunks; the page resamples to
  // 44.1 kHz, applies the start delay and pads to the video length.
  async deliverAudio(pcm, { sampleRate, audioDelay, totalDuration }) {
    await this.rpc('beginAudio', { sampleRate, audioDelay, totalDuration });
    const CHUNK = 2 * 1024 * 1024; // 2M floats = 8 MB
    for (let i = 0; i < pcm.length; i += CHUNK) {
      const part = pcm.slice(i, Math.min(i + CHUNK, pcm.length));
      await this.rpcBin('pushAudio', part);
    }
  }

  // Push a batch of captured frames; returns the encoder's frame counter.
  async addFrameBatch(batch) { return this.rpc('addFrames', batch); }

  // Pull the finalized MP4 back to Node and write it to `outPath`.
  async pullToFile(outPath, onProgress) {
    const chunks = [];
    for (;;) {
      const { b64, done, total } = await this.rpc('pullBytes');
      chunks.push(Buffer.from(b64, 'base64'));
      onProgress?.(chunks.reduce((a, b) => a + b.length, 0), total);
      if (done) break;
    }
    const buf = Buffer.concat(chunks);
    await fs.promises.mkdir(path.dirname(path.resolve(outPath)), { recursive: true });
    await fs.promises.writeFile(outPath, buf);
    return buf;
  }
}
