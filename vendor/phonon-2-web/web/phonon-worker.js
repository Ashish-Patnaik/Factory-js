// Engine worker: owns the wasm module, its rayon thread pool and (optionally) the WebGPU device.
//
// Requests  {id, cmd, ...args}  ->  replies {id, ok: true, result} | {id, ok: false, error}
// Events    {event: 'progress', stage, done, total} | {event: 'live', partial, finals, segment_s, decode_ms}
//
// Live audio arrives on a MessagePort straight from the AudioWorklet (16 kHz Float32Array blocks) and is drained in
// batches, so a slow decode never queues more than one re-decode of the segment in flight.

import { autoSelect, deviceSpecs, fingerprint } from './autoselect.js';

let pkg = null;
let build = null;
let phonon = null;
let ready = null;
let busy = Promise.resolve();
let livePort = null;
let livePending = [];
let liveRunning = false;
let liveActive = false;

const threads = Math.max(1, Math.min(navigator.hardwareConcurrency || 4, 16));

// (module (func (result v128) i32.const 1 i8x16.splat i32.const 2 i8x16.splat i8x16.relaxed_swizzle))
function relaxedSimd() {
  try {
    return WebAssembly.validate(new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 15, 1, 13, 0,
      65, 1, 253, 15, 65, 2, 253, 15, 253, 128, 2, 11,
    ]));
  } catch { return false; }
}

async function boot() {
  // The threaded builds need SharedArrayBuffer (a cross-origin-isolated page); otherwise use the single-threaded one.
  // Among the threaded builds, prefer relaxed SIMD (hardware FMA) where the engine validates it.
  const variant = !self.crossOriginIsolated ? 'pkg-st' : relaxedSimd() ? 'pkg-relaxed' : 'pkg';
  pkg = await import(`./${variant}/phonon_wasm.js`);
  build = variant;
  await pkg.default();
  let pool = 1;
  if (typeof pkg.initThreadPool === 'function') {
    await pkg.initThreadPool(threads);
    pool = threads;
  }
  let webgl = false;
  try { webgl = !!new OffscreenCanvas(1, 1).getContext('webgl2'); } catch { /* no OffscreenCanvas */ }
  return { threads: pool, build, isolated: self.crossOriginIsolated, webgpu: !!self.navigator.gpu, webgl };
}

// serialise every engine call
function serial(fn) {
  const run = busy.then(fn, fn);
  busy = run.catch(() => {});
  return run;
}

const handlers = {
  async boot() {
    ready ??= boot();
    return ready;
  },

  async load({ bytes, config }) {
    await (ready ??= boot());
    const progress = (stage, done, total) => postMessage({ event: 'progress', stage, done, total });
    const t0 = performance.now();
    phonon = pkg.Phonon.load(new Uint8Array(bytes), config ? new Uint8Array(config) : null, progress);
    return { ms: performance.now() - t0, info: JSON.parse(phonon.info()) };
  },

  // name: 'cpu' | 'webgpu' | 'webgl'; the first switch to a GPU backend uploads the encoder
  async backend({ name }) {
    if (!phonon) throw new Error('model not loaded');
    const t0 = performance.now();
    let adapter = null;
    if (name === 'webgpu') adapter = await phonon.enableGpu();
    else if (name === 'webgl') adapter = phonon.enableGl();
    else phonon.useBackend('cpu');
    return { adapter, ms: performance.now() - t0 };
  },

  // what the browser reveals about this device, and the cache key for a backend decision
  async specs({ disable = [] }) {
    await (ready ??= boot());
    const specs = await deviceSpecs(disable);
    return { specs, build, fingerprint: fingerprint(specs, build) };
  },

  // time the candidates on this device and switch to the fastest (see autoselect.js)
  async auto({ disable = [] }) {
    if (!phonon) throw new Error('model not loaded');
    const specs = await deviceSpecs(disable);
    const r = await autoSelect(phonon, specs, (msg) => postMessage({ event: 'progress', stage: msg, done: 0, total: 0 }));
    return { ...r, build, fingerprint: fingerprint(specs, build) };
  },

  async release({ name }) {
    phonon?.release(name);
    return true;
  },

  async transcribe({ audio }) {
    if (!phonon) throw new Error('model not loaded');
    return JSON.parse(await phonon.transcribe(new Float32Array(audio)));
  },

  async info() {
    return phonon ? JSON.parse(phonon.info()) : null;
  },

  async liveStart({ port }) {
    if (!phonon) throw new Error('model not loaded');
    phonon.liveReset();
    livePending = [];
    liveActive = true;
    livePort = port;
    livePort.onmessage = (e) => {
      if (!liveActive) return;
      livePending.push(e.data);
      pumpLive();
    };
    return true;
  },

  async liveStop() {
    liveActive = false;
    if (livePort) livePort.onmessage = null;
    livePort = null;
    // flush what arrived, then close the segment
    await serial(async () => {
      const rest = drain();
      if (rest) emitLive(JSON.parse(await phonon.liveFeed(rest)));
      emitLive(JSON.parse(await phonon.liveFinish()), true);
    });
    return true;
  },
};

function drain() {
  if (!livePending.length) return null;
  let n = 0;
  for (const b of livePending) n += b.length;
  const out = new Float32Array(n);
  let o = 0;
  for (const b of livePending) { out.set(b, o); o += b.length; }
  livePending = [];
  return out;
}

function emitLive(ev, done = false) {
  postMessage({ event: 'live', done, ...ev });
}

async function pumpLive() {
  if (liveRunning) return;
  liveRunning = true;
  try {
    while (liveActive && livePending.length) {
      const chunk = drain();
      const ev = await serial(async () => JSON.parse(await phonon.liveFeed(chunk)));
      emitLive(ev);
    }
  } catch (err) {
    postMessage({ event: 'error', error: String(err?.message || err) });
  } finally {
    liveRunning = false;
  }
}

self.onmessage = async (e) => {
  const { id, cmd, ...args } = e.data;
  const h = handlers[cmd];
  if (!h) return postMessage({ id, ok: false, error: `unknown command ${cmd}` });
  try {
    const result = cmd === 'liveStart' || cmd === 'liveStop' || cmd === 'boot' ? await h(args) : await serial(() => h(args));
    postMessage({ id, ok: true, result });
  } catch (err) {
    postMessage({ id, ok: false, error: String(err?.message || err) });
  }
};
