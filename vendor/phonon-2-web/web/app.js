// Page logic: model download + cache, backend choice (automatic by default), live dictation, file transcription.
// The engine itself lives in phonon-worker.js; backend selection in autoselect.js.
import { gpuCandidates, LABEL as BACKEND_LABEL, tidyRenderer } from './autoselect.js';

const DEFAULT_MODEL = 'https://huggingface.co/FermionResearch/Phonon-2/resolve/main/phonon-2.bps.tar.zst';
const CACHE = 'phonon-2-model-v1';
const params = new URLSearchParams(location.search);
const MODEL_URL = params.get('model') || DEFAULT_MODEL;
const DISABLE = (params.get('disable') || '').split(',').filter(Boolean); // e.g. ?disable=webgpu

const $ = (id) => document.getElementById(id);
const fmtMB = (n) => `${(n / 1e6).toFixed(n < 10e6 ? 1 : 0)} MB`;
const fmtS = (ms) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`);

// ---------------------------------------------------------------- worker rpc
const worker = new Worker(new URL('./phonon-worker.js', import.meta.url), { type: 'module' });
let nextId = 1;
const pending = new Map();
const listeners = { progress: [], live: [], error: [] };
worker.onmessage = (e) => {
  const m = e.data;
  if (m.event) return (listeners[m.event] || []).forEach((f) => f(m));
  const p = pending.get(m.id);
  if (!p) return;
  pending.delete(m.id);
  m.ok ? p.resolve(m.result) : p.reject(new Error(m.error));
};
worker.onerror = (e) => setModelState(`worker error: ${e.message || e}`, 'err');
function rpc(cmd, args = {}, transfer = []) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, cmd, ...args }, transfer);
  });
}
const on = (ev, f) => listeners[ev].push(f);

// ---------------------------------------------------------------- state
const state = { loaded: false, backend: 'cpu', pref: 'auto', boot: null, fingerprint: null };

function setModelState(text, kind = '') {
  const el = $('model-state');
  el.textContent = text;
  el.className = `pill ${kind}`;
}

function showProgress(label, frac) {
  $('progress').hidden = false;
  $('progress-fill').style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`;
  $('progress-text').textContent = label;
}

function hideProgress() {
  $('progress').hidden = true;
}

function setFacts(items) {
  const f = $('facts');
  f.hidden = false;
  f.innerHTML = items.map(([v, k]) => `<div class="fact"><b>${v}</b><span>${k}</span></div>`).join('');
}

// ---------------------------------------------------------------- boot
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage blocked */ } },
};
const PREF = 'phonon-backend-pref'; // 'auto' | 'cpu' | 'webgpu' | 'webgl'
const DECISION = 'phonon-backend-auto'; // {fingerprint, choice, reason, speeds}

(async () => {
  try {
    state.boot = await rpc('boot');
    const b = state.boot;
    $('cpu-sub').textContent = `SIMD${b.build === 'pkg-relaxed' ? ' + FMA' : ''} · ${b.threads} thread${b.threads === 1 ? '' : 's'}`;
    const sp = await rpc('specs', { disable: DISABLE });
    state.fingerprint = sp.fingerprint;
    describeDevice(sp.specs, b);
    const { skipped } = gpuCandidates(sp.specs);
    for (const name of DISABLE) skipped[name] ??= 'disabled by ?disable=';
    for (const [name, why] of Object.entries(skipped)) {
      $(`be-${name}`).disabled = true;
      $(`${name}-sub`).textContent = why;
    }
    markBackend();
    // a model already in the browser cache loads straight away
    if (!params.get('noautoload') && (await isCached(MODEL_URL))) loadModel(() => fetchModel(MODEL_URL));
  } catch (err) {
    setModelState('engine failed to start', 'err');
    $('backend-note').textContent = String(err.message || err);
  }
})();

function describeDevice(sp, b) {
  // the WebGL renderer string is usually the most descriptive name a browser reveals
  const gpu = sp.webgl && !sp.webgl.software ? tidyRenderer(sp.webgl.renderer) : sp.webgpu && !sp.webgpu.software ? sp.webgpu.name : 'no hardware GPU access';
  const bits = [tidyRenderer(gpu), `${sp.threads} CPU threads`];
  if (!b.isolated) bits.push('not cross-origin isolated: CPU runs single-threaded');
  $('device').textContent = bits.join(' · ');
  if (!b.isolated && window.top !== window) {
    $('backend-note').innerHTML = 'Embedded pages cannot use threads. <a href="' + location.href + '" target="_blank" rel="noopener">Open the full page</a> for the multi-threaded CPU engine.';
  }
}

async function isCached(url) {
  try { return !!(await (await caches.open(CACHE)).match(url)); } catch { return false; }
}

on('progress', (m) => {
  if (m.total) showProgress(`Unpacking weights… ${Math.round((100 * m.done) / m.total)}%`, m.done / m.total);
});
on('error', (m) => {
  $('live-stat').textContent = `Error: ${m.error}`;
});

// ---------------------------------------------------------------- model
async function fetchModel(url) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch { /* no Cache Storage (private mode, file://) */ }
  const hit = cache && (await cache.match(url));
  if (hit) {
    showProgress('Reading model from browser cache…', 1);
    return { bytes: await hit.arrayBuffer(), cached: true };
  }
  const res = await fetch(url, { mode: 'cors' });
  if (!res.ok) throw new Error(`download failed: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 163515201;
  const reader = res.body.getReader();
  const buf = new Uint8Array(total);
  let got = 0;
  const extra = [];
  const t0 = performance.now();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (got + value.length <= buf.length) buf.set(value, got);
    else extra.push(value);
    got += value.length;
    const rate = got / ((performance.now() - t0) / 1000);
    showProgress(`Downloading ${fmtMB(got)} of ${fmtMB(total)} · ${fmtMB(rate)}/s`, got / total);
  }
  let bytes = buf;
  if (got !== total) {
    bytes = new Uint8Array(got);
    bytes.set(buf.subarray(0, Math.min(got, total)));
    let o = Math.min(got, total);
    for (const v of extra) { bytes.set(v, o); o += v.length; }
  }
  if (cache) {
    try { await cache.put(url, new Response(bytes.slice(0), { headers: { 'content-type': 'application/octet-stream' } })); } catch { /* quota */ }
  }
  return { bytes: bytes.buffer, cached: false };
}

async function loadModel(getBytes) {
  $('load-btn').disabled = true;
  setModelState('loading…', 'busy');
  try {
    const { bytes, config, cached } = await getBytes();
    const size = bytes.byteLength;
    showProgress('Unpacking weights…', 0);
    const r = await rpc('load', { bytes, config }, [bytes, ...(config ? [config] : [])]);
    hideProgress();
    state.loaded = true;
    setModelState('ready', 'ok');
    $('load-btn').textContent = 'Loaded';
    $('model-hint').textContent = cached === undefined ? 'Loaded from local files.' : cached ? 'Loaded from the browser cache.' : 'Downloaded and cached for next time.';
    setFacts([
      [fmtMB(size), 'download'],
      [fmtMB(r.info.weight_bytes), 'weights in memory'],
      [fmtS(r.ms), 'unpack time'],
      [`${r.info.threads}`, r.info.threads === 1 ? 'CPU thread' : 'CPU threads'],
    ]);
    $('mic-btn').disabled = false;
    $('sample-btn').disabled = false;
    await applyPreference();
  } catch (err) {
    hideProgress();
    setModelState('failed', 'err');
    $('model-hint').textContent = String(err.message || err);
    $('load-btn').disabled = false;
  }
}

$('load-btn').onclick = () => loadModel(() => fetchModel(MODEL_URL));

$('model-file').onchange = (e) => {
  const files = [...e.target.files];
  if (!files.length) return;
  loadModel(async () => {
    const zst = files.find((f) => f.name.endsWith('.zst'));
    if (zst) return { bytes: await zst.arrayBuffer() };
    const fermion = files.find((f) => f.name.endsWith('.fermion'));
    const cfg = files.find((f) => f.name.endsWith('.json'));
    if (!fermion || !cfg) throw new Error('choose phonon-2.bps.tar.zst, or model.fermion together with config.json');
    return { bytes: await fermion.arrayBuffer(), config: await cfg.arrayBuffer() };
  });
};

// ---------------------------------------------------------------- backend
for (const btn of document.querySelectorAll('[data-backend]')) {
  btn.onclick = () => (btn.dataset.backend === 'auto' ? chooseAuto(true) : chooseManual(btn.dataset.backend));
}

function markBackend() {
  const sel = state.pref === 'auto' ? 'auto' : state.backend;
  for (const b of document.querySelectorAll('[data-backend]')) b.setAttribute('aria-checked', String(b.dataset.backend === sel));
}

async function applyPreference() {
  state.pref = store.get(PREF) || 'auto';
  if (state.pref === 'auto') return chooseAuto(false);
  const ok = await chooseManual(state.pref);
  if (!ok) await chooseAuto(false);
}

/** Use the cached decision for this device, or time the candidates (`force` re-times). */
async function chooseAuto(force) {
  if (!state.loaded) return;
  state.pref = 'auto';
  store.set(PREF, 'auto');
  markBackend();
  const note = $('backend-note');
  const cached = store.get(DECISION);
  if (!force && cached && cached.fingerprint === state.fingerprint) {
    try {
      const r = await rpc('backend', { name: cached.choice });
      return settle(cached.choice, r.adapter, `Auto: ${BACKEND_LABEL[cached.choice]}, ${cached.reason}.`);
    } catch { /* the GPU went away: time again */ }
  }
  note.textContent = 'Finding the fastest backend on this device…';
  $('auto-sub').textContent = 'timing…';
  try {
    const r = await rpc('auto', { disable: DISABLE });
    store.set(DECISION, { fingerprint: r.fingerprint, choice: r.choice, reason: r.reason, speeds: r.speeds, at: Date.now() });
    settle(r.choice, null, `Auto: ${BACKEND_LABEL[r.choice]}, ${r.reason}.`);
  } catch (err) {
    note.textContent = `Auto-selection failed (${err.message || err}); using the CPU.`;
    await rpc('backend', { name: 'cpu' }).catch(() => {});
    settle('cpu', null, null);
  }
}

async function chooseManual(which) {
  if (!state.loaded) {
    $('backend-note').textContent = 'Load the model first.';
    return false;
  }
  const note = $('backend-note');
  try {
    if (which !== 'cpu') note.textContent = `Starting ${BACKEND_LABEL[which]} and uploading the encoder…`;
    const r = await rpc('backend', { name: which });
    state.pref = which;
    store.set(PREF, which);
    settle(which, r.adapter, which === 'cpu'
      ? 'Encoder and decoder on the CPU (chosen by you; pick Auto to let the page decide).'
      : `Encoder on ${r.adapter} via ${BACKEND_LABEL[which]} (${fmtS(r.ms)} to start); the decoder stays on the CPU.`);
    return true;
  } catch (err) {
    note.textContent = `${BACKEND_LABEL[which]} unavailable: ${err.message || err}`;
    return false;
  }
}

function settle(backend, adapter, text) {
  state.backend = backend;
  if (adapter) $(`${backend}-sub`).textContent = adapter;
  $('auto-sub').textContent = state.pref === 'auto' ? `→ ${BACKEND_LABEL[backend]}` : 'let the page decide';
  if (text) $('backend-note').textContent = text;
  markBackend();
}

// ---------------------------------------------------------------- live dictation
const live = { on: false, ctx: null, stream: null, node: null, finals: [], partial: '' };

function renderLive() {
  const out = $('live-out');
  const text = live.finals.join(' ');
  if (!text && !live.partial) {
    out.innerHTML = `<span class="placeholder">${live.on ? 'Listening…' : 'Your words appear here.'}</span>`;
  } else {
    out.textContent = text + (text && live.partial ? ' ' : '');
    if (live.partial) {
      const p = document.createElement('span');
      p.className = 'partial';
      p.textContent = live.partial;
      out.appendChild(p);
    }
  }
  out.scrollTop = out.scrollHeight;
  $('live-copy').disabled = $('live-clear').disabled = !text;
}

on('live', (m) => {
  if (m.finals.length) {
    live.finals.push(...m.finals);
    live.partial = '';
  }
  if (m.partial) live.partial = m.partial;
  if (m.done) live.partial = '';
  if (m.decode_ms) $('live-stat').textContent = `Last decode ${fmtS(m.decode_ms)} on ${BACKEND_LABEL[state.backend]} · segment ${m.segment_s.toFixed(1)} s`;
  renderLive();
});

async function startLive() {
  live.stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: true },
  });
  live.ctx = new AudioContext();
  await live.ctx.audioWorklet.addModule(new URL('./mic-worklet.js', import.meta.url));
  const src = live.ctx.createMediaStreamSource(live.stream);
  live.node = new AudioWorkletNode(live.ctx, 'mic-resampler', { numberOfInputs: 1, numberOfOutputs: 0, channelCount: 1, channelCountMode: 'explicit' });
  const ch = new MessageChannel();
  live.node.port.postMessage({ enginePort: ch.port1 }, [ch.port1]);
  await rpc('liveStart', { port: ch.port2 }, [ch.port2]);
  live.node.port.onmessage = (e) => {
    const lvl = Math.min(1, Math.sqrt(e.data.rms) * 2.2);
    $('meter-fill').style.width = `${lvl * 100}%`;
  };
  src.connect(live.node);
  live.on = true;
  live.partial = '';
  $('mic-btn').setAttribute('aria-pressed', 'true');
  $('mic-label').textContent = 'Stop';
  $('live-stat').textContent = `Listening at ${live.ctx.sampleRate / 1000} kHz, resampled to 16 kHz.`;
  renderLive();
}

async function stopLive() {
  live.on = false;
  $('mic-btn').setAttribute('aria-pressed', 'false');
  $('mic-label').textContent = 'Start listening';
  $('meter-fill').style.width = '0';
  live.stream?.getTracks().forEach((t) => t.stop());
  await live.ctx?.close();
  live.ctx = live.node = live.stream = null;
  await rpc('liveStop');
  renderLive();
}

$('mic-btn').onclick = async () => {
  try {
    live.on ? await stopLive() : await startLive();
  } catch (err) {
    $('live-stat').textContent = `Microphone: ${err.message || err}`;
    live.on = false;
    $('mic-btn').setAttribute('aria-pressed', 'false');
    $('mic-label').textContent = 'Start listening';
  }
};
$('live-clear').onclick = () => { live.finals = []; live.partial = ''; renderLive(); };
$('live-copy').onclick = () => navigator.clipboard.writeText(live.finals.join(' '));

// ---------------------------------------------------------------- file transcription
let lastResult = null;

async function decodeTo16k(file) {
  const data = await file.arrayBuffer();
  const ctx = new AudioContext();
  const buf = await ctx.decodeAudioData(data);
  await ctx.close();
  const len = Math.ceil(buf.duration * 16000);
  const off = new OfflineAudioContext(1, Math.max(1, len), 16000);
  const src = off.createBufferSource();
  src.buffer = buf;
  src.connect(off.destination);
  src.start();
  const out = await off.startRendering();
  return out.getChannelData(0).slice();
}

async function transcribeFile(file) {
  if (!state.loaded) {
    $('file-stat').textContent = 'Load the model first.';
    $('file-result').hidden = false;
    return;
  }
  $('file-result').hidden = false;
  $('file-out').textContent = '';
  $('file-stat').textContent = `Decoding ${file.name}…`;
  $('player').src = URL.createObjectURL(file);
  try {
    const audio = await decodeTo16k(file);
    const secs = audio.length / 16000;
    $('file-stat').textContent = `Transcribing ${secs.toFixed(1)} s on ${BACKEND_LABEL[state.backend]}…`;
    const r = await rpc('transcribe', { audio: audio.buffer }, [audio.buffer]);
    lastResult = { ...r, file: file.name };
    const t = r.timings_ms;
    $('file-stat').innerHTML =
      `<b>${secs.toFixed(1)} s</b> of audio in <b>${fmtS(t.total)}</b> · <b>${(secs / (t.total / 1000)).toFixed(1)}×</b> realtime on ${BACKEND_LABEL[r.backend]} ` +
      `<span class="hint">(features ${fmtS(t.features)}, encoder ${fmtS(t.encoder)}, decoder ${fmtS(t.decoder)}; ${r.windows.length} window${r.windows.length === 1 ? '' : 's'})</span>`;
    renderWords(r);
  } catch (err) {
    $('file-stat').textContent = `Failed: ${err.message || err}`;
  }
}

function renderWords(r) {
  const out = $('file-out');
  out.textContent = '';
  if (!r.words.length) {
    out.textContent = r.text || '(no speech found)';
    return;
  }
  r.words.forEach((w, i) => {
    if (i) out.append(' ');
    const s = document.createElement('span');
    s.className = 'w';
    s.textContent = w.text;
    s.dataset.start = w.start;
    s.dataset.end = w.end;
    s.title = `${w.start.toFixed(2)}–${w.end.toFixed(2)} s`;
    s.onclick = () => { const p = $('player'); p.currentTime = w.start; p.play(); };
    out.append(s);
  });
}

$('player').ontimeupdate = () => {
  const t = $('player').currentTime;
  for (const s of $('file-out').querySelectorAll('span.w')) {
    s.classList.toggle('now', t >= +s.dataset.start && t < +s.dataset.end);
  }
};

$('audio-file').onchange = (e) => e.target.files[0] && transcribeFile(e.target.files[0]);
$('sample-btn').onclick = async () => {
  const blob = await (await fetch(new URL('./samples/librispeech-1272-128104-0004.wav', import.meta.url))).blob();
  transcribeFile(new File([blob], 'librispeech-1272-128104-0004.wav', { type: 'audio/wav' }));
};
const drop = $('drop');
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = (e) => {
  e.preventDefault();
  drop.classList.remove('over');
  const f = e.dataTransfer.files[0];
  if (f) transcribeFile(f);
};

function download(name, text, type) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function srt(words) {
  const ts = (s) => {
    const ms = Math.round(s * 1000);
    const p = (n, w = 2) => String(n).padStart(w, '0');
    return `${p(Math.floor(ms / 3600000))}:${p(Math.floor(ms / 60000) % 60)}:${p(Math.floor(ms / 1000) % 60)},${p(ms % 1000, 3)}`;
  };
  const cues = [];
  let cur = [];
  for (const w of words) {
    cur.push(w);
    const dur = w.end - cur[0].start;
    if (/[.?!]$/.test(w.text) || dur > 6 || cur.length >= 14) { cues.push(cur); cur = []; }
  }
  if (cur.length) cues.push(cur);
  return cues.map((c, i) => `${i + 1}\n${ts(c[0].start)} --> ${ts(c[c.length - 1].end)}\n${c.map((w) => w.text).join(' ')}\n`).join('\n');
}

const base = () => (lastResult?.file || 'transcript').replace(/\.[^.]+$/, '');
$('file-copy').onclick = () => lastResult && navigator.clipboard.writeText(lastResult.text);
$('file-srt').onclick = () => lastResult && download(`${base()}.srt`, srt(lastResult.words), 'application/x-subrip');
$('file-json').onclick = () => lastResult && download(`${base()}.json`, JSON.stringify(lastResult, null, 1), 'application/json');
