// Runs every testdata clip through each available backend in the engine worker and compares the transcript with the
// fp32 reference (tools/make_reference.py). Results are also left on window.__results for automation.
const params = new URLSearchParams(location.search);
const backends = (params.get('backends') || 'cpu,webgpu,webgl').split(',');
const model = params.get('model') || '/model/phonon-2.bps.tar.zst';
const $ = (id) => document.getElementById(id);
const log = (s) => { $('log').textContent += s + '\n'; };

const worker = new Worker(new URL('./phonon-worker.js', import.meta.url), { type: 'module' });
let next = 1;
const pending = new Map();
const liveEvents = [];
worker.onmessage = (e) => {
  const m = e.data;
  if (m.event === 'live') liveEvents.push(m);
  if (m.event) return;
  const p = pending.get(m.id);
  pending.delete(m.id);
  m.ok ? p.resolve(m.result) : p.reject(new Error(m.error));
};
const rpc = (cmd, args = {}, transfer = []) => new Promise((resolve, reject) => {
  const id = next++;
  pending.set(id, { resolve, reject });
  worker.postMessage({ id, cmd, ...args }, transfer);
});

function wav16k(buf) {
  const v = new DataView(buf);
  let o = 12, data = null, ch = 1;
  while (o < v.byteLength) {
    const id = String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3));
    const n = v.getUint32(o + 4, true);
    if (id === 'fmt ') ch = v.getUint16(o + 10, true);
    if (id === 'data') data = [o + 8, n];
    o += 8 + n + (n & 1);
  }
  const s = new Float32Array(data[1] / 2 / ch);
  for (let i = 0; i < s.length; i++) s[i] = v.getInt16(data[0] + i * 2 * ch, true) / 32768;
  return s;
}

function wordErrorRate(ref, hyp) {
  const norm = (t) => t.toLowerCase().replace(/[^a-z0-9' ]/g, ' ').split(/\s+/).filter(Boolean);
  const a = norm(ref), b = norm(hyp);
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++)
    for (let j = 1; j <= b.length; j++)
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
  return d[a.length][b.length] / Math.max(1, a.length);
}

const results = [];
window.__results = results;
(async () => {
  try {
    const boot = await rpc('boot');
    log(`boot: ${JSON.stringify(boot)}`);
    const ref = await (await fetch('/testdata/ref/reference.json')).json();
    const bytes = await (await fetch(model)).arrayBuffer();
    const t0 = performance.now();
    const ld = await rpc('load', { bytes }, [bytes]);
    log(`load: ${(performance.now() - t0).toFixed(0)} ms, ${JSON.stringify(ld.info)}`);
    const clips = Object.keys(ref).sort();
    const audio = {};
    for (const c of clips) audio[c] = wav16k(await (await fetch(`/testdata/${c}.wav`)).arrayBuffer());
    for (const be of backends) {
      try {
        const b = await rpc('backend', { name: be });
        log(`${be}: ${b.adapter || 'cpu'} (${b.ms.toFixed(0)} ms to start)`);
      } catch (err) {
        log(`${be}: unavailable: ${err.message}`);
        results.push({ backend: be, error: err.message });
        continue;
      }
      // warm-up pass so the timings below are steady state
      await rpc('transcribe', { audio: audio[clips[0]].slice() });
      for (const c of clips) {
        const a = audio[c].slice();
        const r = await rpc('transcribe', { audio: a.buffer }, [a.buffer]);
        const ok = r.text === ref[c].text;
        const secs = audio[c].length / 16000;
        results.push({ backend: be, clip: c, ok, ms: r.timings_ms.total, enc: r.timings_ms.encoder, dec: r.timings_ms.decoder, text: r.text });
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${be}</td><td>${c}</td><td>${secs.toFixed(1)} s</td><td>${r.timings_ms.total.toFixed(0)} ms</td>` +
          `<td>${(secs / (r.timings_ms.total / 1000)).toFixed(1)}×</td><td class="${ok ? 'pass' : 'fail'}">${ok ? 'identical' : 'differs'}</td>`;
        $('tbl').tBodies[0].append(tr);
        if (!ok) log(`${be}/${c}\n  ours: ${r.text}\n  ref:  ${ref[c].text}`);
      }
    }
    if (params.get('live') !== '0') {
      // live dictation, simulated: two clips with a second of silence after each, fed in 50 ms blocks at 4x speed
      const parts = [audio.libri1, new Float32Array(16000), audio.libri0, new Float32Array(16000)];
      const total = parts.reduce((n, p) => n + p.length, 0);
      const sig = new Float32Array(total);
      let o = 0;
      for (const p of parts) { sig.set(p, o); o += p.length; }
      const ch = new MessageChannel();
      await rpc('liveStart', { port: ch.port2 }, [ch.port2]);
      const t1 = performance.now();
      for (let i = 0; i < sig.length; i += 800) {
        const b = sig.slice(i, i + 800);
        ch.port1.postMessage(b, [b.buffer]);
        await new Promise((r) => setTimeout(r, 12));
      }
      await rpc('liveStop');
      const finals = liveEvents.flatMap((e) => e.finals);
      const partials = liveEvents.filter((e) => e.partial).length;
      const want = `${ref.libri1.text} ${ref.libri0.text}`;
      const got = finals.join(' ');
      const wer = wordErrorRate(want, got);
      window.__live = { finals, partials, wer };
      log(`live (${backends[backends.length - 1]}): ${partials} partial updates, ${finals.length} finals\n  finals: ${JSON.stringify(finals)}\n  whole-clip reference: ${want}`);
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>live</td><td>libri1 + libri0</td><td>${(sig.length / 16000).toFixed(1)} s</td><td>${(performance.now() - t1).toFixed(0)} ms</td><td>4× feed</td><td class="${wer < 0.1 ? 'pass' : 'fail'}">WER ${(100 * wer).toFixed(1)}% vs whole clips</td>`;
      $('tbl').tBodies[0].append(tr);
    }
    const done = results.filter((r) => r.clip);
    const pass = done.filter((r) => r.ok).length;
    $('status').textContent = `${pass}/${done.length} identical`;
    $('status').className = `pill ${pass === done.length ? 'ok' : 'err'}`;
    window.__done = true;
  } catch (err) {
    log(`FAILED: ${err.message || err}`);
    $('status').textContent = 'failed';
    $('status').className = 'pill err';
    window.__done = true;
  }
})();
