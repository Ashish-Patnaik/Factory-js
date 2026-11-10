#!/usr/bin/env node
// Runs web/test.html in headless Chromium and prints each backend's results against the reference.
// Needs `node serve.mjs` running with PHONON_MODEL set.
//
//   node tools/browser-check.mjs [backends] [browser binary]
//   e.g. node tools/browser-check.mjs cpu,webgpu,webgl "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

import { launch } from './cdp.mjs';

const backends = process.argv[2] || 'cpu,webgpu,webgl';
const browser = await launch({ port: 9333, binary: process.argv[3] });
console.log(`${browser.name}: ${await browser.evaluate('navigator.userAgent')}`);
await browser.navigate(`http://localhost:8787${process.env.PREFIX || ''}/test.html?backends=${backends}`);
await browser.waitFor('!!window.__done', 600000, 1000);

const out = await browser.evaluate(`({status: document.getElementById('status').textContent, log: document.getElementById('log').textContent, results: window.__results, live: window.__live})`);
console.log(out.log);
const by = {};
for (const r of out.results || []) {
  if (!r.clip) continue;
  const s = (by[r.backend] ??= { ok: 0, n: 0, ms: 0 });
  s.n++; s.ok += r.ok; s.ms += r.ms;
  if (r.clip === 'libri4') s.libri4 = `${r.ms.toFixed(0)} ms (enc ${r.enc.toFixed(0)}, dec ${r.dec.toFixed(0)})`;
}
for (const [b, s] of Object.entries(by)) console.log(`${b.padEnd(7)} ${s.ok}/${s.n} identical | all clips ${s.ms.toFixed(0)} ms | 29.4 s clip ${s.libri4}`);
if (out.live) console.log(`live    ${out.live.partials} partials, ${out.live.finals.length} finals, WER ${(100 * out.live.wer).toFixed(1)}% vs whole-clip transcripts`);
console.log(`status: ${out.status}`);
browser.close();
process.exit(0);
