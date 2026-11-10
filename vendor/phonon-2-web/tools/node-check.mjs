// The single-threaded wasm build (web/pkg-st) under Node: transcribe wav files and print the timings.
//
//   node tools/node-check.mjs <phonon-2.bps.tar.zst> testdata/libri0.wav ...
import { readFileSync } from 'node:fs';
import { initSync, Phonon } from '../web/pkg-st/phonon_wasm.js';

const [model, ...wavs] = process.argv.slice(2);
initSync({ module: readFileSync(new URL('../web/pkg-st/phonon_wasm_bg.wasm', import.meta.url)) });

function readWav(p) {
  const b = readFileSync(p);
  let o = 12, data = null, fmt = null;
  while (o < b.length) {
    const id = b.toString('ascii', o, o + 4), n = b.readUInt32LE(o + 4);
    if (id === 'fmt ') fmt = { ch: b.readUInt16LE(o + 10), rate: b.readUInt32LE(o + 12), bits: b.readUInt16LE(o + 22) };
    if (id === 'data') data = b.subarray(o + 8, o + 8 + n);
    o += 8 + n + (n & 1);
  }
  if (fmt.rate !== 16000 || fmt.bits !== 16) throw new Error(`${p}: need 16 kHz PCM16`);
  const s = new Float32Array(data.length / 2 / fmt.ch);
  for (let i = 0; i < s.length; i++) s[i] = data.readInt16LE(i * 2 * fmt.ch) / 32768;
  return s;
}

const t = performance.now();
const p = Phonon.load(readFileSync(model), null, null);
console.log(`load ${((performance.now() - t) / 1000).toFixed(2)}s`, p.info());
for (const w of wavs) {
  const audio = readWav(w);
  const r = JSON.parse(await p.transcribe(audio));
  const s = audio.length / 16000;
  console.log(`${w}: ${s.toFixed(1)}s audio, ${(r.timings_ms.total / 1000).toFixed(2)}s (enc ${(r.timings_ms.encoder / 1000).toFixed(2)} dec ${(r.timings_ms.decoder / 1000).toFixed(2)}) = ${(s / (r.timings_ms.total / 1000)).toFixed(2)}x | ${r.text}`);
}
