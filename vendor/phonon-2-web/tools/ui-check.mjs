#!/usr/bin/env node
// Drives the app page itself (index.html) in headless Chromium: loads the model, switches through the backends,
// transcribes a file through the UI on each and saves screenshots. Needs `node serve.mjs` running with PHONON_MODEL set.
//
//   node tools/ui-check.mjs [screenshot dir] [browser binary]
//   SAMPLE=1     also press "Try a sample clip"
//   AUTO_ONLY=1  stop after the automatic backend choice

import { writeFileSync } from 'node:fs';
import { launch } from './cdp.mjs';

const shots = process.argv[2] || '.';
const browser = await launch({ port: 9334, binary: process.argv[3] });
const { evaluate, waitFor } = browser;
const text = (id) => evaluate(`document.getElementById('${id}').innerText`);
const shot = async (name) => {
  const r = await browser.send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(`${shots}/${name}.png`, Buffer.from(r.result.data, 'base64'));
};

await browser.send('Emulation.setDeviceMetricsOverride', { width: 1100, height: 1500, deviceScaleFactor: 1, mobile: false });
await browser.navigate(`http://localhost:8787${process.env.PAGE || '/?model=/model/phonon-2.bps.tar.zst'}${process.env.QUERY || ''}`);
await new Promise((r) => setTimeout(r, 1500));

await evaluate(`document.getElementById('load-btn').click()`);
await waitFor(`['ready','failed'].includes(document.getElementById('model-state').textContent)`, 600000);
await waitFor(`/^Auto:|failed/.test(document.getElementById('backend-note').textContent)`);
console.log('auto:', await text('backend-note'));
console.log('device:', await text('device'));

if (process.env.SAMPLE) {
  await evaluate(`document.getElementById('sample-btn').click()`);
  await waitFor(`/realtime|Failed/.test(document.getElementById('file-stat').innerText)`);
  console.log('sample:', await text('file-stat'));
  console.log('  ', (await text('file-out')).slice(0, 120));
  console.log('footer:', await evaluate(`document.querySelector('footer').innerText.slice(0, 60)`));
  await shot('space');
}
if (process.env.AUTO_ONLY) { browser.close(); process.exit(0); }

console.log('model:', await text('model-state'), '/', (await text('facts')).split('\n').join(' '));
for (const be of ['cpu', 'webgpu', 'webgl']) {
  if (be !== 'cpu') {
    await evaluate(`document.getElementById('be-${be}').click()`);
    await waitFor(`!/Starting/.test(document.getElementById('backend-note').textContent)`);
    console.log(`${be}:`, await text('backend-note'));
  }
  await evaluate(`(async () => {
    const blob = await (await fetch('/testdata/libri2.wav')).blob();
    const dt = new DataTransfer(); dt.items.add(new File([blob], 'libri2.wav', { type: 'audio/wav' }));
    const inp = document.getElementById('audio-file'); inp.files = dt.files;
    document.getElementById('file-stat').innerText = '';
    inp.dispatchEvent(new Event('change'));
  })()`);
  await waitFor(`/realtime|Failed/.test(document.getElementById('file-stat').innerText)`);
  console.log(`  ${await text('file-stat')}`);
  console.log(`  words: ${await evaluate(`document.querySelectorAll('#file-out span.w').length`)} | ${(await text('file-out')).slice(0, 90)}`);
}
await shot('ui-after');
browser.close();
process.exit(0);
