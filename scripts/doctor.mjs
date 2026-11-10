#!/usr/bin/env node
// doctor.mjs - verify the factory environment: Node version, browsers, obscura, TTS/ASR
// engines, model caches and API keys. Run `npm run doctor` before your first render.
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';
import { ROOT } from '../src/utils.js';
import { findObscura } from '../src/scrape.js';
import { asrAvailable } from '../src/asr.js';

const ok = (m) => console.log(`  \u2713 ${m}`);
const warn = (m) => console.log(`  ! ${m}`);
const bad = (m) => console.log(`  \u2717 ${m}`);

async function main() {
  console.log('AI Content Factory (Node.js) - environment check\n' + '-'.repeat(50));

  // Node
  const [maj, min] = process.versions.node.split('.').map(Number);
  if (maj > 20 || (maj === 20 && min >= 11)) ok(`Node.js ${process.versions.node}`);
  else bad(`Node.js ${process.versions.node} (need >= 20.11)`);

  // OS / arch
  ok(`${os.platform()} ${os.arch()} / ${os.cpus().length} cores / ${(os.totalmem() / 1e9).toFixed(0)} GB RAM`);

  // Encoder browser + codec capability
  let encoderBrowser = null;
  const stub = path.join(ROOT, '.cache', 'encoder.html');
  fs.mkdirSync(path.dirname(stub), { recursive: true });
  fs.writeFileSync(stub, '<!doctype html><title>x</title>');
  const withTimeout = (p, ms, label) => Promise.race([
    p, new Promise((_, rej) => setTimeout(() => rej(new Error(`${label} timed out`)), ms)),
  ]);
  for (const channel of ['chrome', 'msedge', null]) {
    try {
      const b = await withTimeout(chromium.launch({ headless: true, args: ['--no-first-run'], ...(channel ? { channel } : {}) }), 20000, channel ?? 'chromium');
      const p = await b.newPage();
      await p.goto(pathToFileURL(stub).href); // secure context so WebCodecs is exposed
      const caps = await p.evaluate(async () => ({
        avc: (await VideoEncoder.isConfigSupported({ codec: 'avc1.640028', width: 1080, height: 1920, framerate: 30, bitrate: 8e6 })).supported,
        aac: (await AudioEncoder.isConfigSupported({ codec: 'mp4a.40.2', sampleRate: 44100, numberOfChannels: 1, bitrate: 128e3 })).supported,
      }));
      await b.close();
      encoderBrowser = { name: channel ?? 'bundled chromium', ...caps };
      break;
    } catch { /* try next */ }
  }
  if (encoderBrowser) {
    ok(`encoder browser: ${encoderBrowser.name} - H.264 ${encoderBrowser.avc ? 'ok' : 'MISSING'}, ` +
      `AAC ${encoderBrowser.aac ? 'native' : 'via WASM extension (automatic)'}`);
    if (!encoderBrowser.avc) bad('this browser cannot encode H.264 - install Chrome or Edge');
  } else {
    bad('no browser for the encoder. Run: npx playwright install chromium (or install Chrome/Edge)');
  }

  // Playwright chromium for page capture
  try {
    const p = await chromium.executablePath({});
    if (p && fs.existsSync(p)) ok('page-capture Chromium installed');
    else warn('page-capture Chromium missing. Run: npx playwright install chromium');
  } catch {
    warn('playwright browsers missing. Run: npx playwright install chromium');
  }

  // Obscura
  const obscura = findObscura(null);
  if (obscura) ok(`Obscura: ${obscura}`);
  else warn('Obscura not found (optional - for sites that block headless Chrome). ' +
    'Put obscura.exe + obscura-worker.exe in the obscura/ folder. https://github.com/h4ckf0r0day/obscura/releases');

  // TTS
  try {
    const pkg = path.join(ROOT, 'node_modules', 'phonon-tts', 'wasm', 'phonon_tts_bg.wasm');
    if (fs.existsSync(pkg)) ok('phonon-tts WASM engine present (model downloads once to .cache/tts)');
    else bad('phonon-tts not installed - run: npm install');
  } catch { bad('phonon-tts not installed'); }

  // ASR
  const asr = asrAvailable();
  if (asr) ok(`phonon-2-web ASR engine: ${path.relative(ROOT, asr)} (model downloads once to .cache/asr)`);
  else warn('phonon-2-web ASR engine missing - captions need it. Run: npm run setup:asr');

  // API keys
  if (process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY) ok('GEMINI_API_KEY set');
  else warn('GEMINI_API_KEY not set (needed for --narrate / --auto / --long AI scripts)');

  // Publish credentials
  if (fs.existsSync(path.join(ROOT, 'client_secrets.json'))) ok('client_secrets.json found (YouTube upload)');
  else warn('client_secrets.json missing (needed only for --publish youtube)');
  const igOk = process.env.IG_ACCESS_TOKEN && process.env.IG_USER_ID &&
    process.env.IG_ACCESS_TOKEN !== 'YOUR_ACCESS_TOKEN_HERE';
  if (igOk) ok('IG_ACCESS_TOKEN / IG_USER_ID set');
  else warn('IG_ACCESS_TOKEN / IG_USER_ID not set (needed only for --publish instagram)');

  // Fonts
  if (fs.existsSync(path.join(ROOT, 'fonts', 'Anton-Regular.ttf'))) ok('Anton caption font bundled');
  else warn('fonts/Anton-Regular.ttf missing (auto-downloads on first captioned render)');

  console.log('\nQuick start:');
  console.log('  node main.js --auto                      # trending topic -> 9:16 video + metadata');
  console.log('  node main.js --auto --publish all        # ...and upload to YouTube Shorts + Instagram Reels');
  console.log('  node main.js https://example.com demo --mobile --narrate   # single URL with AI voice-over');
}

main();
