// Headless Chromium over the DevTools protocol, shared by the browser checks. Uses a throwaway profile directory.
//
//   const b = await launch({ port: 9333, binary: process.argv[3] });
//   await b.navigate(url); await b.evaluate('1 + 1'); b.close();

import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const BROWSERS = [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Start the browser and attach to its first page. Console messages from the page are echoed. */
export async function launch({ port, binary } = {}) {
  const bin = [binary, ...BROWSERS].filter(Boolean).find((c) => existsSync(c));
  if (!bin) throw new Error('no Chromium-based browser found (pass its path as an argument)');
  const profile = mkdtempSync(join(tmpdir(), 'phonon-cdp-'));
  const proc = spawn(bin, [
    '--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
    '--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--ignore-gpu-blocklist',
    '--no-first-run', '--no-default-browser-check',
    ...(process.env.EXTRA_FLAGS ? process.env.EXTRA_FLAGS.split(' ') : []), 'about:blank',
  ], { stdio: 'ignore' });
  const cleanup = () => {
    try { proc.kill('SIGKILL'); } catch { /* already gone */ }
    try { rmSync(profile, { recursive: true, force: true }); } catch { /* best effort */ }
  };
  process.on('exit', cleanup);

  let target;
  for (let i = 0; i < 50 && !target; i++) {
    await sleep(200);
    try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === 'page'); } catch { /* not up yet */ }
  }
  if (!target) throw new Error('browser did not start');

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  let id = 0;
  const waiting = new Map();
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.method === 'Runtime.consoleAPICalled') console.log('  console:', m.params.args.map((a) => a.value ?? a.description).join(' '));
    if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
  };
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; waiting.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const evaluate = async (expr) => (await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true })).result?.result?.value;
  await send('Runtime.enable');

  return {
    name: bin.split('/').pop(),
    send,
    evaluate,
    navigate: (url) => send('Page.navigate', { url }),
    /** Poll `expr` until it is truthy; false on timeout. */
    async waitFor(expr, ms = 120000, every = 200) {
      const t0 = Date.now();
      while (Date.now() - t0 < ms) {
        if (await evaluate(expr)) return true;
        await sleep(every);
      }
      return false;
    },
    close() { ws.close(); cleanup(); },
  };
}
