// scrape.js - let the AI "visit" the page: extract readable text and links with Obscura (CDP)
// via Playwright; Chromium as fallback.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright';
import { ROOT, sleep, ev } from './utils.js';

export const EXTRACT_JS = `() => ({
  title: document.title || '',
  desc: ((document.querySelector('meta[name=description],meta[property="og:description"]') || {}).content) || '',
  text: document.body ? (document.body.innerText || document.body.textContent || '') : '',
})`;

export const EXTRACT_LINKS_JS = `() => {
  const bad = /(login|signup|signin|register|account|privacy|terms|about|contact|advertise|faq|security|pricing|careers|jobs|sponsor|settings|vote\\?|hide\\?|fave\\?|submit|bookmark)/i;
  const out = [], seen = new Set();
  for (const a of document.querySelectorAll('a[href]')) {
    const u = a.href.split('#')[0].trim();
    const t = (a.innerText || a.getAttribute('aria-label') || '').replace(/\\s+/g, ' ').trim();
    if (!u.startsWith('http') || seen.has(u) || t.length < 12 || t.length > 220 || bad.test(u)) continue;
    seen.add(u);
    out.push({ url: u, title: t });
  }
  return out.slice(0, 100);
}`;

export function findObscura(explicit = null) {
  // Locate OS-appropriate obscura binary: .exe on Windows, 'obscura' on Linux/macOS.
  const isWin = os.platform() === 'win32';
  const names = isWin ? ['obscura.exe', 'obscura.exe.exe'] : ['obscura'];

  const ready = (p) => {
    const full = path.resolve(p);
    if (!isWin) {
      // Ensure the Linux binary and its worker have executable permissions (chmod +x)
      const folder = path.dirname(full);
      for (const fn of ['obscura', 'obscura-worker']) {
        const fp = path.join(folder, fn);
        try {
          if (fs.existsSync(fp) && !(fs.statSync(fp).mode & 0o111)) fs.chmodSync(fp, 0o755);
        } catch { /* best effort */ }
      }
    }
    return full;
  };

  const isFile = (p) => { try { return fs.statSync(p).isFile(); } catch { return false; } };

  if (explicit) {
    if (fs.existsSync(explicit) && fs.statSync(explicit).isDirectory()) {
      for (const n of names) {
        const cand = path.join(explicit, n);
        if (isFile(cand)) return ready(cand);
      }
    } else if (isFile(explicit)) return ready(explicit);
  }

  const envP = process.env.OBSCURA_PATH;
  if (envP && isFile(envP)) return ready(envP);

  const roots = [ROOT, process.cwd()];
  for (const r of roots) {
    for (const d of ['obscura', '.']) {
      const base = path.join(r, d);
      for (const n of names) {
        const cand = path.join(base, n);
        if (isFile(cand)) return ready(cand);
      }
      // Also walk one level deeper in case zip/tar extracted to a subfolder inside obscura/
      if (d === 'obscura' && fs.existsSync(base) && fs.statSync(base).isDirectory()) {
        const walk = (dir, depth) => {
          if (depth > 2) return null;
          for (const f of fs.readdirSync(dir)) {
            const fp = path.join(dir, f);
            let st;
            try { st = fs.statSync(fp); } catch { continue; }
            if (st.isDirectory()) {
              const found = walk(fp, depth + 1);
              if (found) return found;
            } else if (names.includes(f.toLowerCase())) return ready(fp);
          }
          return null;
        };
        const found = walk(base, 0);
        if (found) return found;
      }
    }
  }

  const whichP = process.platform === 'win32'
    ? null // shutil.which equivalent; rely on PATH resolution by spawn
    : '/usr/bin/obscura';
  if (!isWin && whichP && isFile(whichP)) return ready(whichP);
  return null;
}

function clean(d, limit = 12000) {
  const lines = [];
  const seen = new Set();
  for (let ln of String(d.text ?? '').split(/[\r\n]+/)) {
    ln = ln.replace(/\s+/g, ' ').trim();
    if (ln.length > 1 && !seen.has(ln)) {
      seen.add(ln);
      lines.push(ln);
    }
  }
  const head = `Title: ${d.title}\n` + (d.desc ? `Description: ${d.desc}\n` : '');
  return (head + lines.join('\n')).slice(0, limit);
}

async function visit(page, cfg, url = null) {
  await page.goto(url ?? cfg.url, { waitUntil: 'load', timeout: cfg.timeout * 1000 });
  await page.waitForTimeout(Math.max(cfg.wait, 1.5) * 1000);
  return clean(await ev(page, EXTRACT_JS));
}

async function visitLinks(page, cfg, url) {
  await page.goto(url, { waitUntil: 'load', timeout: cfg.timeout * 1000 });
  await page.waitForTimeout(Math.max(cfg.wait, 1.5) * 1000);
  return ev(page, EXTRACT_LINKS_JS);
}

export function startObscura(cfg, exe) {
  // Start `obscura serve` on a free port with cwd set to obscura's directory so it finds obscura-worker(.exe).
  return new Promise(async (resolve, reject) => {
    const port = await new Promise((res, rej) => {
      const sk = net.createServer();
      sk.listen(0, '127.0.0.1', () => { const p = sk.address().port; sk.close(() => res(p)); });
      sk.on('error', rej);
    });
    const exeDir = path.dirname(path.resolve(exe));
    const env = { ...process.env, PATH: exeDir + path.delimiter + (process.env.PATH ?? '') };
    const args = [exe, 'serve', '--port', String(port)];
    if (cfg.private_net) args.push('--allow-private-network');
    const proc = spawn(args[0], args.slice(1), { cwd: exeDir, env, stdio: ['ignore', 'ignore', 'ignore'] });
    const deadline = Date.now() + 15000;
    const tryConn = async () => {
      if (proc.exitCode !== null) return reject(new Error(`Obscura exited during startup (code ${proc.exitCode})`));
      const ok = await new Promise((res) => {
        const c = net.connect(port, '127.0.0.1');
        c.on('connect', () => { c.destroy(); res(true); });
        c.on('error', () => res(false));
        setTimeout(() => { c.destroy(); res(false); }, 300);
      });
      if (ok) return resolve({ proc, port });
      if (Date.now() > deadline) { stopProcess(proc); return reject(new Error('Obscura did not open its CDP port in time')); }
      setTimeout(tryConn, 100);
    };
    tryConn();
  });
}

export function stopProcess(proc) {
  if (!proc || proc.exitCode !== null) return;
  proc.kill();
  const t = setTimeout(() => { try { proc.kill('SIGKILL'); } catch { /* gone */ } }, 5000);
  proc.once('exit', () => clearTimeout(t));
}

async function withObscura(cfg, exe, fn) {
  const { proc, port } = await startObscura(cfg, exe);
  let browser;
  try {
    browser = await chromium.connectOverCDP(`ws://127.0.0.1:${port}`);
    const ctx = browser.contexts[0] ?? await browser.newContext();
    const page = ctx.newPage ? await ctx.newPage() : await browser.newPage();
    return await fn(page);
  } finally {
    try { await browser?.close(); } catch { /* already gone */ }
    stopProcess(proc);
  }
}

async function withChromium(cfg, fn) {
  const browser = await chromium.launch({ headless: true, ...(cfg.channel ? { channel: cfg.channel } : {}) });
  try {
    return await fn(await browser.newPage());
  } finally {
    try { await browser.close(); } catch { /* ignore */ }
  }
}

export async function pageText(cfg, url = null) {
  const target = url ?? cfg.url;
  if (cfg.scraper === 'obscura') {
    const exe = findObscura(cfg.obscura_path);
    if (!exe) console.log('[scrape] Obscura not found; using Chromium');
    else {
      try {
        const text = await withObscura(cfg, exe, (pg) => visit(pg, cfg, target));
        if (text.length >= 300) {
          console.log(`[scrape] Obscura read ${text.length} chars from ${target}`);
          return text;
        }
        console.log(`[scrape] Obscura returned only ${text.length} chars; retrying with Chromium`);
      } catch (e) {
        console.log(`[scrape] Obscura failed (${firstLine(e)}); retrying with Chromium`);
      }
    }
  }
  let text;
  try {
    text = await withChromium(cfg, (pg) => visit(pg, cfg, target));
  } catch (e) {
    throw new Error(`could not read ${target}: ${firstLine(e)}`);
  }
  console.log(`[scrape] Chromium read ${text.length} chars from ${target}`);
  return text;
}

export async function pageLinks(cfg, hubUrl) {
  // Extract candidate nested links from a hub URL (GitHub Trending, HN, Dev.to, blogs, etc.).
  let links = [];
  if (cfg.scraper === 'obscura') {
    const exe = findObscura(cfg.obscura_path);
    if (exe) {
      try {
        links = await withObscura(cfg, exe, (pg) => visitLinks(pg, cfg, hubUrl));
      } catch { links = []; }
    }
  }
  if (links.length < 3) {
    try {
      links = await withChromium(cfg, (pg) => visitLinks(pg, cfg, hubUrl));
    } catch (e) {
      console.log(`[scrape] could not extract links from ${hubUrl}: ${firstLine(e)}`);
      return [];
    }
  }
  const hubHost = new URL(hubUrl).hostname;
  const filtered = [];
  for (const item of links) {
    const u = item.url;
    let p;
    try { p = new URL(u); } catch { continue; }
    // On GitHub trending, only keep actual repo links (/owner/repo)
    if (hubHost.includes('github.com') && p.hostname === 'github.com') {
      const parts = p.pathname.split('/').filter(Boolean);
      if (parts.length !== 2 || ['trending', 'topics', 'sponsors', 'collections', 'features', 'orgs'].includes(parts[0])) continue;
    }
    filtered.push({ url: u, title: item.title, source_hub: hubUrl });
  }
  return filtered;
}

function firstLine(e) {
  return String(e?.message ?? e).split('\n')[0];
}
