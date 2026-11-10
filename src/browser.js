// browser.js - Playwright Chromium (or installed Chrome/Edge) plus Obscura-over-CDP, with the
// popup killer, dismiss sweep and the deterministic virtual-time shim used by frame capture.
import { chromium } from 'playwright';
import { findObscura, startObscura, stopProcess } from './scrape.js';
import { ev } from './utils.js';

export const MOBILE_UA =
  'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 ' +
  '(KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36';

// GPU is used when available; otherwise Chromium falls back to software (SwiftShader) WebGL.
export const ARGS = [
  '--ignore-gpu-blocklist', '--enable-unsafe-swiftshader', '--enable-gpu-rasterization',
  '--autoplay-policy=no-user-gesture-required', '--mute-audio', '--hide-scrollbars',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
  '--disable-backgrounding-occluded-windows',
];

export const SETTLE_JS = `() => {
  const t = new Promise((r) => setTimeout(r, 6000));
  const imgs = [...document.images].filter((i) => !i.complete).map((i) => new Promise((r) => { i.onload = i.onerror = r; }));
  return Promise.race([Promise.all([document.fonts ? document.fonts.ready : 0, ...imgs]), t]);
}`;

export const MEDIA_JS = `() => {
  const v = [...document.querySelectorAll('video')];
  v.forEach((x) => { x.muted = true; try { const p = x.play(); p && p.catch && p.catch(() => {}); } catch (e) {} });
  let gl = false; try { const c = document.createElement('canvas'); gl = !!(c.getContext('webgl2') || c.getContext('webgl')); } catch (e) {}
  return { videos: v.length, canvases: document.querySelectorAll('canvas').length, webgl: gl };
}`;

// ---------------------------------------------------------------------------
// Popup suppression (dev.to & friends): promo modals, cookie consent banners and newsletter
// dialogs would be burned into the recorded video. Two defenses:
//   1. POPUP_KILLER_JS is injected at document start and removes popup-like elements the
//      instant they appear - including delayed popups that would pop up mid-recording.
//   2. dismissPopups() runs after the page settles: presses Escape, clicks visible
//      "Okay / Got it / Accept / Dismiss" buttons, then sweeps any leftover overlay.
// ---------------------------------------------------------------------------
export const POPUP_KILLER_JS = `(()=>{if(window.__popupKiller)return;window.__popupKiller=1;
const RX=/modal|popup|lightbox|interstitial|onetrust|cookie|consent|gdpr|klaviyo|osano|truste|quantcast|paywall|crayons-modal|snackbar|toast|announcement|hello-?bar|newsletter|subscribe-?modal|sign-?up|welcome-?modal|promo/i;
const MAIN='main,article,[role="main"],#main-content';
const name=e=>((e.id||'')+' '+(e.getAttribute('class')||'')+' '+(e.getAttribute('aria-label')||'')+' '+(e.getAttribute('data-testid')||''));
let mainEl=null,mainAt=0;
const getMain=()=>{const n=Date.now();if(!mainEl||n-mainAt>1000){try{mainEl=document.querySelector(MAIN)}catch(e){}mainAt=n}return mainEl};
const looksPopup=e=>{try{
 if(!e||e.nodeType!==1||e===document.body||e===document.documentElement)return false;
 const m=getMain();if(m&&(m===e||m.contains(e)))return false;
 if(RX.test(name(e))){const s=getComputedStyle(e);
  if(s.display!=='none'&&s.visibility!=='hidden'&&+s.opacity!==0)return true}
 if(e.parentElement===document.body){const s=getComputedStyle(e);
  if((s.position==='fixed'||s.position==='sticky')&&+s.zIndex>=999&&s.display!=='none'){
   const r=e.getBoundingClientRect(),W=innerWidth||1,H=innerHeight||1;
   if(r.width>=W*.7&&r.height>=H*.7)return true}}
}catch(err){}return false};
const check=e=>{if(looksPopup(e)){try{e.remove()}catch(err){}return true}return false};
const sweep=()=>{try{
 document.querySelectorAll('dialog[open]').forEach(d=>{try{d.close()}catch(e){}});
 document.querySelectorAll('body *').forEach(e=>{if(looksPopup(e))try{e.remove()}catch(err){}})
}catch(err){}};
window.__popupSweep=sweep;
new MutationObserver(ms=>{for(const m of ms){
 if(m.type==='childList'){for(const e of m.addedNodes){if(e.nodeType!==1)continue;
  if(check(e))continue;if(e.children)for(const c of e.children)check(c)}}
 else if(m.type==='attributes'&&m.target&&m.target.parentElement===document.body)check(m.target)
}}).observe(document,{childList:true,subtree:true,attributes:true,attributeFilter:['class','style']});
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',()=>setTimeout(sweep,300));
else setTimeout(sweep,300)})()`;

// Clicks up to 8 visible dismiss-style buttons ("Okay", "Got it", "Accept", "Close"...).
export const DISMISS_JS = `()=>{const RX=/^(okay|ok|got\\\\s?it|accept( all)?|i\\\\s?agree|agree|allow( all)?|dismiss|close|no,?\\\\s?thanks?|maybe\\\\s?later|later|continue|skip|understood|fine|sure|x|\u00d7|\u2715|\u2716)$|^(okay|ok|got it|close|dismiss|accept)\\\\b/i;
let clicked=0;const els=[...document.querySelectorAll('button,[role="button"],input[type="button"],input[type="submit"]')];
for(const e of els){if(clicked>=8)break;
 try{const s=getComputedStyle(e);if(s.display==='none'||s.visibility==='hidden')continue;
 const t=((e.innerText||e.value||e.getAttribute('aria-label')||'')+'').trim().toLowerCase();
 if(!t||!RX.test(t))continue;const r=e.getBoundingClientRect();if(r.width<8||r.height<8)continue;
 e.click();clicked++}catch(err){}}
return clicked}`;

// Makes rendering deterministic/synchronous so each virtual-time frame is fully painted.
export const DETERMINISTIC_ARGS = [
  '--run-all-compositor-stages-before-draw', '--disable-new-content-rendering-timeout',
  '--disable-threaded-animations', '--disable-threaded-scrolling',
  '--disable-checker-imaging', '--disable-image-animation-resync',
];

// Injected before page scripts. Passes through to real rAF until activated, then queues rAF
// callbacks so the recorder can fire them once per video frame, and seeks every CSS/WAAPI
// animation to the exact frame time.
export const SHIM_JS = `(()=>{if(window.__vt)return;
const real=window.requestAnimationFrame.bind(window),q=new Map(),reg=new WeakMap();let id=0,on=false;
window.requestAnimationFrame=cb=>on?(q.set(++id,cb),id):real(cb);
window.cancelAnimationFrame=i=>q.delete(i);
const frame=()=>new Promise(r=>real(()=>r()));
window.__vt={activate(){on=true},
async step(y,elapsed,ts){
 window.scrollTo({top:y,behavior:'instant'});
 await frame();
 for(const a of document.getAnimations()){
  if(a.timeline&&typeof DocumentTimeline!=='undefined'&&!(a.timeline instanceof DocumentTimeline))continue;
  let r=reg.get(a);if(!r){r={base:a.currentTime||0,at:elapsed};reg.set(a,r);a.pause()}
  a.currentTime=r.base+(elapsed-r.at)}
 const cbs=[...q.values()];q.clear();
 for(const cb of cbs){try{cb(ts)}catch(e){}}
 await frame()}}})()`;

function contextOpts(cfg, vw, vh) {
  if (!cfg.mobile) {
    return { viewport: { width: vw, height: vh }, deviceScaleFactor: 1 };
  }
  // Scale up so the captured frame matches (or exceeds) the output video resolution natively.
  const dpr = Math.max(1, Math.ceil(Math.max(cfg.width / vw, cfg.height / vh)));
  return {
    viewport: { width: vw, height: vh },
    screen: { width: vw, height: vh },
    deviceScaleFactor: dpr,
    isMobile: true,
    hasTouch: true,
    userAgent: MOBILE_UA,
  };
}

export class Browser {
  /** Playwright Chromium (or installed Chrome/Edge via --channel). Always cleaned up. */
  constructor(cfg) {
    this.cfg = cfg;
    this.browser = null;
  }

  async open() {
    const args = [...ARGS];
    if (this.cfg.mode === 'virtual') args.push(...DETERMINISTIC_ARGS);
    const opts = { headless: !this.cfg.headed, args };
    if (this.cfg.channel) opts.channel = this.cfg.channel;
    try {
      this.browser = await chromium.launch(opts);
    } catch (e) {
      const msg = String(e?.message ?? e);
      if (/Executable doesn't exist|not found/i.test(msg)) {
        console.error('error: Chromium is not installed. Run: npx playwright install chromium   ' +
          '(or use --channel chrome / msedge to use an installed browser)');
        console.log('error: Chromium is not installed. Run: npx playwright install chromium');
        process.exit(1);
      }
      throw new Error(`could not launch browser: ${msg.split('\n')[0]}`);
    }
    return this;
  }

  async close() {
    try { await this.browser?.close(); } catch { /* already gone */ }
  }

  async newPage(vw, vh) {
    const ctx = await this.browser.newContext(contextOpts(this.cfg, vw, vh));
    try { await ctx.addInitScript(POPUP_KILLER_JS); } catch { /* ignore */ }
    if (this.cfg.mode === 'virtual') {
      try { await ctx.addInitScript(SHIM_JS); } catch { /* ignore */ }
    }
    const page = await ctx.newPage();
    page.setDefaultTimeout(this.cfg.timeout * 1000);
    return page;
  }

  async load(page) {
    const url = this.cfg.url;
    try {
      const resp = await page.goto(url, { waitUntil: 'load', timeout: this.cfg.timeout * 1000 });
      if (resp && resp.status() >= 400) console.log(`warning: server returned HTTP ${resp.status()}`);
    } catch (e) {
      if (String(e).includes('Timeout')) {
        console.log('warning: page did not finish loading in time; continuing with what rendered');
      } else {
        throw new Error(`failed to load ${url}: ${String(e?.message ?? e).split('\n')[0]}`);
      }
    }
    try { await ev(page, SETTLE_JS); } catch { /* ignore */ }
    await page.waitForTimeout(this.cfg.wait * 1000);
    await Browser.dismissPopups(page);
    await Browser.prepareMedia(page);
  }

  static async dismissPopups(page) {
    // Best-effort removal of anything popup-like left on the page (cookie banners,
    // promo modals, newsletter dialogs...) so it is never burned into the video.
    try { await page.keyboard.press('Escape'); } catch { /* ignore */ }
    try {
      const n = await ev(page, DISMISS_JS);
      await page.waitForTimeout(400);
      await ev(page, '() => window.__popupSweep && window.__popupSweep()');
      if (n) console.log(`note: dismissed ${n} popup button(s) (Escape + click) before recording.`);
    } catch { /* ignore */ }
  }

  static async prepareMedia(page) {
    let info;
    try { info = await ev(page, MEDIA_JS); } catch { return; }
    if (!info) return;
    if (info.videos) console.log(`note: ${info.videos} <video> element(s) found and started (muted).`);
    if (info.canvases && !info.webgl) {
      console.log('warning: page has <canvas> but WebGL is unavailable here; 3D scenes may render blank. ' +
        'Try --headed or --channel chrome.');
    }
  }
}

export class ObscuraBrowser extends Browser {
  /** Obscura (lightweight Rust browser, native rendering) driven over CDP. Not detected as
   * headless Chrome, so it can capture sites that block Chromium. No virtual clock: use the
   * stepped or realtime capture modes. */
  async open() {
    const exe = findObscura(this.cfg.obscura_path);
    if (!exe) {
      console.error('error: Obscura not found. Put obscura.exe + obscura-worker.exe in an `obscura` folder next to main.js, ' +
        'set OBSCURA_PATH, or pass --obscura-path. Download: https://github.com/h4ckf0r0day/obscura/releases');
      console.log('error: Obscura not found (obscura/ folder, OBSCURA_PATH or --obscura-path)');
      process.exit(1);
    }
    const { proc, port } = await startObscura(this.cfg, exe);
    this.proc = proc;
    this.port = port;
    try {
      this.browser = await chromium.connectOverCDP(`ws://127.0.0.1:${this.port}`);
    } catch (e) {
      await this.close();
      throw new Error(`could not connect to Obscura: ${String(e?.message ?? e).split('\n')[0]}`);
    }
    return this;
  }

  async close() {
    try { await this.browser?.close(); } catch { /* ignore */ }
    stopProcess(this.proc);
  }

  async newPage(vw, vh) {
    const opts = contextOpts(this.cfg, vw, vh);
    let page;
    try {
      const ctx = await this.browser.newContext(opts);
      try { await ctx.addInitScript(POPUP_KILLER_JS); } catch { /* ignore */ }
      page = await ctx.newPage();
    } catch {
      const ctx = this.browser.contexts[0] ?? await this.browser.newContext();
      try { await ctx.addInitScript(POPUP_KILLER_JS); } catch { /* ignore */ }
      page = await ctx.newPage();
      try { await page.setViewportSize({ width: vw, height: vh }); } catch { /* ignore */ }
      if (this.cfg.mobile) {
        try {
          const cdp = await page.context().newCDPSession(page);
          await cdp.send('Emulation.setUserAgentOverride', { userAgent: MOBILE_UA, platform: 'Android' });
          await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: true });
          await cdp.send('Emulation.setDeviceMetricsOverride', {
            width: vw, height: vh, deviceScaleFactor: opts.deviceScaleFactor, mobile: true,
          });
        } catch { /* best effort */ }
      }
    }
    page.setDefaultTimeout(this.cfg.timeout * 1000);
    return page;
  }
}
