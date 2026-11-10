// scroll.js - scroll planning (pure math) and page-side helper JS snippets.
import { ev } from './utils.js';

export const HEIGHT_JS = '() => Math.max(document.documentElement.scrollHeight, document.body ? document.body.scrollHeight : 0)';
export const SCROLL_JS = "y => window.scrollTo({ top: y, behavior: 'instant' })";

function ease(t) { // ease-in-out cubic
  return t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2;
}

export async function maxScroll(page) {
  const h = await ev(page, HEIGHT_JS);
  const ih = await ev(page, '() => window.innerHeight');
  return Math.max(0, h - ih);
}

export async function warmUp(page, step, cap = 150000) {
  // Scroll through once so lazy images/content load, wait for fonts, return to top.
  await ev(page, "() => { const s = document.createElement('style'); s.textContent = 'html{scroll-behavior:auto!important}'; document.head.appendChild(s); }");
  let y = 0;
  let h = await ev(page, HEIGHT_JS);
  while (y < Math.min(h, cap)) { // height re-read each step: handles growing pages
    await ev(page, SCROLL_JS, y);
    await page.waitForTimeout(120);
    y += step;
    h = await ev(page, HEIGHT_JS);
  }
  try {
    await ev(page, "() => Promise.race([document.fonts ? document.fonts.ready : 0, new Promise((r) => setTimeout(r, 4000))])");
  } catch { /* ignore */ }
  await ev(page, SCROLL_JS, 0);
  await page.waitForTimeout(300);
}

export function plan(cfg, maxY) {
  // One integer scroll offset per output frame (deterministic timing).
  const fps = cfg.fps;
  const ys = new Array(Math.round(cfg.start_delay * fps)).fill(0);
  if (maxY > 0) {
    const secs = cfg.speed ? maxY / cfg.speed : cfg.duration; // speed==null -> duration seconds
    const n = Math.max(2, Math.round(secs * fps));
    const ef = cfg.scroll_mode === 'smooth' ? ease : (t) => t;
    let stop = cfg.pause_every;
    for (let i = 0; i <= n; i++) {
      const y = Math.round(maxY * ef(i / n));
      ys.push(y);
      if (stop && y > 0 && y < maxY && y >= stop) {
        for (let k = 0; k < Math.round(cfg.pause_for * fps); k++) ys.push(y);
        while (stop <= y) stop += cfg.pause_every;
      }
    }
  } else {
    for (let i = 0; i < Math.round(cfg.duration * fps); i++) ys.push(0);
  }
  for (let i = 0; i < Math.round(cfg.pause * fps); i++) ys.push(maxY);
  return ys;
}

export function planLoop(cfg, maxY, nTotal) {
  // --long mode: one integer scroll offset per output frame covering EXACTLY n_total frames, so the
  // video length is locked to the narration. The page is scrolled down slowly, back up, down again...
  // (smooth eased ping-pong) until the whole length is filled, ending with a hold wherever it landed.
  const fps = cfg.fps;
  const ys = new Array(Math.round(cfg.start_delay * fps)).fill(0);
  nTotal = Math.max(nTotal, ys.length + 1);
  if (maxY <= 0) return ys.concat(new Array(nTotal - ys.length).fill(0));
  const hold = Math.round(cfg.pause * fps);
  const scrollFrames = Math.max(2, nTotal - ys.length - hold);
  const n = cfg.speed
    ? Math.max(2, Math.round(maxY / cfg.speed * fps)) // frames in one full pass
    : scrollFrames;                                   // one pass = the whole video
  const ef = cfg.scroll_mode === 'smooth' ? ease : (t) => t;
  const down = [];
  for (let i = 0; i <= n; i++) down.push(Math.round(maxY * ef(i / n)));
  const up = down.slice(0, -1).reverse(); // reverse, skipping the shared turnaround frame
  const body = [];
  let stop = cfg.pause_every;
  let dirIsDown = true;
  while (body.length < scrollFrames) {
    const dir = dirIsDown ? down : up;
    for (const y of dir) {
      body.push(y);
      if (dirIsDown && stop && y > 0 && y < maxY && y >= stop) {
        for (let k = 0; k < Math.round(cfg.pause_for * fps); k++) body.push(y);
        while (stop <= y) stop += cfg.pause_every;
      }
      if (body.length >= scrollFrames) break;
    }
    dirIsDown = !dirIsDown;
    if (dirIsDown) stop = cfg.pause_every;
  }
  body.length = scrollFrames;
  ys.push(...body);
  while (ys.length < nTotal) ys.push(body[body.length - 1]); // hold the final position until the audio ends
  return ys.slice(0, nTotal);
}
