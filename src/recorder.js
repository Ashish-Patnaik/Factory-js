// recorder.js - frame capture from the target page (virtual time / stepped / realtime) feeding
// the mediabunny encoder page. This is the Node port of recorder.py: identical planning inputs
// and output framing, but the "ffmpeg image2pipe writer" is now the browser-side encoder.
import { SCROLL_JS, warmUp, maxScroll } from './scroll.js';
import { ev, sleep } from './utils.js';

// --- render-speed tunables (these caps fix the "some URLs render for 30-40 minutes" bug) ---
// Emulation.setVirtualTimePolicy's `pauseIfNetworkFetchesPending` defers the per-frame
// virtual-time budget while ANY network fetch is in flight. Pages with autoplaying videos
// (segment streaming), rotating ads and analytics beacons keep fetches perpetually pending,
// which used to stall EVERY frame until the old 10 s guard tripped -> 2+ s/frame, entirely
// URL-dependent. Both per-frame waits are now tightly capped: on quiet pages the budget
// still expires in microseconds of real time (determinism unchanged), and on streaming
// pages the frame simply captures one paint later instead of hanging.
const VT_BUDGET_CAP_MS = Number(process.env.RENDER_VT_CAP_MS ?? '400');
const VT_BUDGET_CAP_FIRST_MS = Number(process.env.RENDER_VT_CAP_FIRST_MS ?? '1500');
const VIDEO_READY_CAP_MS = Number(process.env.RENDER_VIDEO_CAP_MS ?? '250');
const VIDEO_CAP_FRAMES = 30; // early frames use the generous budget so opening images can land
// Cap on the whole per-frame __vt.step() evaluate (it contains two real-rAF waits). On quiet
// pages a paint lands in ~16 ms; a page whose main thread is congested (video seeks, huge
// animation sets, heavy script) can stall real rAF for seconds - this was the unbounded tail
// of the "some URLs render for 30-40 minutes" bug (2+ s/frame, entirely URL-dependent). When
// the cap trips we capture whatever is painted; the stale step neutralizes itself via the
// shim's generation counter instead of corrupting the next frame.
const STEP_CAP_MS = Number(process.env.RENDER_STEP_CAP_MS ?? '600');
// Adaptive video freeze: after this many frames whose video seek never settled (streaming /
// slow-network videos re-fetch ranges on EVERY frame and stall both the ready-poll and the
// next virtual-time budget), all page videos are paused once and per-frame re-seeking stops -
// they become static imagery instead of taxing every remaining frame of a 1300-frame render.
// 0 keeps re-seeking forever (old behavior).
const VIDEO_FREEZE_AFTER = Number(process.env.RENDER_VIDEO_FREEZE_AFTER ?? '8');

// ---------------------------------------------------------------------------
// Pacer: turns irregularly-timed captured frames into an exact constant-fps stream.
// Output slot k (time k/fps) shows the newest frame captured at or before it. O(1) memory.
// ---------------------------------------------------------------------------
export class Pacer {
  constructor(onWrite, fps, limit) {
    this.onWrite = onWrite;
    this.fps = fps;
    this.limit = limit;
    this.k = 0;
    this.prev = null;
    this.unique = 0;
  }

  push(frame, t) {
    if (this.prev !== null) this.fill(t);
    this.prev = frame;
    this.unique += 1;
  }

  fill(t) {
    if (this.prev === null) return;
    while (this.k < this.limit && this.k / this.fps < t) {
      this.onWrite(this.prev);
      this.k += 1;
    }
  }

  finish(n) {
    while (this.prev !== null && this.k < n) {
      this.onWrite(this.prev);
      this.k += 1;
    }
  }
}

export function yAt(ys, fps, t) {
  // Scroll offset at wall-clock time t (interpolated between planned samples).
  const x = Math.min(Math.max(t * fps, 0), ys.length - 1);
  const i = Math.floor(x);
  const j = Math.min(i + 1, ys.length - 1);
  return Math.round(ys[i] + (ys[j] - ys[i]) * (x - i));
}

// ---------------------------------------------------------------------------
// Capture backends
// ---------------------------------------------------------------------------
const VIDEO_SYNC_JS = `t => {
  let n = 0;
  for (const v of document.querySelectorAll('video')) {
    if (!(v.duration > 0) || v.readyState < 1) continue;
    n++; v.pause();
    const g = v.loop ? t % v.duration : Math.min(t, v.duration);
    if (Math.abs(v.currentTime - g) > 0.01) v.currentTime = g;
  }
  return n;
}`;
const VIDEOS_READY_JS = "() => [...document.querySelectorAll('video')].every((v) => !v.seeking)";
const FREEZE_VIDEOS_JS = `() => {
  let n = 0;
  for (const v of document.querySelectorAll('video')) { try { v.pause(); n++; } catch {} }
  return n;
}`;

function bounce(i, n) {
  // Frame index for output frame i when n frames play forward, then backward, then forward...
  if (n < 2) return 0;
  const period = 2 * (n - 1);
  const k = i % period;
  return k < n ? k : period - k;
}

async function captureScreenshot(cdp, clip = null) {
  // q92: at 1080x1920 the q95 JPEG encode costs roughly twice the q92 encode for detail
  // the H.264 stage re-quantizes away anyway (realtime mode already ran at 90-92).
  //
  // clip (document-relative CSS rect + scale): WITHOUT it, Chromium's captureScreenshot
  // returns the viewport at CSS-pixel resolution (360x640) even under deviceScaleFactor 3
  // - measured on Chromium 158: the "native 1080x1920 capture" was actually a 1x shot the
  // encoder upscaled. clip.scale + captureBeyondViewport yields exact DPR output and, since
  // the rect is document-relative, tiles can be captured without scrolling the page at all.
  const res = clip
    ? await cdp.send('Page.captureScreenshot', { format: 'jpeg', quality: 92, clip, captureBeyondViewport: true })
    : await cdp.send('Page.captureScreenshot', { format: 'jpeg', quality: 92 });
  return Buffer.from(res.data, 'base64');
}

// Deterministic render: browser time is frozen, then advanced exactly 1/fps s per frame; a frame
// is captured only after it is fully painted. Animations, rAF/WebGL loops, timers and scroll
// effects all follow the video clock. If n_out > len(ys) (video must be longer, e.g. to cover a
// voice-over), the rendered frames are replayed in reverse (ping-pong) from a disk cache.
export async function recordVirtual({ page, ys, encoder, cfg, nOut = null, cache }) {
  const fps = cfg.fps;
  const dt = 1000.0 / fps;
  if (nOut) ys = ys.slice(0, nOut);
  const total = ys.length;
  nOut = nOut ?? total;
  const cdp = await page.context().newCDPSession(page);
  let expired = false;
  cdp.on('Emulation.virtualTimeBudgetExpired', () => { expired = true; });
  try {
    await cdp.send('Emulation.setVirtualTimePolicy', { policy: 'pause' });
  } catch (e) {
    throw new Error(`virtual time unsupported by this browser (${e}); try --mode realtime`);
  }
  const t0 = Date.now();
  await ev(page, '() => window.__vt.activate()');
  const frameCache = [];
  // Probe video presence once: on video-less pages this skips a seek evaluate AND a
  // readiness poll per frame (1300+ CDP roundtrips saved over a typical render).
  const videoCount = await ev(page, '() => document.querySelectorAll("video").length').catch(() => 0);
  let videosFrozen = false, stalledVideoFrames = 0, stepTrips = 0;
  for (let i = 0; i < total; i++) {
    const y = ys[i];
    expired = false;
    try {
      await cdp.send('Emulation.setVirtualTimePolicy', { policy: 'pauseIfNetworkFetchesPending', budget: dt });
    } catch (e) {
      throw new Error(`virtual time failed (${e}); try --mode stepped or realtime`);
    }
    // Capped wait (see tunables note): streaming-heavy pages never drain their fetch queue,
    // so the budget's expiry can be deferred indefinitely - the old 10 s guard turned that
    // into up to 10 s PER FRAME. The budget is re-armed every frame, so capping the wait
    // only means the page clock lags the video clock slightly on busy pages (invisible).
    const cap = i < VIDEO_CAP_FRAMES ? VT_BUDGET_CAP_FIRST_MS : VT_BUDGET_CAP_MS;
    const end = Date.now() + cap;
    while (!expired && Date.now() < end) await page.waitForTimeout(2);
    // Bounded step: race the evaluate against STEP_CAP_MS. If the race is lost (main thread
    // congested, real rAF not firing), we capture what is painted; the stale step aborts via
    // the shim's generation counter and cannot corrupt the next frame.
    const stepped = ev(page, '([y, e]) => window.__vt.step(y, e, performance.now())', [y, (i + 1) * dt]);
    const done = await Promise.race([stepped.then(() => true), sleep(STEP_CAP_MS).then(() => false)]);
    if (!done) { stepTrips++; stepped.catch(() => {}); }
    if (videoCount && !videosFrozen) {
      // Seeks fire new video range requests; give them a short window to stop seeking, then
      // capture whatever is painted (the next frame re-seeks to its own time anyway).
      await ev(page, VIDEO_SYNC_JS, (i + 1) / fps);
      const end2 = Date.now() + VIDEO_READY_CAP_MS;
      let ready = false;
      while (Date.now() < end2 && !(ready = await ev(page, VIDEOS_READY_JS))) await page.waitForTimeout(25);
      if (!ready && i >= VIDEO_CAP_FRAMES && VIDEO_FREEZE_AFTER > 0
        && ++stalledVideoFrames >= VIDEO_FREEZE_AFTER) {
        console.log(`\nnote: page video seek(s) never settle (stalled ${stalledVideoFrames} frames) - ` +
          'freezing videos as still images for the rest of the render');
        await ev(page, FREEZE_VIDEOS_JS).catch(() => {});
        videosFrozen = true;
      }
    }
    const frame = await captureScreenshot(cdp);
    await encoder.pushFrame(frame, i / fps);
    if (nOut > total) frameCache.push(frame); // keep for reverse playback
    if (i % 10 === 0 || i === total - 1) {
      const el = (Date.now() - t0) / 1000;
      const eta = el / (i + 1) * (total - i - 1);
      process.stderr.write(`  rendering frame ${i + 1}/${total} (${Math.floor((i + 1) * 100 / total)}%)  elapsed ${el.toFixed(0)}s  eta ${eta.toFixed(0)}s   \r`);
    }
  }
  process.stderr.write('\n');
  if (stepTrips) {
    console.log(`note: ${stepTrips} frame(s) captured without full-paint confirmation ` +
      '(the page kept the main thread busy; those frames may jitter slightly)');
  }
  for (let i = total; i < nOut; i++) { // reverse playback until the voice-over ends
    await encoder.pushFrame(frameCache[bounce(i, total)], i / fps);
  }
  return { backend: 'virtual-time', captured: total, seconds: nOut / fps, frames: nOut,
    reversed: nOut - total, renderS: (Date.now() - t0) / 1000, stepTrips, videosFrozen };
}

// ---------------------------------------------------------------------------
// Tile-snapshot capture (the real fix for the "some URLs render for 30-40 minutes" bug).
//
// The live backends above talk to the page ONCE PER OUTPUT FRAME (virtual-time budget,
// paint wait, video seek + ready poll, screenshot = 4 round-trips x 1300 frames). On heavy
// pages (entertainment hubs: rotating ads, lazy images, autoplaying article video) every
// single one of those waits burns to its cap -> a hard ~1.5-2 s/frame floor even after the
// caps landed, i.e. still 30-45 min for a 42 s reel. The caps bound the damage; they cannot
// remove the per-frame page dependency.
//
// recordTiled removes it: the page is flattened and screenshotted ONCE as a vertical strip
// of viewport-sized tiles (a 7919 px page @ 649 px viewport = 14 shots, not 1270), the tiles
// are decoded once in the encoder page, and every video frame is composed there from two
// adjacent tile bitmaps (GPU drawImage + captions/avatar + H.264, same visual pipeline as
// addFrame/renderSlides). Zero per-frame page interaction -> render time is identical for a
// quiet blog and movieweb. Animations & video are frozen in the snapshot (paused, not
// seeked), sticky/fixed chrome is un-stuck so it cannot repeat inside every tile band.
// RENDER_TILED=0 (or --mode stepped/realtime) restores the live capture paths; any failure
// in the tile path falls back to recordVirtual before the first frame is encoded.
// ---------------------------------------------------------------------------
const TILE_IMAGES_SETTLE_MS = Number(process.env.RENDER_TILE_SETTLE_MS ?? '2000');
const TILE_MAX = Number(process.env.RENDER_TILE_MAX ?? '400');

// Pause videos/animations and neutralize sticky/fixed positioning so the tile strip is a
// flat, consistent snapshot of the document flow (like a print view):
//   * sticky -> relative with auto offsets: the element keeps its place in the flow
//     instead of re-anchoring to the top of every tile's viewport;
//   * fixed elements (back-to-top buttons, docked bars) are hidden - they would otherwise
//     repeat in every tile. Full-viewport fixed containers are KEPT (some web apps render
//     their whole UI fixed); hiding the page's real content would be worse than repetition.
const FREEZE_PAGE_JS = `() => {
  let sticky = 0, fixed = 0, anims = 0, videos = 0;
  for (const v of document.querySelectorAll('video')) {
    try { if (!v.paused) { v.pause(); videos++; } } catch {}
  }
  try { for (const a of document.getAnimations()) { try { a.pause(); anims++; } catch {} } } catch {}
  for (const el of document.querySelectorAll('*')) {
    let s; try { s = getComputedStyle(el); } catch { continue; }
    if (s.position === 'sticky') {
      el.style.position = 'relative';
      el.style.top = el.style.bottom = el.style.left = el.style.right = 'auto';
      sticky++;
    } else if (s.position === 'fixed' && el !== document.body && el !== document.documentElement) {
      const r = el.getBoundingClientRect();
      if (r.width < innerWidth * 0.85 || r.height < innerHeight * 0.85) {
        el.style.setProperty('visibility', 'hidden', 'important');
        fixed++;
      }
    }
  }
  return { sticky, fixed, anims, videos };
}`;

// Bounded wait for ALL images on the page (lazy content the pre-scroll warmed but that is
// still decoding). Returns the number still incomplete when the cap trips.
const TILE_IMAGES_SETTLE_JS = '() => {'
  + ' const pending = [...document.images].filter((i) => !i.complete);'
  + ' if (!pending.length) return 0;'
  + ' return new Promise((res) => {'
  + '  const t0 = Date.now();'
  + '  const tick = () => {'
  + '   if (!pending.some((i) => !i.complete) || Date.now() - t0 > ' + TILE_IMAGES_SETTLE_MS + ') return res(pending.length);'
  + '   setTimeout(tick, 50);'
  + '  };'
  + '  tick();'
  + ' });'
  + '}';

export async function recordTiled({ page, ys, encoder, cfg, nOut = null }) {
  if (nOut) ys = ys.slice(0, nOut);
  const total = ys.length;
  nOut = nOut ?? total;
  const fps = cfg.fps;
  const t0 = Date.now();
  const fail = (msg, phase) => Object.assign(new Error(msg), { tilePhase: phase });
  console.log('[tiles] snapshot mode: capturing the page once as tiles, then composing every frame ' +
    'locally (animations & video frozen; RENDER_TILED=0 restores the live capture)');

  // 1. Pre-scroll so lazy images below the fold are triggered and loaded, then one bounded
  //    wait for everything still decoding: the tiles themselves are captured without
  //    scrolling (document-relative clips), so there is no later chance to load anything.
  const [iw, vh, dpr] = await ev(page, '() => [window.innerWidth, window.innerHeight, window.devicePixelRatio]');
  await warmUp(page, Math.max(vh - 100, 100));
  const incomplete = await ev(page, TILE_IMAGES_SETTLE_JS).catch(() => -1);
  if (incomplete > 0) console.log(`note: ${incomplete} image(s) never finished loading; captured as-is`);

  // 2. Flatten: pause videos/animations, un-stick sticky/fixed chrome.
  const flat = await ev(page, FREEZE_PAGE_JS).catch(() => null);
  if (!flat) throw fail('could not freeze the page for tile capture', 'capture');
  const notes = [`${flat.videos} video(s) paused`, `${flat.anims} animation(s) frozen`];
  if (flat.sticky || flat.fixed) notes.push(`${flat.sticky} sticky -> flow, ${flat.fixed} fixed hidden`);
  console.log(`[tiles] page frozen: ${notes.join(', ')}`);

  const maxY = await maxScroll(page);
  const step = Math.max(1, vh);
  const nTiles = Math.min(Math.ceil(maxY / step) + 1, TILE_MAX);
  if (Math.ceil(maxY / step) + 1 > TILE_MAX) throw fail('page too tall for a tile snapshot', 'capture');

  // 3. Capture the tile strip WITHOUT scrolling: document-relative clips at scroll offsets
  //    0, step, 2*step... rendered directly at the device pixel ratio (exact 1080x1920
  //    bitmaps on a 360x640 @3x viewport). No scroll -> no per-tile settling or re-layout.
  let cdp;
  try {
    cdp = await page.context().newCDPSession(page);
  } catch (e) {
    throw fail(`no CDP session for tile capture (${String(e?.message ?? e).split('\n')[0]})`, 'capture');
  }
  const tiles = [];
  const tc0 = Date.now();
  for (let k = 0; k < nTiles; k++) {
    const y = Math.min(k * step, maxY);
    const clip = { x: 0, y, width: iw, height: Math.min(step, maxY + step - y), scale: dpr };
    tiles.push(await captureScreenshot(cdp, clip));
    process.stderr.write(`  [tiles] capturing tile ${k + 1}/${nTiles}\r`);
  }
  process.stderr.write('\n');
  const captureS = (Date.now() - tc0) / 1000;

  // 4. Hand the tiles to the encoder page (decoded once into ImageBitmaps there).
  await encoder.prepareTiles(tiles.map((b) => b.toString('base64')), step)
    .catch((e) => { throw fail(`tile upload failed (${String(e?.message ?? e).split('\n')[0]})`, 'capture'); });

  // 5. Per-frame band plan [tileIndex, offsetCssPx, tSeconds] - the exact scroll offsets the
  //    live captures would have used, including the reverse-playback fill (bounce).
  const frames = new Array(nOut);
  for (let i = 0; i < nOut; i++) {
    const y = Math.max(0, Math.min(ys[i < total ? i : bounce(i, total)], maxY));
    const k = Math.min(Math.floor(y / step), nTiles - 1);
    frames[i] = [k, y - k * step, i / fps];
  }
  const CHUNK = Math.max(10, Number(process.env.RENDER_TILE_CHUNK ?? '90'));
  let done = 0;
  try {
    for (let i = 0; i < frames.length; i += CHUNK) {
      await encoder.renderTiledChunk({ frames: frames.slice(i, i + CHUNK) });
      done = Math.min(i + CHUNK, frames.length);
      const el = (Date.now() - t0) / 1000;
      const eta = el / done * (frames.length - done);
      process.stderr.write(`  rendering frame ${done}/${frames.length} (${Math.floor(done * 100 / frames.length)}%)  ` +
        `elapsed ${el.toFixed(0)}s  eta ${eta.toFixed(0)}s   \r`);
    }
  } catch (e) {
    // Frames are already in the encoder - falling back now would duplicate content.
    throw fail(`tile render failed mid-way: ${String(e?.message ?? e).split('\n')[0]}`, 'render');
  }
  process.stderr.write('\n');
  const renderS = (Date.now() - t0) / 1000;
  console.log(`[tiles] ${nTiles} tile(s) in ${captureS.toFixed(1)}s, ${frames.length} frame(s) composed locally in ${(renderS - captureS).toFixed(1)}s`);
  return { backend: 'tiled', captured: nTiles, tiles: nTiles, seconds: nOut / fps, frames: nOut,
    reversed: Math.max(0, nOut - total), renderS };
}

// Deterministic capture without a virtual clock (used with Obscura): for every video frame,
// scroll to the exact planned offset, let it paint, capture it. Perfectly even scrolling for
// static pages; page animations are not frame-locked. Holds reuse the previous frame; if
// n_out > len(ys) the frames replay in reverse.
export async function recordStepped({ page, ys, encoder, cfg, nOut = null, cache }) {
  if (nOut) ys = ys.slice(0, nOut);
  const total = ys.length;
  nOut = nOut ?? total;
  const t0 = Date.now();
  let prev = null;
  let frame = Buffer.alloc(0);
  await ev(page, SCROLL_JS, 0);
  const cacheDir = [];
  for (let i = 0; i < total; i++) {
    const y = ys[i];
    if (y !== prev) {
      await ev(page, SCROLL_JS, y);
      await page.waitForTimeout(15); // let scroll handlers / lazy content apply before the shot
      frame = await page.screenshot({ type: 'jpeg', quality: 92 });
      prev = y;
    }
    await encoder.pushFrame(frame, i / cfg.fps);
    if (nOut > total) cacheDir.push(frame);
    if (i % 10 === 0 || i === total - 1) {
      const el = (Date.now() - t0) / 1000;
      const eta = el / (i + 1) * (total - i - 1);
      process.stderr.write(`  capturing frame ${i + 1}/${total} (${Math.floor((i + 1) * 100 / total)}%)  elapsed ${el.toFixed(0)}s  eta ${eta.toFixed(0)}s   \r`);
    }
  }
  process.stderr.write('\n');
  for (let i = total; i < nOut; i++) {
    await encoder.pushFrame(cacheDir[bounce(i, total)], i / cfg.fps);
  }
  return { backend: 'stepped', captured: total, seconds: nOut / cfg.fps, frames: nOut,
    reversed: nOut - total, renderS: (Date.now() - t0) / 1000 };
}

async function startScreencast(page) {
  // CDP Page.startScreencast. Returns (session, queue) or null if unsupported/silent.
  try {
    const s = await page.context().newCDPSession(page);
    const q = [];
    s.on('Page.screencastFrame', (p) => q.push({ sessionId: p.sessionId, data: p.data }));
    const vp = page.viewportSize();
    const sizeOpt = vp ? { maxWidth: vp.width, maxHeight: vp.height } : {};
    await s.send('Page.startScreencast', { format: 'jpeg', quality: 90, everyNthFrame: 1, ...sizeOpt });
    await ev(page, '() => { window.scrollTo(0, 1); window.scrollTo(0, 0); }'); // force a repaint to probe
    const end = Date.now() + 2500;
    while (Date.now() < end && !q.length) await page.waitForTimeout(50);
    if (!q.length) {
      await s.send('Page.stopScreencast');
      return null;
    }
    return { s, q };
  } catch {
    return null;
  }
}

// Real-time capture: the page scrolls on a wall clock while frames are grabbed continuously,
// so CSS/JS animations and video play at true speed. Returns capture stats.
export async function recordRealtime({ page, ys, encoder, cfg, backend = 'auto' }) {
  const fps = cfg.fps;
  const total = ys.length / fps;
  const nOut = Math.ceil(total * fps);
  let pushed = 0;
  const pushFrame = async (frame) => { await encoder.pushFrame(frame, pushed / fps); pushed += 1; };
  const pacer = { unique: 0, last: null,
    push(frame) { this.last = frame; this.unique += 1; },
    fill(t) { /* handled below with pushFrame holds */ },
  };
  const sc = backend === 'screenshot' ? null : await startScreencast(page);
  if (backend === 'screencast' && !sc) throw new Error('Page.startScreencast is not producing frames in this browser');
  const used = sc ? 'screencast' : 'screenshot';
  let lastPrint = -1;
  await ev(page, SCROLL_JS, 0);
  const t0 = Date.now();
  try {
    for (;;) {
      const t = (Date.now() - t0) / 1000;
      if (t >= total) break;
      await ev(page, SCROLL_JS, yAt(ys, fps, t));
      if (sc) {
        const { s, q } = sc;
        await page.waitForTimeout(4); // pumps CDP events
        while (q.length) {
          const { sessionId, data } = q.shift();
          pacer.push(Buffer.from(data, 'base64'), (Date.now() - t0) / 1000);
          await s.send('Page.screencastFrameAck', { sessionId }).catch(() => {});
          await pushFrame(pacer.last);
        }
      } else {
        const shot = await page.screenshot({ type: 'jpeg', quality: 92 });
        pacer.push(shot, t);
        await pushFrame(shot);
      }
      if (Math.floor(t) !== lastPrint) {
        lastPrint = Math.floor(t);
        process.stderr.write(`  recording ${t.toFixed(1).padStart(4)}/${total.toFixed(1)}s (${used}, ${pacer.unique} captured frames)\r`);
      }
    }
  } finally {
    if (sc) { try { await sc.s.send('Page.stopScreencast'); } catch { /* gone */ } }
  }
  process.stderr.write('\n');
  // pad the remaining slots by holding the last frame (Pacer.finish equivalent)
  const t = (Date.now() - t0) / 1000;
  while (pushed < nOut && pacer.last) {
    await pushFrame(pacer.last);
    if (pushed / fps > t + 1) break;
  }
  return { backend: used, captured: pacer.unique, seconds: total, frames: nOut };
}
