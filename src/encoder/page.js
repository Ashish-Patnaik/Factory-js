// encoder/page.js - browser-side media engine, running inside a headless Chromium/Chrome/Edge
// "encoder page" launched by host.js. This is what replaces ffmpeg in the Node port:
//
//   * composes every video frame on an OffscreenCanvas (fit/pad on white, burned captions,
//     avatar gif overlay, Ken Burns slideshow motion, crossfades) - the same visual pipeline
//     the Python factory built with ffmpeg filters;
//   * encodes H.264 (yuv420p, limited range) via WebCodecs and muxes MP4 with mediabunny;
//   * encodes AAC 44.1 kHz audio (native WebCodecs, with @mediabunny/aac-encoder's WASM
//     encoder as the fallback when the browser build lacks AAC);
//   * probes and auto-heals files for the Instagram Graph API (duration cap, silent audio).
//
// The host injects this file as a blob module with these two tokens rewritten:
//   '__MEDIABUNNY_URL__'  -> blob URL of the mediabunny ESM bundle
//   '__AAC_ENCODER_URL__' -> blob URL of the @mediabunny/aac-encoder ESM bundle
import * as mb from '__MEDIABUNNY_URL__';
import { registerAacEncoder } from '__AAC_ENCODER_URL__';

const { Output, Mp4OutputFormat, BufferTarget, CanvasSource, VideoSampleSource, AudioBufferSource,
  VideoSample, Input, ALL_FORMATS, BlobSource, Conversion, Quality } = mb;

function log(...a) { console.log('[enc]', ...a); }

// Instagram/Meta ingest requires H.264 yuv420p LIMITED (tv) range. Canvas-derived VideoFrames
// are full range by default and newer Chromium builds ignore the colorSpace override on canvas
// frames - the exact bug that killed uploads with ProcessingFailedError in the Python era. So
// every frame goes through mediabunny's own RGBA->I420 conversion with an explicit limited
// bt709 color space (faster than WebCodecs' canvas path too).
const LIMITED_BT709 = { fullRange: false, matrix: 'bt709', primaries: 'bt709', transfer: 'bt709' };

function sampleFromCanvas(ctx, canvas, tSec, fps) {
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const sample = new VideoSample(img.data.buffer, {
    format: 'RGBA',
    codedWidth: canvas.width,
    codedHeight: canvas.height,
    timestamp: tSec,          // VideoSampleInit timestamps are in SECONDS
    duration: 1 / fps,
    colorSpace: LIMITED_BT709,
  });
  return sample;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
const QUALITY_LADDER = [ // CRF-equivalent -> bitrate share of a 24 Mbps 1080p30 ceiling
  { max: 14, per: 0.42 }, { max: 18, per: 0.33 }, { max: 22, per: 0.25 },
  { max: 28, per: 0.19 }, { max: 34, per: 0.12 }, { max: 52, per: 0.08 },
];

function crfToBitrate(quality, w, h, fps) {
  const unit = (w * h) / (1920 * 1080);
  const per = QUALITY_LADDER.find((q) => quality <= q.max).per;
  return Math.round(Math.min(24e6, Math.max(600e3, 24e6 * unit * per * (fps / 30))));
}

export function parseBitrate(str) {
  const m = /^(\d+(?:\.\d+)?)\s*([kKmM]?)/.exec(String(str).trim());
  if (!m) return null;
  const n = parseFloat(m[1]);
  return /m/i.test(m[2]) ? n * 1e6 : /k/i.test(m[2]) ? n * 1e3 : n;
}

function hexToRgb(hex) {
  const c = hex.replace('#', '');
  return [parseInt(c.slice(0, 2), 16), parseInt(c.slice(2, 4), 16), parseInt(c.slice(4, 6), 16)];
}

async function decodeImage(bytes, type = 'image/jpeg') {
  return createImageBitmap(new Blob([bytes], { type }));
}

// ---------------------------------------------------------------------------
// Caption + avatar drawing (mirrors captions.py buildAss & the avatar overlay)
// ---------------------------------------------------------------------------
function strokeAndFill(ctx, text, x, y) {
  ctx.strokeText(text, x, y);
  ctx.fillText(text, x, y);
}

// \kf karaoke: left-to-right highlight sweep within the current word.
function drawKaraokeWord(ctx, text, cx, y, width, frac, hi) {
  const x0 = cx - width / 2;
  ctx.fillStyle = '#ffffff';
  strokeAndFill(ctx, text, cx, y);
  if (frac <= 0) return;
  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, y - 2, width * Math.min(1, frac), ctx.canvas.height);
  ctx.clip();
  ctx.fillStyle = hi;
  ctx.fillText(text, cx, y);
  ctx.restore();
}

// TikTok-style captions drawn per frame: pop (recolour + 115%->100% scale-in) or karaoke,
// \pos centring, \fad(80,40), outline 4 / shadow 1.5 (scaled), shrink-to-fit long lines.
function drawCaptions(ctx, cap, t) {
  if (!cap) return;
  const { lines, style } = cap;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  for (const line of lines) {
    if (t < line.start || t > line.end) continue;
    const fsLine = line.shrink || style.size;
    ctx.font = `${fsLine}px "${style.font}", sans-serif`;
    const lineAge = t - line.start;
    const alpha = Math.max(0, Math.min(1, Math.min(lineAge / 0.08, (line.end - t) / 0.04)));
    ctx.globalAlpha = alpha;
    const words = line.words;
    const gap = fsLine * 0.24;
    const widths = words.map((w) => ctx.measureText(w.text).width);
    const totalW = widths.reduce((a, b) => a + b, 0) + gap * (words.length - 1);
    let x = style.centerX - totalW / 2;
    const y = style.posY;
    ctx.lineJoin = 'round';
    ctx.miterLimit = 2;
    ctx.strokeStyle = '#000';
    ctx.lineWidth = style.outline * (line.shrink ? fsLine / style.size : 1);
    ctx.shadowColor = 'rgba(0,0,0,0.55)';
    ctx.shadowBlur = 0;
    ctx.shadowOffsetY = style.shadow;
    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      const cx = x + widths[i] / 2;
      const active = !style.karaoke && t >= w.start && t < w.end;
      if (style.karaoke) {
        const frac = t >= w.end ? 1 : t <= w.start ? 0 : (t - w.start) / (w.end - w.start);
        drawKaraokeWord(ctx, w.text, cx, y, widths[i], frac, style.css);
      } else if (active) {
        const pop = 1 + 0.15 * Math.max(0, 1 - lineAge / 0.06);
        ctx.save();
        ctx.translate(cx, y - fsLine * 0.36);
        ctx.scale(pop, pop);
        ctx.fillStyle = style.css;
        strokeAndFill(ctx, w.text, 0, fsLine * 0.36);
        ctx.restore();
      } else {
        ctx.fillStyle = '#ffffff';
        strokeAndFill(ctx, w.text, cx, y);
      }
      x += widths[i] + gap;
    }
    ctx.shadowColor = 'transparent';
    ctx.shadowOffsetY = 0;
    ctx.globalAlpha = 1;
  }
}

// --avatar: the talking-character gif, looping in a corner until the render ends.
function drawAvatar(ctx, av, t, width, height) {
  if (!av || !av.frames.length) return;
  const gt = av.total > 0 ? t % av.total : 0;
  let f = av.frames[av.frames.length - 1];
  for (const fr of av.frames) {
    if (fr.timestamp <= gt) f = fr; else break;
  }
  const short = Math.min(width, height);
  const aw = Math.max(16, Math.round(short * av.sizePct / 100 / 2) * 2);
  const m = Math.max(8, Math.round(short * 0.03 / 2) * 2);
  const scale = aw / f.image.displayWidth;
  const dh = Math.max(2, Math.round(f.image.displayHeight * scale / 2) * 2);
  let x = 0, y = 0;
  if (av.corner === 'bottom-right') { x = width - aw - m; y = height - dh - m; }
  else if (av.corner === 'bottom-left') { x = m; y = height - dh - m; }
  else if (av.corner === 'top-right') { x = width - aw - m; y = m; }
  else { x = m; y = m; }
  ctx.drawImage(f.image, x, y, aw, dh);
}

// ---------------------------------------------------------------------------
// Encoder session (one output file)
// ---------------------------------------------------------------------------
class Session {
  constructor() {
    this.output = null;
    this.vSrc = null;
    this.aSrc = null;
    this.frames = 0;
    this.fps = 30;
    this.width = 0;
    this.height = 0;
    this.captions = null;   // { lines, style }
    this.avatar = null;     // { frames: [{image, timestamp, duration}], total, corner, sizePct }
    this.audio = { rate: 0, chunks: [] };
    this.audioDelay = 0;
    this.t0 = 0;
    this.bytes = null;
  }

  async init({ width, height, fps, quality, bitrate }) {
    this.width = width;
    this.height = height;
    this.fps = fps;
    const rate = bitrate ?? crfToBitrate(quality, width, height, fps);
    this.output = new Output({ format: new Mp4OutputFormat({ fastStart: 'in-memory' }), target: new BufferTarget() });
    this.canvas = new OffscreenCanvas(width, height);
    this.ctx = this.canvas.getContext('2d', { alpha: false });
    this.ctx.imageSmoothingQuality = 'high';
    this.vSrc = new VideoSampleSource({
      codec: 'avc',
      keyFrameInterval: 2,
      quality: new Quality({ bitrate: rate }),
      latencyMode: 'quality',
    });
    this.output.addVideoTrack(this.vSrc, { frameRate: fps });
    this.bitrate = rate;
    // the audio track must exist before output.start(); samples arrive later
    this.aSrc = new AudioBufferSource({ codec: 'aac', bitrate: 192000 });
    this.output.addAudioTrack(this.aSrc);
  }

  async beginAudio({ sampleRate, audioDelay, totalDuration }) {
    // AAC at 44.1 kHz like the Python factory's mux step; the TTS waveform is resampled here.
    this.audioDelay = audioDelay ?? 0;
    this.totalDuration = totalDuration ?? 0;
    this.audio = { rate: sampleRate, chunks: [] };
  }

  async pushAudio(pcm) { // Float32Array (transferred from the host)
    this.audio.chunks.push(pcm);
  }

  buildAudioBuffer() {
    const chunks = this.audio.chunks;
    const n = chunks.reduce((a, c) => a + c.length, 0);
    const src = new Float32Array(n);
    let o = 0;
    for (const c of chunks) { src.set(c, o); o += c.length; }
    const OUT = 44100;
    const rate = this.audio.rate || 24000;
    const delay = Math.round((this.audioDelay || 0) * OUT);
    const need = Math.max(delay + Math.ceil(src.length / rate * OUT), Math.ceil((this.totalDuration || 0) * OUT), 1);
    const out = new Float32Array(need); // leading zeros + padding = adelay + apad, like the ffmpeg flags
    for (let i = 0; i < src.length; i++) {
      const x = i / rate * OUT;
      const i0 = Math.floor(x);
      if (delay + i0 + 1 >= need) break;
      const frac = x - i0;
      out[delay + i0] += src[i] * (1 - frac);
      out[delay + i0 + 1] += src[i] * frac;
    }
    const buf = new AudioBuffer({ numberOfChannels: 1, length: out.length, sampleRate: OUT });
    // AudioBuffer is ZERO-FILLED on construction - without this copy the muxed AAC track
    // is N seconds of perfect silence (video fine, audio "missing"). A/B-proven.
    buf.getChannelData(0).set(out);
    return buf;
  }

  async start() {
    await this.output.start();
    this.t0 = performance.now();
  }

  // Compose + add one captured frame (JPEG bytes) at time t.
  async addFrame(bytes, t) {
    const bmp = await decodeImage(bytes);
    const ctx = this.ctx;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, this.width, this.height);
    const s = Math.min(this.width / bmp.width, this.height / bmp.height);
    const dw = Math.max(2, Math.round(bmp.width * s));
    const dh = Math.max(2, Math.round(bmp.height * s));
    ctx.drawImage(bmp, (this.width - dw) / 2, (this.height - dh) / 2, dw, dh);
    bmp.close();
    drawCaptions(ctx, this.captions, t);
    drawAvatar(ctx, this.avatar, t, this.width, this.height);
    const sample = sampleFromCanvas(ctx, this.canvas, t, this.fps);
    await this.vSrc.add(sample);
    sample.close();
    this.frames++;
  }

  async attachAudio() {
    const buf = this.buildAudioBuffer();
    await this.aSrc.add(buf);
    await this.aSrc.close();
  }

  async finish() {
    await this.attachAudio();
    await this.vSrc.close();
    await this.output.finalize();
    this.bytes = new Uint8Array(this.output.target.buffer);
    return { frames: this.frames, ms: performance.now() - this.t0, bytes: this.bytes.byteLength };
  }
}

// ---------------------------------------------------------------------------
// Slideshow (--image): Ken Burns + crossfades, composed here with no host round-trips
// ---------------------------------------------------------------------------
const SS_W = 2880, SS_H = 1620;   // supersampled slide canvas (16:9), like slides.py
const COVER_KEEP = 0.60;
const MOTIONS = ['in', 'out', 'panright', 'panleft', 'in', 'out', 'panright', 'out'];

function fitSlideCanvas(bmp) {
  // Port of slides.py fit_slide: cover-crop when safe, cinematic blur-fill for extremes.
  const c = new OffscreenCanvas(SS_W, SS_H);
  const ctx = c.getContext('2d');
  const ar = bmp.width / bmp.height;
  const target = SS_W / SS_H;
  const okAr = ar >= 1 / 8 && ar <= 8;
  const coverOk = okAr && ((ar >= target && target / ar >= COVER_KEEP) || (ar < target && ar / target >= COVER_KEEP));
  ctx.imageSmoothingQuality = 'high';
  if (coverOk) {
    // ImageOps.fit equivalent, centre-weighted slightly up (centering=(0.5, 0.42))
    const s = Math.max(SS_W / bmp.width, SS_H / bmp.height);
    const dw = bmp.width * s, dh = bmp.height * s;
    ctx.drawImage(bmp, (SS_W - dw) / 2, (SS_H - dh) * 0.42, dw, dh);
    return c;
  }
  // blur-fill: blurred + darkened cover backdrop, full image contained on top
  ctx.filter = `blur(${Math.round(SS_W / 48)}px) brightness(0.55)`;
  const s = Math.max(SS_W / bmp.width, SS_H / bmp.height);
  ctx.drawImage(bmp, (SS_W - bmp.width * s) / 2, (SS_H - bmp.height * s) / 2, bmp.width * s, bmp.height * s);
  ctx.filter = 'none';
  let fw = Math.min(SS_W, Math.max(Math.round(SS_H * ar), 1));
  let fh = Math.min(SS_H, Math.max(Math.round(SS_W / ar), 1));
  if (fw > SS_W * 0.62 || fh > SS_H * 0.92) {
    const scale = Math.min((SS_W * 0.62) / fw, (SS_H * 0.92) / fh, 1.0);
    fw = Math.max(Math.floor(fw * scale / 2) * 2, 16);
    fh = Math.max(Math.floor(fh * scale / 2) * 2, 16);
  }
  ctx.drawImage(bmp, (SS_W - fw) / 2, (SS_H - fh) / 2, fw, fh);
  return c;
}

function kbMotion(kind, on, D, amp = 0.14) {
  const d1 = Math.max(D - 1, 1);
  if (kind === 'in') return { z: 1 + amp * on / d1, fx: 0.5, fy: 0.5 };
  if (kind === 'out') return { z: Math.max(1 + amp - amp * on / d1, 1), fx: 0.5, fy: 0.5 };
  if (kind === 'panright') return { z: 1.12, fx: on / d1, fy: 0.5 };
  return { z: 1.12, fx: 1 - on / d1, fy: 0.5 }; // panleft
}

// ---------------------------------------------------------------------------
// Top-level API used by host.js
// ---------------------------------------------------------------------------
let session = null;
let aacReady = false;

async function ensureAacFallback() {
  if (aacReady) return;
  aacReady = true;
  try {
    if (!(await mb.canEncodeAudio('aac'))) {
      registerAacEncoder();
      log('native AAC unavailable - registered the WASM AAC encoder extension');
    }
  } catch (e) {
    log('AAC capability check failed:', e?.message ?? e);
  }
}

export const api = {
  // --- session lifecycle -------------------------------------------------
  async init(opts) {
    await ensureAacFallback();
    session = new Session();
    await session.init(opts);
    log(`session: ${opts.width}x${opts.height} @${opts.fps}fps`);
  },
  async setFont(bytes) {
    const ff = new FontFace('Anton', bytes);
    await ff.load();
    document.fonts.add(ff);
    log('Anton font registered');
  },
  async setCaptions(data) {
    if (!data) return;
    session.captions = data;
    session.captions.style.css = data.style.color;
    log(`captions: ${data.lines.length} line(s), style ${data.style.karaoke ? 'karaoke' : 'pop'}`);
  },
  async setAvatar(bytes, corner, sizePct) {
    if (!bytes?.byteLength) return;
    const dec = new ImageDecoder({ data: bytes, type: 'image/gif' });
    await dec.tracks.ready;
    const track = dec.tracks.selectedTrack;
    const frames = [];
    let total = 0;
    const count = Math.min(track.frameCount || 1, 4000);
    for (let i = 0; i < count; i++) {
      const { image } = await dec.decode({ frameIndex: i });
      const dur = (image.duration ?? 100000) / 1e6;
      frames.push({ image, timestamp: total, duration: dur });
      total += dur;
      if (total > 600) break; // huge gifs: cap at 10 minutes of loop material
    }
    session.avatar = { frames, total, corner, sizePct };
    log(`avatar gif: ${frames.length} frame(s), ${total.toFixed(2)}s loop`);
  },
  async beginAudio(opts) { await session.beginAudio(opts); },
  async pushAudio(pcm) { await session.pushAudio(pcm); },
  async start() { await session.start(); },

  // --- scroll-video frames ------------------------------------------------
  async addFrames(batch) {
    // batch: [{ b64, t }] - decoded and composed in order
    for (const f of batch) {
      const bytes = Uint8Array.from(atob(f.b64), (c) => c.charCodeAt(0));
      await session.addFrame(bytes, f.t);
    }
    return session.frames;
  },

  // --- tile-snapshot rendering (recorder.recordTiled) ----------------------
  // The page was screenshotted once as a vertical strip of viewport-sized tiles at scroll
  // offsets 0, step, 2*step... (CSS px). prepareTiles decodes them once; renderTiledChunk
  // composes each video frame by drawing two adjacent tiles shifted by the frame's scroll
  // offset - identical geometry to a live scrolling capture (tile k+1's top edge lands
  // exactly on tile k's bottom edge), with zero per-frame page round-trips. The composed
  // band then goes through the same fit/pad + captions + avatar path as addFrame.
  async prepareTiles(jpegs, step) {
    const tiles = [];
    for (const b64 of jpegs) {
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      tiles.push(await decodeImage(bytes));
    }
    session.tiles = tiles;
    session.tileStep = step;
    log(`tiles: ${tiles.length} snapshot(s), step ${step}px`);
    return tiles.length;
  },

  async renderTiledChunk({ frames }) {
    // frames: [[tileIndex, offsetCssPx, tSeconds], ...]
    const tiles = session.tiles;
    if (!tiles?.length || !session.tileStep) throw new Error('no tile snapshot prepared');
    const ctx = session.ctx;
    const W = session.width, H = session.height;
    const first = tiles[0];
    const s = Math.min(W / first.width, H / first.height); // same fit/pad math as addFrame
    const dw = Math.max(2, Math.round(first.width * s));
    const dh = Math.max(2, Math.round(first.height * s));
    const dx = (W - dw) / 2, dy = (H - dh) / 2;
    const ratio = first.height / session.tileStep; // tile bitmap px per CSS px
    for (const [k, off, t] of frames) {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, W, H);
      ctx.save();
      ctx.beginPath();
      ctx.rect(dx, dy, dw, dh); // keep the tiles inside the fitted band (white pad elsewhere)
      ctx.clip();
      const offB = off * ratio;
      ctx.drawImage(tiles[Math.min(k, tiles.length - 1)], dx, dy - offB * s, dw, dh);
      const nxt = tiles[k + 1];
      if (nxt) ctx.drawImage(nxt, dx, dy + (first.height - offB) * s, dw, dh);
      ctx.restore();
      drawCaptions(ctx, session.captions, t);
      drawAvatar(ctx, session.avatar, t, W, H);
      const sample = sampleFromCanvas(ctx, session.canvas, t, session.fps);
      await session.vSrc.add(sample);
      sample.close();
      session.frames++;
    }
    return session.frames;
  },

  // Debug helper: compose one tile band exactly like renderTiledChunk (no captions/encode)
  // and probe pixels - used by the tile geometry tests to prove the seams line up.
  async debugTiledFrame({ k, off, probes }) {
    const tiles = session.tiles;
    if (!tiles?.length || !session.tileStep) throw new Error('no tile snapshot prepared');
    const ctx = session.ctx;
    const W = session.width, H = session.height;
    const first = tiles[0];
    const s = Math.min(W / first.width, H / first.height);
    const dw = Math.max(2, Math.round(first.width * s));
    const dh = Math.max(2, Math.round(first.height * s));
    const dx = (W - dw) / 2, dy = (H - dh) / 2;
    const ratio = first.height / session.tileStep;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, W, H);
    const out = { canvasW: W, canvasH: H, tileW: null, tileH: null, bg: null };
    { const d = ctx.getImageData(1, 1, 1, 1).data; out.bg = `#${d[0].toString(16).padStart(2, '0')}${d[1].toString(16).padStart(2, '0')}${d[2].toString(16).padStart(2, '0')}`; }
    ctx.save();
    ctx.beginPath();
    ctx.rect(dx, dy, dw, dh);
    ctx.clip();
    const offB = off * ratio;
    out.tileW = first.width; out.tileH = first.height;
    ctx.drawImage(tiles[Math.min(k, tiles.length - 1)], dx, dy - offB * s, dw, dh);
    const nxt = tiles[k + 1];
    if (nxt) ctx.drawImage(nxt, dx, dy + (first.height - offB) * s, dw, dh);
    ctx.restore();
    for (const [x, y] of probes) {
      const d = ctx.getImageData(Math.round(dx + x * s), Math.round(dy + y * s), 1, 1).data;
      out[`${x},${y}`] = '#' + [d[0], d[1], d[2]].map((v) => v.toString(16).padStart(2, '0')).join('');
    }
    return out;
  },

  // --- slideshow ----------------------------------------------------------
  async prepareSlides(jpegs) {
    const slides = [];
    for (const b64 of jpegs) {
      try {
        const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
        const bmp = await decodeImage(bytes);
        slides.push(fitSlideCanvas(bmp));
        bmp.close();
      } catch (e) {
        log('skipping unreadable slide:', e?.message ?? e);
      }
    }
    session.slides = slides;
    return slides.length;
  },

  async renderSlides({ order, slideSeconds, fps, fade, total, kenburns, width, height, quality, bitrate }) {
    // Port of slides.py render_slideshow: solve the timeline, draw every frame here.
    const n = order.length;
    const D = Math.max(2, Math.round(slideSeconds * fps));
    const t0 = performance.now();
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d', { alpha: false });
    ctx.imageSmoothingQuality = 'high';
    const out = new Output({ format: new Mp4OutputFormat({ fastStart: 'in-memory' }), target: new BufferTarget() });
    const rate = bitrate ?? crfToBitrate(quality, width, height, fps);
    const vSrc = new VideoSampleSource({ codec: 'avc', keyFrameInterval: 2, quality: new Quality({ bitrate: rate }), latencyMode: 'quality' });
    out.addVideoTrack(vSrc, { frameRate: fps });
    const aSrc = new AudioBufferSource({ codec: 'aac', bitrate: 192000 });
    out.addAudioTrack(aSrc);
    session.output2 = out;
    session.vSrc2 = vSrc;
    session.aSrc2 = aSrc;
    await out.start();

    const perSlide = D / fps;
    const crossOff = perSlide - fade;
    const drawSlide = (idx, on, alpha = 1) => {
      const slide = session.slides[order[idx] % session.slides.length];
      const kind = kenburns ? MOTIONS[idx % MOTIONS.length] : null;
      const m = kind ? kbMotion(kind, on, D) : { z: 1, fx: 0.5, fy: 0.5 };
      const sw = SS_W / m.z, sh = SS_H / m.z;
      const sx = (SS_W - sw) * m.fx, sy = (SS_H - sh) * m.fy;
      if (alpha < 1) ctx.globalAlpha = alpha;
      ctx.drawImage(slide, sx, sy, sw, sh, 0, 0, width, height);
      if (alpha < 1) ctx.globalAlpha = 1;
    };

    const totalFrames = Math.round(total * fps);
    for (let f = 0; f < totalFrames; f++) {
      const t = f / fps;
      let idx = 0;
      let local = t;
      for (let k = 1; k < n; k++) {
        const off = k * crossOff;
        if (t >= off + fade) { idx = k; local = t - off - fade; }
        else if (t >= off) {
          // crossfade window between slide k-1 and k
          const frac = (t - off) / fade;
          drawSlide(k - 1, D - 1 - Math.round((1 - frac) * fade * fps), 1);
          drawSlide(k, Math.round(frac * fade * fps), frac);
          idx = -1;
          break;
        }
      }
      if (idx >= 0) drawSlide(idx, Math.round(local * fps), 1);
      drawCaptions(ctx, session.captions, t);
      drawAvatar(ctx, session.avatar, t, width, height);
      const sample = sampleFromCanvas(ctx, canvas, t, fps);
      await vSrc.add(sample);
      sample.close();
      if (f > 0 && f % 150 === 0) log(`slideshow frame ${f}/${totalFrames}`);
    }
    session.slidesFrames = totalFrames;
    return { frames: totalFrames, ms: performance.now() - t0 };
  },

  async finish() {
    if (session.output2) {
      // slideshow: attach audio, close, finalize
      const buf = session.buildAudioBuffer();
      await session.aSrc2.add(buf);
      await session.aSrc2.close();
      await session.vSrc2.close();
      await session.output2.finalize();
      session.bytes = new Uint8Array(session.output2.target.buffer);
      return { frames: session.slidesFrames, bytes: session.bytes.byteLength };
    }
    const info = await session.finish();
    return { frames: info.frames, bytes: session.bytes.byteLength };
  },

  async pullBytes(chunkSize = 4 * 1024 * 1024) {
    // Stream the finished MP4 back to the host in base64 chunks.
    const b = session.bytes;
    if (!b) throw new Error('nothing finalized yet');
    const pos = session.pullPos ?? 0;
    const end = Math.min(pos + chunkSize, b.length);
    let bin = '';
    for (let i = pos; i < end; i += 0x8000) {
      bin += String.fromCharCode.apply(null, b.subarray(i, Math.min(i + 0x8000, end)));
    }
    session.pullPos = end;
    return { b64: btoa(bin), done: end >= b.length, total: b.length };
  },

  // --- caption width measurement (back-ported into captions.makeMeasure) ---
  measureText(text, sizePx) {
    measureCtx ??= new OffscreenCanvas(8, 8).getContext('2d');
    measureCtx.font = `${sizePx}px "Anton", sans-serif`;
    return measureCtx.measureText(text).width;
  },

  // Debug helper: compose the current canvas at time t (no encoding) and return a PNG b64.
  async debugFrame(t) {
    const ctx = session.ctx;
    ctx.fillStyle = '#202020';
    ctx.fillRect(0, 0, session.width, session.height);
    drawCaptions(ctx, session.captions, t);
    drawAvatar(ctx, session.avatar, t, session.width, session.height);
    const blob = await session.canvas.convertToBlob({ type: 'image/png' });
    const buf = new Uint8Array(await blob.arrayBuffer());
    let bin = '';
    for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
    return btoa(bin);
  },

  // --- Instagram auto-heal (port of publish.py _ig_prepare) -----------------
  async heal(bytes, { maxDuration }) {
    const input = new Input({ formats: ALL_FORMATS, source: new BlobSource(new Blob([bytes])) });
    const video = await input.getPrimaryVideoTrack().catch(() => null);
    const audio = await input.getPrimaryAudioTrack().catch(() => null);
    const dur = input.format.duration ?? 0;
    const stats = video ? await video.computePacketStats(100) : null;
    const out = {
      duration: dur,
      hasAudio: !!audio,
      audioCodec: audio?.codec ?? null,
      audioSampleRate: audio?.sampleRate ?? null,
      videoCodec: video?.codec ?? null,
      width: video?.codedWidth ?? 0,
      height: video?.codedHeight ?? 0,
      fps: stats?.averagePacketRate ?? 0,
      needsTrim: dur > maxDuration,
      needsAudio: !audio,
    };
    await input.cancel();
    if (!out.needsTrim) return { ...out, fixed: null };
    if (out.needsAudio) {
      // Cannot synthesize audio here; the factory always muxes audio, so this only
      // happens with third-party files. Leave the file untouched and report it.
      return { ...out, fixed: null, blocked: 'missing audio track (reels need an audio stream)' };
    }
    const src = new Input({ formats: ALL_FORMATS, source: new BlobSource(new Blob([bytes])) });
    const output = new Output({ format: new Mp4OutputFormat({ fastStart: 'in-memory' }), target: new BufferTarget() });
    const conv = await Conversion.init({ input: src, output, trim: { start: 0, end: maxDuration } });
    await conv.execute();
    out.fixed = new Uint8Array(output.target.buffer);
    return out;
  },
};

let measureCtx = null;
