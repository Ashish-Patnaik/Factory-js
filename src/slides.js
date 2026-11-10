// slides.js - --image slideshow renderer: narration stills -> 16:9 YouTube slideshow with
// crossfades and Ken Burns motion. All composition/encoding runs in the mediabunny encoder
// page (see encoder/page.js fitSlideCanvas/renderSlides); this module plans the timeline and
// drives the encoder session - the Node equivalent of slides.py without ffmpeg zoompan/xfade.
import fs from 'node:fs';
import path from 'node:path';

export const FADE = 0.8;      // crossfade seconds between slides
export const MAX_INPUTS = 100; // encoder input cap (a 10-min video at 6s/slide = 100)

export function expandOrder(slides, n) {
  // Pick n slides from the list: evenly spaced across the whole story when there are
  // more images than slots, cycled without consecutive repeats when there are fewer.
  if (!slides.length) return [];
  if (slides.length === n) return [...slides];
  if (slides.length > n) { // sample evenly so every part of the story stays represented
    if (n === 1) return [slides[0]];
    return Array.from({ length: n }, (_, i) => slides[Math.round(i * (slides.length - 1) / (n - 1))]);
  }
  const out = [];
  for (let i = 0; i < n; i++) out.push(slides[i % slides.length]); // plain cycle: no adjacent repeats
  return out;
}

export function planTimeline(totalS, n, fade = FADE) {
  // Solve per-slide on-screen time so the crossfaded chain lasts exactly total_s.
  // total = n*t - (n-1)*fade
  const t = (totalS + (n - 1) * fade) / n;
  return { t, fade };
}

export async function readSlideJpegsB64(slidePaths) {
  const out = [];
  for (const p of slidePaths) out.push((await fs.promises.readFile(p)).toString('base64'));
  return out;
}

// Renders the finished slideshow MP4 through the encoder host. totalS = exact target length.
export async function renderSlideshow(host, cfg, slidePaths, { assData = null, totalS, work }) {
  const { width, height, fps } = cfg;
  const slides = expandOrder(slidePaths, Math.min(MAX_INPUTS, Math.max(2, Math.round(totalS / Math.max(cfg.slide_seconds, 2.0)))));
  const n = slides.length;
  const { t, fade } = planTimeline(totalS, n);
  const D = Math.max(2, Math.round(t * fps));
  const dur = n * D / fps - (n - 1) * fade;
  console.log(`[image] ${n} slide(s) x ${(D / fps).toFixed(2)}s with ${fade.toFixed(1)}s crossfades -> ${dur.toFixed(1)}s ` +
    `(audio ${totalS.toFixed(1)}s) at ${width}x${height}`);

  if (assData) await host.setCaptions(assData);
  // slide images -> the encoder page pre-fits them to the 2880x1620 slide canvas (cover or blur-fill)
  const jpegs = await readSlideJpegsB64(slides);
  const fitted = await host.rpc('prepareSlides', jpegs);
  if (fitted < 2) throw new Error('--image found fewer than 2 usable images');
  const order = slides.map((_, i) => i);
  const info = await host.rpc('renderSlides', {
    order, slideSeconds: D / fps, fps, fade, total: totalS, kenburns: cfg.kenburns,
    width, height, quality: cfg.quality, bitrate: cfg.bitrate ?? null,
  });
  return { ...info, dur };
}
