// captions.js - TikTok-style captions: group ASR words into short lines, emit an .ass file
// (compat output) and the structured line data the canvas burner draws per frame.
import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from './utils.js';

const FONTS = path.join(ROOT, 'fonts');
const ANTON_URL = 'https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf';
const SAFE = 0.86;   // captions never use more than this fraction of the frame width
export const POS = { high: 0.32, mid: 0.50, low: 0.68 };
const SENTENCE_END = '.!?…';

export class Word {
  constructor(text, start, end) {
    this.text = text;
    this.start = start;
    this.end = end;
  }
}

export async function ensureFont(name = 'Anton') {
  // Bundled Anton file (auto-downloaded once if missing); null for other fonts.
  if (name.toLowerCase() !== 'anton') return null;
  const p = path.join(FONTS, 'Anton-Regular.ttf');
  if (fs.existsSync(p)) return p;
  try {
    fs.mkdirSync(FONTS, { recursive: true });
    const res = await fetch(ANTON_URL, { signal: AbortSignal.timeout(20000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    fs.writeFileSync(p, Buffer.from(await res.arrayBuffer()));
    console.log('[captions] downloaded Anton font');
  } catch {
    console.log('warning: Anton font not found and could not be downloaded; captions use a fallback font. ' +
      'Put Anton-Regular.ttf in the fonts folder next to main.js.');
    return null;
  }
  return p;
}

// Width measurement with the real font when available (server-side canvas via the encoder page);
// falls back to a conservative estimate so lines never overflow.
export function makeMeasure(cfg, sizePx, outlinePx) {
  let font = null; // set by the encoder when a canvas measure is available
  const measure = {
    setFont(impl) { font = impl; },
    width(text) {
      const t = cfg.cap_caps ? text.toUpperCase() : text;
      const w = font ? font(t) : t.length * sizePx * 0.72;
      return w * 1.05 + 2 * outlinePx; // +5%: pop-in scale / kerning differences
    },
  };
  return measure;
}

export function groupWords(words, { maxWords = 4, maxChars = 0, fits = null, gapBreak = 0.6 } = {}) {
  const lines = [];
  let cur = [];
  const flush = () => {
    if (!cur.length) return;
    // extend each word to the next one's start so the highlight never blinks off
    lines.push(cur.map((w, i) => new Word(w.text, w.start,
      i + 1 < cur.length ? Math.max(cur[i + 1].start, w.start + 0.02) : w.end)));
    cur = [];
  };
  for (const w of words) {
    if (cur.length) {
      const chars = cur.reduce((a, x) => a + x.text.length, 0) + cur.length - 1;
      const joined = cur.map((x) => x.text).join(' ') + ' ' + w.text;
      if (w.start - cur[cur.length - 1].end > gapBreak || cur.length >= maxWords ||
          (maxChars && chars + 1 + w.text.length > maxChars) ||
          (fits && !fits(joined)) || SENTENCE_END.includes(cur[cur.length - 1].text.slice(-1))) {
        flush();
      }
    }
    cur.push(w);
  }
  flush();
  return lines;
}

function t(s) {
  const cs = Math.max(0, Math.round(s * 100));
  const h = Math.floor(cs / 360000);
  const r = cs % 360000;
  const m = Math.floor(r / 6000);
  const r2 = r % 6000;
  return `${h}:${String(m).padStart(2, '0')}:${String(Math.floor(r2 / 100)).padStart(2, '0')}.${String(r2 % 100).padStart(2, '0')}`;
}

function bgr(c) { // '#RRGGBB' -> 'BBGGRR'
  c = c.replace(/^#/, '');
  return (c.slice(4, 6) + c.slice(2, 4) + c.slice(0, 2)).toUpperCase();
}

function txt(t_, caps) {
  return (caps ? t_.toUpperCase() : t_).replaceAll('{', '(').replaceAll('}', ')');
}

// Structured caption data consumed by the canvas burner inside the encoder page.
// Mirrors the .ass output below (position, size, colors, pop/karaoke behaviour).
export function buildCaptionData(words, cfg, width, height, offset = 0.0) {
  const scale = Math.min(width, height) / 1080;
  const size = Math.round(cfg.cap_size * scale);
  const karaoke = cfg.cap_style === 'karaoke';
  const maxW = width * SAFE;
  const measure = makeMeasure(cfg, size, 4 * scale);
  const fits = (t_) => measure.width(t_) <= maxW;
  const lines = groupWords(words, {
    maxWords: cfg.cap_words, maxChars: cfg.cap_chars, fits,
  }).map((line) => {
    const lw = measure.width(line.map((w) => w.text).join(' '));
    return {
      start: line[0].start + offset,
      end: line[line.length - 1].end + offset,
      shrink: lw > maxW ? Math.max(8, Math.floor(size * maxW / lw)) : 0,
      words: line.map((w) => ({ text: txt(w.text, cfg.cap_caps), start: w.start + offset, end: w.end + offset })),
    };
  });
  return {
    lines,
    style: {
      font: cfg.cap_font,
      size,
      color: cfg.cap_color,
      outline: 4 * scale,
      shadow: 1.5 * scale,
      position: cfg.cap_position,
      posY: Math.round(height * POS[cfg.cap_position]),
      centerX: Math.round(width / 2),
      karaoke,
      caps: cfg.cap_caps,
    },
  };
}

export function buildAss(words, cfg, width, height, offset = 0.0) {
  const scale = Math.min(width, height) / 1080; // 9:16 and 1:1 -> 1.0, 16:9 short side -> 1.0
  const white = 'FFFFFF';
  const hi = bgr(cfg.cap_color);
  const karaoke = cfg.cap_style === 'karaoke';
  const primary = karaoke ? hi : white;
  const secondary = karaoke ? white : hi; // \kf sweeps secondary -> primary
  const head = `[Script Info]\nScriptType: v4.00+\nPlayResX: ${width}\nPlayResY: ${height}\n` +
    'ScaledBorderAndShadow: yes\nWrapStyle: 2\n\n[V4+ Styles]\n' +
    'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, ' +
    'Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, ' +
    'MarginR, MarginV, Encoding\n' +
    `Style: Caption,${cfg.cap_font},${Math.round(cfg.cap_size * scale)},&H00${primary},&H00${secondary},&H00000000,` +
    `&H80000000,0,0,0,0,100,100,0,0,1,${(4 * scale).toFixed(1)},${(1.5 * scale).toFixed(1)},5,40,40,0,1\n\n` +
    '[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n';
  const pos = `\\pos(${(width / 2).toFixed(0)},${(height * POS[cfg.cap_position]).toFixed(0)})`;
  const size = Math.round(cfg.cap_size * scale);
  const measure = makeMeasure(cfg, size, 4 * scale);
  const maxW = width * SAFE;
  const ev = [];
  for (const line of groupWords(words, { maxWords: cfg.cap_words, maxChars: cfg.cap_chars, fits: (t_) => measure.width(t_) <= maxW })) {
    const lw = measure.width(line.map((w) => w.text).join(' '));
    const posFs = lw > maxW ? `\\fs${Math.max(8, Math.floor(size * maxW / lw))}` : '';
    if (karaoke) {
      const body = line.map((w) =>
        `{\\kf${Math.max(1, Math.round((w.end - w.start) * 100))}}${txt(w.text, cfg.cap_caps)}`).join(' ');
      ev.push(`Dialogue: 0,${t(line[0].start + offset)},${t(line[line.length - 1].end + offset)},Caption,,0,0,0,,` +
        `{${pos}${posFs}\\fad(80,40)}${body}`);
      continue;
    }
    line.forEach((w, i) => { // pop: one event per word; the active word is recolored and pops in
      const fade = `\\fad(${i === 0 ? 80 : 0},${i === line.length - 1 ? 40 : 0})`;
      const body = line.map((o, j) =>
        j === i
          ? `{\\c&H${hi}&\\fscx115\\fscy115\\t(0,60,\\fscx100\\fscy100)}${txt(o.text, cfg.cap_caps)}{\\r}`
          : txt(o.text, cfg.cap_caps)).join(' ');
      ev.push(`Dialogue: 0,${t(w.start + offset)},${t(w.end + offset)},Caption,,0,0,0,,{${pos}${posFs}${fade}}${body}`);
    });
  }
  return head + ev.join('\n') + '\n';
}
