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
// ---------------------------------------------------------------------------
// Ground-truth script alignment: keep the AI script's exact spelling & punctuation,
// and pull ONLY the word timestamps from ASR (handles 1:1, 1:2, 1:3, 2:1 & gaps).
// ---------------------------------------------------------------------------
const DIGIT_WORDS = {
  0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
  6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
  11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen',
  16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen', 20: 'twenty',
  30: 'thirty', 40: 'forty', 50: 'fifty', 60: 'sixty', 70: 'seventy',
  80: 'eighty', 90: 'ninety', 100: 'hundred', 1000: 'thousand',
  vi: 'six', iv: 'four', iii: 'three', ii: 'two',
};

function normWord(w) {
  const s = String(w ?? '').toLowerCase().replace(/[^a-z0-9]/g, '');
  return DIGIT_WORDS[s] ?? s;
}

function wordSim(a, b) {
  if (!a || !b) return 0;
  if (a === b) return 1;
  const la = a.length, lb = b.length;
  const maxL = Math.max(la, lb);
  // Levenshtein distance (two-row O(min) space)
  let prev = new Uint16Array(lb + 1);
  let curr = new Uint16Array(lb + 1);
  for (let j = 0; j <= lb; j++) prev[j] = j;
  for (let i = 1; i <= la; i++) {
    curr[0] = i;
    const ca = a.charCodeAt(i - 1);
    for (let j = 1; j <= lb; j++) {
      const cost = ca === b.charCodeAt(j - 1) ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
    }
    [prev, curr] = [curr, prev];
  }
  return 1 - prev[lb] / maxL;
}

export function alignWordsWithScript(script, asrWords, totalDuration = 0) {
  const scriptTokens = String(script ?? '')
    .trim()
    .split(/\s+/)
    .filter((t) => /[a-zA-Z0-9]/.test(t)); // drop stray standalone dashes/punctuation
  if (!scriptTokens.length || !asrWords?.length) return asrWords ?? [];

  const N = scriptTokens.length;
  const M = asrWords.length;
  const sNorm = scriptTokens.map(normWord);
  const aNorm = asrWords.map((w) => normWord(w.text));

  const cols = M + 1;
  const dp = new Float32Array((N + 1) * cols);
  const act = new Uint8Array((N + 1) * cols);
  // Action codes: 1=1:1, 2=skip script, 3=skip ASR, 4=1:2 (1 script -> 2 ASR),
  //               5=1:3 (1 script -> 3 ASR), 6=2:1 (2 script -> 1 ASR)
  const GAP = -0.85;

  for (let i = 1; i <= N; i++) { dp[i * cols] = i * GAP; act[i * cols] = 2; }
  for (let j = 1; j <= M; j++) { dp[j] = j * GAP; act[j] = 3; }

  for (let i = 1; i <= N; i++) {
    const sn = sNorm[i - 1];
    const row = i * cols;
    const prevRow = (i - 1) * cols;
    for (let j = 1; j <= M; j++) {
      const an = aNorm[j - 1];
      const sim11 = wordSim(sn, an);
      // 1:1 substitution penalty (-0.55) beats skip+insert (-1.70) so misspelled words
      // still lock onto their exact spoken timestamp.
      const s11 = sim11 === 1 ? 3.0 : sim11 >= 0.45 ? 3.0 * sim11 - 0.3 : -0.55;

      let best = dp[prevRow + (j - 1)] + s11;
      let bestAct = 1;

      const del = dp[prevRow + j] + GAP;
      if (del > best) { best = del; bestAct = 2; }

      const ins = dp[row + (j - 1)] + GAP;
      if (ins > best) { best = ins; bestAct = 3; }

      // 1 script word spoken as 2 ASR words (e.g. "AI" -> "a"+"i", "YouTube" -> "you"+"tube")
      if (j >= 2) {
        const sim12 = wordSim(sn, aNorm[j - 2] + an);
        if (sim12 >= 0.6) {
          const sc = dp[prevRow + (j - 2)] + 3.4 * sim12;
          if (sc > best) { best = sc; bestAct = 4; }
        }
      }
      // 1 script word spoken as 3 ASR words (e.g. "Nerdysics" -> "nerd"+"is"+"six", "GTA" -> "g"+"t"+"a")
      if (j >= 3) {
        const sim13 = wordSim(sn, aNorm[j - 3] + aNorm[j - 2] + an);
        if (sim13 >= 0.65) {
          const sc = dp[prevRow + (j - 3)] + 3.8 * sim13;
          if (sc > best) { best = sc; bestAct = 5; }
        }
      }
      // 2 script words merged into 1 ASR word (e.g. "GTA" + "6" -> "gta6")
      if (i >= 2) {
        const sim21 = wordSim(sNorm[i - 2] + sn, an);
        if (sim21 >= 0.6) {
          const sc = dp[(i - 2) * cols + (j - 1)] + 3.4 * sim21;
          if (sc > best) { best = sc; bestAct = 6; }
        }
      }

      dp[row + j] = best;
      act[row + j] = bestAct;
    }
  }

  // Backtrack optimal alignment
  const slots = new Array(N).fill(null);
  let i = N, j = M;
  while (i > 0 || j > 0) {
    const a = act[i * cols + j];
    if (a === 1) {
      slots[i - 1] = { start: asrWords[j - 1].start, end: asrWords[j - 1].end };
      i--; j--;
    } else if (a === 2) {
      slots[i - 1] = null;
      i--;
    } else if (a === 3) {
      j--;
    } else if (a === 4) {
      slots[i - 1] = { start: asrWords[j - 2].start, end: asrWords[j - 1].end };
      i--; j -= 2;
    } else if (a === 5) {
      slots[i - 1] = { start: asrWords[j - 3].start, end: asrWords[j - 1].end };
      i--; j -= 3;
    } else if (a === 6) {
      const s = asrWords[j - 1].start, e = asrWords[j - 1].end;
      const w1 = Math.max(1, sNorm[i - 2].length);
      const w2 = Math.max(1, sNorm[i - 1].length);
      const mid = s + (e - s) * (w1 / (w1 + w2));
      slots[i - 2] = { start: s, end: mid };
      slots[i - 1] = { start: mid, end: e };
      i -= 2; j--;
    } else {
      break;
    }
  }

  // Smoothly interpolate any script words ASR missed between their timed neighbors
  const maxEnd = totalDuration > 0 ? totalDuration : asrWords[M - 1].end;
  for (let k = 0; k < N; k++) {
    if (slots[k]) continue;
    let endIdx = k;
    while (endIdx + 1 < N && !slots[endIdx + 1]) endIdx++;
    const count = endIdx - k + 1;
    const t0 = k > 0 ? slots[k - 1].end : Math.max(0, asrWords[0].start);
    let t1 = endIdx + 1 < N ? slots[endIdx + 1].start : maxEnd;
    if (t1 <= t0 + 0.02) t1 = Math.min(maxEnd, t0 + 0.08 * count);

    const weights = [];
    let wSum = 0;
    for (let u = k; u <= endIdx; u++) {
      const w = Math.max(1, sNorm[u].length || scriptTokens[u].length);
      weights.push(w);
      wSum += w;
    }
    let cur = t0;
    for (let u = 0; u < count; u++) {
      const dur = Math.max(0.03, ((t1 - t0) * weights[u]) / wSum);
      slots[k + u] = { start: cur, end: cur + dur };
      cur += dur;
    }
    k = endIdx;
  }

  // Build final Word[] with exact ground-truth script text & monotonic timings
  const out = [];
  for (let k = 0; k < N; k++) {
    let { start, end } = slots[k];
    start = Math.max(0, start);
    if (k > 0 && start < out[k - 1].start) start = out[k - 1].start;
    end = Math.max(start + 0.03, end);
    if (totalDuration > 0) end = Math.min(end, totalDuration);
    out.push(new Word(scriptTokens[k], start, end));
  }
  return out;
}
