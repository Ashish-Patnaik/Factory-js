// gta.js - --gta mode: latest GTA 6 videos from source-gta.md -> viral 9:16 clip + AI voiceover.
//
// Features:
//   - Pulls latest GTA 6 uploads from @DubStepZz, @glp, @AFGuidesHD (or any URL in source-gta.md).
//   - Strict intro/outro guard: never captures the first 2 min or last 2 min of the video.
//   - Knows what happens at every timestamp via YouTube timestamped captions (instant) with
//     automatic fallback to on-device phonon-2-web ASR (zero ffmpeg via mediabunny).
//   - Combines timestamped transcript + YouTube Most-Replayed heatmap + chapters so Gemini
//     pins the most viral hook-worthy moment, explains it in persona.md voice, and writes
//     targeted GTA 6 hashtags for YouTube Shorts & Instagram Reels.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { Readable } from 'node:stream';
import { checkLlm, llm } from './gemini.js';
import { loadPersona, cleanScript, narration, WPS } from './speech.js';
import { buildCaptionData, buildAss } from './captions.js';
import { makeConfig } from './config.js';
import { loadHistory, saveHistory } from './agent.js';
import { EncoderHost } from './encoder/host.js';
import { readWav } from './tts.js';
import * as asr from './asr.js';
import { ROOT, die, probe } from './utils.js';

const GTA_OUT_DIR = path.join('output', 'gta');
const DEFAULT_TRIM_MARGIN = 120; // skip first 2 minutes & last 2 minutes (intro/outro guard)
const ASR_MAX_WINDOW_SEC = 360;  // cap fallback ASR scan to 6 min centered on hottest part (fast & low RAM)
const ASR_CHUNK_SEC = 60;        // transcribe in 60s chunks so WASM memory stays bounded

const YT_HEADERS = {
  'User-Agent':
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' +
    '(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
  'Accept-Language': 'en-US,en;q=0.9',
  Cookie: 'CONSENT=YES+cb.20210328-17-p0.en+FX+678',
};
const GTA6_RE = /\b(gta\s*(?:6|vi)|grand\s+theft\s+auto\s*(?:6|vi)|leonida|vice\s+city|jason\s+and\s+lucia)\b/i;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
function extractJson(raw) {
  const s = String(raw ?? '').trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
  const m = s.match(/\{[\s\S]*\}/);
  return JSON.parse(m ? m[0] : s);
}

export function extractVideoId(url) {
  const s = String(url ?? '').trim();
  for (const re of [
    /[?&]v=([A-Za-z0-9_-]{11})/,
    /youtu\.be\/([A-Za-z0-9_-]{11})/,
    /youtube\.com\/(?:shorts|embed|live)\/([A-Za-z0-9_-]{11})/,
  ]) {
    const m = s.match(re);
    if (m) return m[1];
  }
  return null;
}

const canonicalYtUrl = (id) => `https://www.youtube.com/watch?v=${id}`;

function parseDurationText(txt) {
  if (!txt) return 0;
  const parts = String(txt).trim().split(':').map((n) => parseInt(n, 10));
  return parts.some((n) => Number.isNaN(n)) ? 0 : parts.reduce((acc, v) => acc * 60 + v, 0);
}

function fmtTime(sec) {
  const s = Math.max(0, Math.round(sec));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

function decodeXml(s) {
  return String(s ?? '')
    .replaceAll('&amp;', '&').replaceAll('&lt;', '<').replaceAll('&gt;', '>')
    .replaceAll('&quot;', '"').replaceAll('&#39;', "'").replaceAll('&apos;', "'");
}

function extractBalancedJson(html, marker) {
  const idx = html.indexOf(marker);
  if (idx < 0) return null;
  const start = html.indexOf('{', idx + marker.length);
  if (start < 0) return null;
  let depth = 0, inStr = false, esc = false;
  for (let i = start; i < html.length; i++) {
    const ch = html[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === '\\') esc = true;
      else if (ch === '"') inStr = false;
    } else if (ch === '"') inStr = true;
    else if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) {
        try { return JSON.parse(html.slice(start, i + 1)); } catch { return null; }
      }
    }
  }
  return null;
}

function walkObjects(node, fn) {
  if (!node || typeof node !== 'object') return;
  if (Array.isArray(node)) { for (const x of node) walkObjects(x, fn); return; }
  fn(node);
  for (const v of Object.values(node)) walkObjects(v, fn);
}

function textOf(obj) {
  if (!obj) return '';
  if (typeof obj === 'string') return obj;
  if (obj.simpleText) return obj.simpleText;
  if (Array.isArray(obj.runs)) return obj.runs.map((r) => r.text ?? '').join('');
  return '';
}

// ---------------------------------------------------------------------------
// Intro/Outro Guard: never clip inside the first 2 min or last 2 min
// ---------------------------------------------------------------------------
export function computeSafeWindow(duration, clipLen, marginSec = DEFAULT_TRIM_MARGIN) {
  const need = Math.max(10, clipLen);
  if (duration <= need + 4) {
    return { start: 0, end: duration, marginApplied: 0, scaled: true };
  }
  if (duration >= marginSec * 2 + need) {
    return { start: marginSec, end: duration - marginSec, marginApplied: marginSec, scaled: false };
  }
  // Video is shorter than (4 min + clipLen): keep the middle ~70% safely away from intro/outro
  const maxMargin = Math.max(0, Math.floor((duration - need) / 2));
  const scaledMargin = Math.min(marginSec, maxMargin, Math.round(duration * 0.15));
  return { start: scaledMargin, end: duration - scaledMargin, marginApplied: scaledMargin, scaled: true };
}

// ---------------------------------------------------------------------------
// 1. Parse source-gta.md & scrape latest GTA 6 videos from channels
// ---------------------------------------------------------------------------
export function parseGtaSources(file = 'source-gta.md') {
  if (!fs.existsSync(file)) {
    die(`GTA sources file not found: ${file}. Create ${file} with YouTube channel URLs, ` +
      'or pass --gta-url <youtube-url> / --gta-file <local.mp4>.');
  }
  const channels = [];
  for (const rawLine of fs.readFileSync(file, 'utf8').split(/\r?\n/)) {
    const line = rawLine.replace(/\s+#.*$/, '').trim();
    if (!line || line.startsWith('#')) continue;
    for (const m of line.matchAll(/https?:\/\/[^\s)>\]]+/g)) channels.push(m[0]);
  }
  if (!channels.length) die(`No YouTube URLs found inside ${file}`);
  return channels;
}

async function fetchChannelVideos(channelUrl) {
  const clean = channelUrl.replace(/\/+$/, '').replace(/\/(videos|featured|streams|shorts)$/i, '');
  const label = clean.replace(/^https?:\/\/(www\.)?youtube\.com\//i, '');
  const res = await fetch(`${clean}/videos`, { headers: YT_HEADERS, signal: AbortSignal.timeout(20000) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const html = await res.text();
  const data =
    extractBalancedJson(html, 'var ytInitialData =') ??
    extractBalancedJson(html, 'window["ytInitialData"] =');
  const chanName = decodeXml((html.match(/<meta property="og:title" content="([^"]+)"/) ?? [])[1] ?? label);

  const out = [];
  const seen = new Set();
  if (data) {
    walkObjects(data, (obj) => {
      const r = obj.videoRenderer ?? obj.gridVideoRenderer;
      if (!r?.videoId || seen.has(r.videoId)) return;
      const badges = JSON.stringify(r.badges ?? r.thumbnailOverlays ?? '');
      if (/"style":"LIVE"|"BADGE_STYLE_TYPE_LIVE_NOW"|"UPCOMING"/i.test(badges)) return;
      const title = textOf(r.title).trim();
      if (!title) return;
      seen.add(r.videoId);
      out.push({
        id: r.videoId,
        url: canonicalYtUrl(r.videoId),
        title,
        channel: chanName,
        duration: parseDurationText(textOf(r.lengthText)),
        views: textOf(r.shortViewCountText ?? r.viewCountText).trim(),
        published: textOf(r.publishedTimeText).trim(),
      });
    });
  }

  if (!out.length) {
    const extId = (html.match(/"externalId":"(UC[A-Za-z0-9_-]+)"/) ?? [])[1];
    if (extId) {
      const rss = await (await fetch(`https://www.youtube.com/feeds/videos.xml?channel_id=${extId}`, {
        headers: YT_HEADERS, signal: AbortSignal.timeout(15000),
      })).text();
      for (const m of rss.matchAll(/<entry>([\s\S]*?)<\/entry>/g)) {
        const id = (m[1].match(/<yt:videoId>([^<]+)<\/yt:videoId>/) ?? [])[1];
        const title = decodeXml((m[1].match(/<title>([^<]+)<\/title>/) ?? [])[1] ?? '');
        if (id && title && !seen.has(id)) {
          seen.add(id);
          out.push({ id, url: canonicalYtUrl(id), title, channel: chanName, duration: 0, views: '', published: '' });
        }
      }
    }
  }
  return out.slice(0, 25);
}

async function discoverLatestGtaVideos(urls, history, { maxVideoSec = 2400, pickRandom = false } = {}) {
  const list = pickRandom ? [...urls].sort(() => Math.random() - 0.5) : [...urls];
  const candidates = [];
  const seen = new Set(history);

  for (const ch of list) {
    console.log(`[gta] checking latest videos: ${ch}`);
    let vids = [];
    try {
      vids = await fetchChannelVideos(ch);
    } catch (e) {
      console.log(`[gta] channel unreachable (${String(e?.message ?? e).split('\n')[0]}), skipping...`);
      continue;
    }
    for (const v of vids) {
      if (seen.has(v.url)) continue;
      if (!GTA6_RE.test(v.title)) continue;
      if (v.duration > 0 && (v.duration < 20 || v.duration > maxVideoSec)) continue;
      seen.add(v.url);
      candidates.push(v);
    }
    if (pickRandom && candidates.length >= 5) break;
  }
  return candidates;
}

// ---------------------------------------------------------------------------
// 2. yt-dlp stream + metadata (heatmap, chapters, subtitles) — no ffmpeg
// ---------------------------------------------------------------------------
export function findYtDlp(explicit = null) {
  const isWin = os.platform() === 'win32';
  const names = isWin ? ['yt-dlp.exe', 'yt-dlp'] : ['yt-dlp'];
  const isFile = (p) => { try { return fs.statSync(p).isFile(); } catch { return false; } };

  for (const cand of [explicit, process.env.YTDLP_PATH].filter(Boolean)) {
    if (isFile(cand)) return path.resolve(cand);
  }
  for (const root of [ROOT, process.cwd()]) {
    for (const sub of ['', 'bin', 'tools']) {
      for (const n of names) {
        const p = path.join(root, sub, n);
        if (isFile(p)) return path.resolve(p);
      }
    }
  }
  for (const dir of (process.env.PATH ?? '').split(path.delimiter).filter(Boolean)) {
    for (const n of names) {
      const p = path.join(dir, n);
      if (isFile(p)) return p;
    }
  }
  return null;
}

function runCmd(bin, args, { timeoutMs = 120000 } = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(bin, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    const out = [], err = [];
    proc.stdout.on('data', (c) => out.push(c));
    proc.stderr.on('data', (c) => err.push(c));
    const timer = setTimeout(() => {
      try { proc.kill('SIGKILL'); } catch { /* ignore */ }
      reject(new Error(`${path.basename(bin)} timed out after ${timeoutMs / 1000}s`));
    }, timeoutMs);
    proc.on('error', (e) => { clearTimeout(timer); reject(e); });
    proc.on('close', (code) => {
      clearTimeout(timer);
      if (code === 0) resolve(Buffer.concat(out).toString('utf8'));
      else reject(new Error(Buffer.concat(err).toString('utf8').trim().split('\n').slice(-1)[0] || `exit ${code}`));
    });
  });
}

async function resolveYtVideo(ytdlp, videoUrl) {
  const stdout = await runCmd(ytdlp, [
    '--dump-single-json', '--no-warnings', '--no-playlist', videoUrl,
  ], { timeoutMs: 120000 });
  const info = JSON.parse(stdout);
  const formats = Array.isArray(info.formats) ? info.formats : [];

  const isDirect = (f) => !String(f.protocol ?? '').includes('m3u8') && !/\.m3u8(\?|$)/i.test(String(f.url ?? ''));

  const h264 = formats
    .filter((f) => f.url && f.ext === 'mp4' && /^avc1/i.test(String(f.vcodec ?? '')) &&
      (f.height ?? 0) >= 360 && (f.height ?? 0) <= 1080)
    .sort((a, b) =>
      (Number(isDirect(b)) - Number(isDirect(a))) ||
      ((b.height ?? 0) - (a.height ?? 0)) ||
      ((b.fps ?? 0) - (a.fps ?? 0)) ||
      (Number(b.acodec && b.acodec !== 'none') - Number(a.acodec && a.acodec !== 'none')));

  const videoFormat = h264[0]
    ?? formats.find((f) => f.url && isDirect(f) && f.ext === 'mp4' && f.vcodec !== 'none')
    ?? formats.find((f) => f.url && f.ext === 'mp4' && f.vcodec !== 'none');
  if (!videoFormat) throw new Error('no H.264 MP4 stream available for this video');

  const hasMuxedAudio = videoFormat.acodec && videoFormat.acodec !== 'none';
  const audioFormat = hasMuxedAudio ? null : (formats
    .filter((f) => f.url && (f.ext === 'm4a' || f.ext === 'mp4') && f.vcodec === 'none' && f.acodec && f.acodec !== 'none')
    .sort((a, b) =>
      (Number(isDirect(b)) - Number(isDirect(a))) ||
      ((b.abr ?? 0) - (a.abr ?? 0)))[0] ?? null);

  return {
    title: info.title ?? '',
    channel: info.channel ?? info.uploader ?? '',
    duration: Number(info.duration ?? 0),
    description: String(info.description ?? '').slice(0, 2000),
    rawHeatmap: Array.isArray(info.heatmap) ? info.heatmap : [],
    rawChapters: Array.isArray(info.chapters) ? info.chapters : [],
    subtitles: info.subtitles ?? {},
    automaticCaptions: info.automatic_captions ?? {},
    videoFormat,
    audioFormat,
  };
}

async function downloadSingleStream(ytdlp, videoUrl, formatId, outPath) {
  await runCmd(ytdlp, [
    '-f', String(formatId), '--no-playlist', '--no-warnings', '-o', outPath, videoUrl,
  ], { timeoutMs: 300000 });
  if (!fs.existsSync(outPath)) throw new Error('yt-dlp did not create output file');
  return outPath;
}
async function downloadClipSection(ytdlp, videoUrl, formatSpec, outPath, startSec, endSec) {
  const s0 = Math.max(0, Math.floor(startSec));
  const s1 = Math.max(s0 + 2, Math.ceil(endSec));
  if (fs.existsSync(outPath)) fs.rmSync(outPath, { force: true });
  await runCmd(ytdlp, [
    '-f', String(formatSpec),
    '--download-sections', `*${s0}-${s1}`,
    '--merge-output-format', 'mp4',
    '--downloader-args', 'ffmpeg_i:-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5',
    '--no-playlist', '--no-warnings', '--no-part', '--force-overwrites',
    '-o', outPath,
    videoUrl,
  ], { timeoutMs: 180000 });
  if (!fs.existsSync(outPath)) throw new Error('yt-dlp did not create clip section file');
  return outPath;
}

// ---------------------------------------------------------------------------
// 3. Timestamped Transcript: YouTube Captions (fast) -> phonon-2-web ASR (fallback)
// ---------------------------------------------------------------------------
function groupSegmentsToBlocks(segments, safeStart, safeEnd, blockSec = 10) {
  // Filter strictly inside [safeStart, safeEnd] and pack into ~10s timestamped lines
  const clean = segments
    .filter((s) => s.text && s.end >= safeStart && s.start <= safeEnd)
    .sort((a, b) => a.start - b.start);
  if (!clean.length) return [];

  const blocks = [];
  let cur = null;
  for (const seg of clean) {
    const txt = seg.text.replace(/\s+/g, ' ').trim();
    if (!txt) continue;
    if (!cur || seg.start - cur.start >= blockSec || cur.text.length > 260) {
      if (cur) blocks.push(cur);
      cur = { start: Math.max(safeStart, seg.start), end: Math.min(safeEnd, seg.end), text: txt };
    } else {
      cur.end = Math.min(safeEnd, Math.max(cur.end, seg.end));
      cur.text += ' ' + txt;
    }
  }
  if (cur) blocks.push(cur);
  return blocks;
}

function parseVttTime(str) {
  const p = String(str).trim().split(':').map(parseFloat);
  if (p.length === 3) return p[0] * 3600 + p[1] * 60 + p[2];
  if (p.length === 2) return p[0] * 60 + p[1];
  return 0;
}

function parseVtt(vttText) {
  const segs = [];
  const lines = String(vttText ?? '').split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^(\d{1,2}:[\d:.]+)\s*-->\s*(\d{1,2}:[\d:.]+)/);
    if (!m) continue;
    const start = parseVttTime(m[1]);
    const end = parseVttTime(m[2]);
    const body = [];
    while (i + 1 < lines.length && lines[i + 1].trim()) {
      body.push(lines[++i].replace(/<[^>]+>/g, '').trim());
    }
    const text = decodeXml(body.join(' ')).replace(/\s+/g, ' ').trim();
    if (text && (!segs.length || segs[segs.length - 1].text !== text)) {
      segs.push({ start, end, text });
    }
  }
  return segs;
}

async function fetchYoutubeTranscript(subtitles, automaticCaptions, safeStart, safeEnd) {
  const pickTrack = (map) => {
    if (!map || typeof map !== 'object') return null;
    const keys = Object.keys(map);
    const langKey =
      ['en', 'en-orig', 'en-US', 'en-GB'].find((k) => Array.isArray(map[k]) && map[k].length) ??
      keys.find((k) => /^en\b/i.test(k) && Array.isArray(map[k]) && map[k].length);
    if (!langKey) return null;
    const formats = map[langKey];
    return formats.find((f) => f.ext === 'json3' && f.url) ??
      formats.find((f) => f.ext === 'vtt' && f.url) ?? null;
  };

  const track = pickTrack(subtitles) ?? pickTrack(automaticCaptions);
  if (!track) return null;

  try {
    const res = await fetch(track.url, { headers: YT_HEADERS, signal: AbortSignal.timeout(15000) });
    if (!res.ok) return null;
    let segs = [];
    if (track.ext === 'json3') {
      const j = await res.json();
      for (const ev of j?.events ?? []) {
        if (!Array.isArray(ev.segs) || ev.tStartMs == null) continue;
        const text = ev.segs.map((s) => s.utf8 ?? '').join('').replace(/\s+/g, ' ').trim();
        if (!text || text === '\n') continue;
        const start = ev.tStartMs / 1000;
        const end = start + (ev.dDurationMs ?? 2000) / 1000;
        segs.push({ start, end, text });
      }
    } else {
      segs = parseVtt(await res.text());
    }
    const blocks = groupSegmentsToBlocks(segs, safeStart, safeEnd);
    const wordCount = blocks.reduce((n, b) => n + b.text.split(/\s+/).length, 0);
    if (wordCount < 15) return null;
    console.log(`[gta] extracted YouTube transcript (${blocks.length} blocks, ~${wordCount} words in safe window ${fmtTime(safeStart)}-${fmtTime(safeEnd)})`);
    return blocks;
  } catch (e) {
    console.log(`[gta] YouTube captions unavailable (${String(e?.message ?? e).split('\n')[0]})`);
    return null;
  }
}

async function transcribeSafeWindowWithAsr(host, audioMediaUrl, safeStart, safeEnd, heatmap) {
  if (!asr.asrAvailable()) {
    console.log('[gta] phonon-2-web ASR not built; skipping audio transcription fallback');
    return null;
  }
  // Scan up to 6 minutes inside [safeStart, safeEnd], centered on the top heatmap peak if present
  const safeSpan = safeEnd - safeStart;
  let asrStart = safeStart;
  let asrEnd = safeEnd;
  if (safeSpan > ASR_MAX_WINDOW_SEC) {
    const anchor = heatmap?.[0]?.start ?? (safeStart + safeSpan * 0.5);
    asrStart = Math.max(safeStart, Math.min(safeEnd - ASR_MAX_WINDOW_SEC, anchor - ASR_MAX_WINDOW_SEC / 2));
    asrEnd = Math.min(safeEnd, asrStart + ASR_MAX_WINDOW_SEC);
  }
  console.log(`[gta] no YouTube captions — running on-device ASR over ${fmtTime(asrStart)}-${fmtTime(asrEnd)} (${Math.round(asrEnd - asrStart)}s)...`);

  const info = await host.rpc('extractAudio16k', { url: audioMediaUrl, startSec: asrStart, endSec: asrEnd });
  if (!info?.samples) {
    console.log('[gta] source has no decodable audio track for ASR');
    return null;
  }
  const chunks = [];
  for (;;) {
    const { b64, done } = await host.rpc('pullExtractedAudio');
    if (b64) chunks.push(Buffer.from(b64, 'base64'));
    if (done) break;
  }
  const rawBuf = Buffer.concat(chunks);
  const pcm16 = new Int16Array(rawBuf.buffer, rawBuf.byteOffset, Math.floor(rawBuf.byteLength / 2));
  const f32 = new Float32Array(pcm16.length);
  for (let i = 0; i < pcm16.length; i++) f32[i] = pcm16[i] / 32768;

  // Transcribe in 60s chunks for safe WASM memory usage
  const SR = 16000;
  const totalSec = f32.length / SR;
  const segs = [];
  for (let off = 0; off < totalSec; off += ASR_CHUNK_SEC) {
    const slice = f32.subarray(Math.floor(off * SR), Math.floor(Math.min(totalSec, off + ASR_CHUNK_SEC) * SR));
    if (slice.length < SR) break;
    const words = await asr.transcribe({ pcm: slice, sampleRate: SR });
    for (const w of words) {
      segs.push({ start: asrStart + off + w.start, end: asrStart + off + w.end, text: w.text });
    }
  }
  const blocks = groupSegmentsToBlocks(segs, safeStart, safeEnd);
  const wordCount = blocks.reduce((n, b) => n + b.text.split(/\s+/).length, 0);
  if (wordCount < 10) {
    console.log('[gta] ASR found little/no dialogue (pure ambient gameplay); using heatmap & chapters');
    return null;
  }
  console.log(`[gta] ASR transcript ready (${blocks.length} blocks, ~${wordCount} words)`);
  return blocks;
}

function formatTranscriptForPrompt(blocks, maxChars = 14000) {
  if (!blocks?.length) return '(no spoken dialogue in this gameplay video — rely on chapters, heatmap peaks & description)';
  // Evenly thin blocks if a long video has a huge transcript so the full safe window stays covered
  let picked = blocks;
  const approx = blocks.reduce((n, b) => n + b.text.length + 22, 0);
  if (approx > maxChars) {
    const keep = Math.max(12, Math.floor((blocks.length * maxChars) / approx));
    picked = Array.from({ length: keep }, (_, i) => blocks[Math.round((i * (blocks.length - 1)) / (keep - 1))]);
  }
  return picked
    .map((b) => `[${fmtTime(b.start)}] (${Math.round(b.start)}s) ${b.text}`)
    .join('\n')
    .slice(0, maxChars);
}

// ---------------------------------------------------------------------------
// 4. Local CORS + HTTP Range media server
// ---------------------------------------------------------------------------
async function createMediaServer() {
  const routes = new Map();
  const server = http.createServer(async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Range, Content-Type');
    res.setHeader('Access-Control-Expose-Headers', 'Content-Length, Content-Range, Accept-Ranges');
    if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

    const entry = routes.get(new URL(req.url, 'http://127.0.0.1').pathname);
    if (!entry) { res.writeHead(404); res.end('not found'); return; }

    if (entry.type === 'file') {
      const size = fs.statSync(entry.path).size;
      const mime = entry.mime ?? 'video/mp4';
      const range = req.headers.range;
      if (range) {
        const m = /bytes=(\d*)-(\d*)/.exec(range);
        let start = m?.[1] ? parseInt(m[1], 10) : 0;
        let end = m?.[2] ? parseInt(m[2], 10) : size - 1;
        if (!m?.[1] && m?.[2]) { start = Math.max(0, size - parseInt(m[2], 10)); end = size - 1; }
        end = Math.min(end, size - 1);
        res.writeHead(206, {
          'Content-Type': mime, 'Accept-Ranges': 'bytes',
          'Content-Range': `bytes ${start}-${end}/${size}`,
          'Content-Length': String(end - start + 1),
        });
        if (req.method === 'HEAD') { res.end(); return; }
        fs.createReadStream(entry.path, { start, end }).pipe(res);
      } else {
        res.writeHead(200, { 'Content-Type': mime, 'Accept-Ranges': 'bytes', 'Content-Length': String(size) });
        if (req.method === 'HEAD') { res.end(); return; }
        fs.createReadStream(entry.path).pipe(res);
      }
      return;
    }

    const ac = new AbortController();
    req.on('close', () => ac.abort());
    try {
      const headers = { ...(entry.headers ?? {}) };
      if (req.headers.range) headers.Range = req.headers.range;
      const upstream = await fetch(entry.url, {
        method: req.method === 'HEAD' ? 'HEAD' : 'GET', headers, signal: ac.signal,
      });
      const outHeaders = { 'Accept-Ranges': 'bytes' };
      for (const h of ['content-type', 'content-length', 'content-range']) {
        const v = upstream.headers.get(h);
        if (v) outHeaders[h] = v;
      }
      res.writeHead(upstream.status, outHeaders);
      if (req.method === 'HEAD' || !upstream.body) { res.end(); return; }
      const stream = Readable.fromWeb(upstream.body);
      stream.on('error', () => { try { res.end(); } catch { /* ignore */ } });
      stream.pipe(res);
    } catch {
      if (!res.headersSent) res.writeHead(502);
      try { res.end(); } catch { /* ignore */ }
    }
  });

  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  return {
    baseUrl: `http://127.0.0.1:${server.address().port}`,
    setFile(route, filePath, mime = 'video/mp4') { routes.set(route, { type: 'file', path: filePath, mime }); },
    setProxy(route, url, headers = {}) { routes.set(route, { type: 'proxy', url, headers }); },
    clear(route) { routes.delete(route); },
    close() { return new Promise((r) => server.close(() => r())); },
  };
}

// ---------------------------------------------------------------------------
// 5. Gemini selection & grounded clip + script + hashtags planning
// ---------------------------------------------------------------------------
async function aiPickVideo(cfg, persona, candidates) {
  const listing = candidates.slice(0, 35).map((c, i) =>
    `${i}. [${c.title}] (${c.channel} | ${c.duration ? fmtTime(c.duration) : '?'} | ${c.published || 'recent'} | ${c.views || ''})`).join('\n');

  const prompt = `You are a viral gaming Shorts director scouting the coolest latest GTA 6 gameplay video to clip into a 9:16 Short/Reel.

Creator Persona:
${persona}

Latest GTA 6 Videos From Our Trusted Channels:
${listing}

Pick the single BEST, most viral, hook-worthy GTA 6 video (insane details, physics, chases, NPCs, Vice City/Leonida showcase, stunts, or hot new gameplay). Prefer newer uploads.

Return ONLY valid JSON:
{"pick": 0, "reason": "One sentence explaining why this makes the best viral GTA 6 Short."}`;

  const data = extractJson(await llm(cfg, prompt));
  let idx = Number(data.pick);
  if (!Number.isInteger(idx) || idx < 0 || idx >= candidates.length) idx = 0;
  console.log(`[gta] AI pick: "${candidates[idx].title}" — ${data.reason ?? 'viral GTA 6 hook'}`);
  return candidates[idx];
}

async function aiPlanClipAndPackage(cfg, persona, meta, transcriptBlocks, safe, videoUrl) {
  const maxWords = Math.floor(cfg.script_seconds * WPS);
  const maxAllowedStart = Math.max(safe.start, safe.end - cfg.script_seconds);
  const extra = cfg.prompt ? `\nCreator Extra Instructions: ${cfg.prompt}\n` : '';

  const heatmapBlock = meta.heatmap?.length
    ? meta.heatmap.map((h) => `- ${fmtTime(h.start)}-${fmtTime(h.end)} (${h.start}s, replay score ${h.score})`).join('\n')
    : '(none)';
  const chaptersBlock = meta.chapters?.length
    ? meta.chapters.map((c) => `- ${fmtTime(c.start)} (${c.start}s): ${c.title}`).join('\n')
    : '(none)';
  const transcriptText = formatTranscriptForPrompt(transcriptBlocks);

  const prompt = `You are a viral gaming short-form creator turning a GTA 6 video moment into a trendy, hook-worthy 9:16 YouTube Short & Instagram Reel with AI voiceover.

=== CREATOR PERSONA & VOICE (FOLLOW STRICTLY) ===
${persona}

=== SOURCE GTA 6 VIDEO ===
Title: ${meta.title}
Channel: ${meta.channel || 'unknown'}
Total Duration: ${meta.duration.toFixed(1)}s
URL: ${videoUrl}
Description:
${meta.description || '(none)'}

=== ALLOWED CLIP WINDOW (STRICT — DO NOT PICK INTRO/OUTRO) ===
clip_start MUST be between ${Math.round(safe.start)}s (${fmtTime(safe.start)}) and ${Math.round(maxAllowedStart)}s (${fmtTime(maxAllowedStart)}).
(The first ${safe.marginApplied}s and last ${safe.marginApplied}s are strictly excluded to avoid intros & outros.)

=== CHAPTERS INSIDE SAFE WINDOW ===
${chaptersBlock}

=== MOST-REPLAYED PEAKS INSIDE SAFE WINDOW ===
${heatmapBlock}

=== TIMESTAMPED TRANSCRIPT INSIDE SAFE WINDOW ===
${transcriptText}

=== INSTRUCTIONS ===
1. Read the timestamped transcript, chapters, and Most-Replayed peaks above to find the SINGLE most viral, cool, or surprising GTA 6 moment inside [${Math.round(safe.start)}s, ${Math.round(maxAllowedStart)}s].
2. Set clip_start (seconds) to the exact start of that moment (MUST be >= ${Math.round(safe.start)} and <= ${Math.round(maxAllowedStart)}).
3. Write a punchy spoken AI voiceover script explaining WHAT ACTUALLY HAPPENS at that exact timestamp:
   - Sentence 1 MUST be a scroll-stopping hook ("Rockstar really hid this inside GTA 6...", "Wait until you hear what happens right here in GTA 6...", "Nobody noticed this insane GTA 6 moment...", etc.).
   - Ground the narration in the real details from that timestamp in the transcript/chapters.
   - Plain spoken English ONLY for TTS: NO markdown, NO emojis, NO hashtags, NO stage directions, NEVER read URLs aloud.
   - HARD LENGTH CAP: at most ${maxWords} words (under ${cfg.script_seconds} seconds when spoken, target ${Math.floor(maxWords * 0.85)} words).
   - End with a short comment-driving question.${extra}
4. Generate viral titles, descriptions, and accurate, high-reach GTA 6 hashtags for YouTube Shorts AND Instagram Reels.

Return ONLY valid JSON:
{
  "slug": "short_kebab_case_filename_slug",
  "clip_start": ${Math.round(safe.start)},
  "clip_reason": "What happens at this exact timestamp and why it hooks viewers",
  "script": "The plain spoken voiceover script under ${maxWords} words...",
  "youtube": {
    "title": "Viral GTA 6 curiosity-gap title under 70 chars",
    "description": "2-3 punchy lines matching the persona. Gameplay credit: ${meta.channel || 'source'} (${videoUrl})",
    "tags": ["gta 6", "gta 6 gameplay", "gta vi", "rockstar games", "gta 6 details", "gta 6 clips", "vice city"],
    "hashtags": ["#Shorts", "#GTA6", "#GTAVI", "#GrandTheftAuto6", "#RockstarGames", "#Gaming"],
    "category": "gaming"
  },
  "instagram": {
    "title": "Scroll-stopping first-line GTA 6 hook",
    "description": "2-3 punchy lines about this GTA 6 moment + a question to spark comments.",
    "hashtags": ["#reels", "#gta6", "#gtavi", "#grandtheftauto", "#rockstargames", "#gaming", "#gtaclips", "#vicecity", "#gamer"]
  }
}`;

  const pkg = extractJson(await llm(cfg, prompt));
  pkg.script = cleanScript(pkg.script ?? '', maxWords);
  const yt = pkg.youtube ?? {};
  if (!Array.isArray(yt.tags) || !yt.tags.length) {
    yt.tags = (yt.hashtags ?? []).map((h) => String(h).replace(/^#/, ''));
  }
  yt.category = yt.category || 'gaming';
  pkg.youtube = yt;
  pkg.instagram = pkg.instagram ?? null;
  pkg.url = videoUrl;
  pkg.created_at = new Date().toISOString().replace('T', ' ').slice(0, 19);
  pkg.long_form = false;

  // Hard-clamp clip_start inside [safe.start, maxAllowedStart] so intro/outro can NEVER slip through
  const fallbackStart = meta.heatmap?.[0]?.start ?? Math.round((safe.start + maxAllowedStart) / 2);
  const rawStart = Number(pkg.clip_start);
  const clipStart = Number.isFinite(rawStart)
    ? Math.max(safe.start, Math.min(maxAllowedStart, rawStart))
    : Math.max(safe.start, Math.min(maxAllowedStart, fallbackStart));

  return { clipStart, clipReason: pkg.clip_reason ?? '', pkg };
}

// ---------------------------------------------------------------------------
// 6. Main orchestrator — called from main.js when --gta is passed
// ---------------------------------------------------------------------------
export async function runGtaFlow(cfg, g, work) {
  checkLlm();
  fs.mkdirSync(GTA_OUT_DIR, { recursive: true });
  const persona = await loadPersona(cfg.persona_file);
  const fit = g.gta_fit || 'crop';
  const gameAudioGain = g.gta_game_audio ?? 0.12;
  const maxVideoSec = g.gta_max_video ?? 2400;
  const trimMargin = g.gta_trim_margin ?? DEFAULT_TRIM_MARGIN;

  console.log(`[gta] mode: 9:16 GTA 6 clip | fit: ${fit} | trim guard: ±${trimMargin}s | game audio: ${(gameAudioGain * 100).toFixed(0)}% | publish: ${g.publish}`);

  let ytdlp = null;
  if (!g.gta_file) {
    ytdlp = findYtDlp(g.yt_dlp_path);
    if (!ytdlp) {
      die(
        'yt-dlp not found (needed to pull YouTube MP4 streams without ffmpeg).\n' +
        '  Install once:\n' +
        '    Windows: winget install yt-dlp.yt-dlp   (or drop yt-dlp.exe next to main.js)\n' +
        '    macOS  : brew install yt-dlp\n' +
        '    Linux  : pip install yt-dlp\n' +
        '  Or clip a local file directly:  --gta --gta-file <path/to/video.mp4>'
      );
    }
  }

  // 1) Pick the video & 2) Resolve stream + metadata (with automatic candidate failover)
  const history = loadHistory();
  let chosen = null;
  let resolved = null;
  let meta = { title: '', channel: '', duration: 0, description: '', heatmap: [], chapters: [] };
  let videoFormat = null, audioFormat = null;

  if (g.gta_file) {
    const abs = path.resolve(g.gta_file);
    if (!fs.existsSync(abs)) die(`--gta-file not found: ${abs}`);
    chosen = { localFile: abs, url: `file://${abs}`, id: 'local', title: path.basename(abs, path.extname(abs)), channel: 'local', duration: 0 };
    const { probeMedia } = await import('./utils.js');
    const pm = await probeMedia(chosen.localFile);
    meta = { title: chosen.title, channel: chosen.channel, duration: pm.duration || 60, description: '', heatmap: [], chapters: [] };
  } else if (g.gta_url) {
    const id = extractVideoId(g.gta_url);
    if (!id) die(`--gta-url is not a valid YouTube video URL: ${g.gta_url}`);
    chosen = { id, url: canonicalYtUrl(id), title: g.gta_url, channel: '', duration: 0 };
    console.log(`[gta] fetching stream, captions & highlight metadata for ${chosen.url}`);
    resolved = await resolveYtVideo(ytdlp, chosen.url);
    meta = { title: resolved.title, channel: resolved.channel, duration: resolved.duration, description: resolved.description, heatmap: [], chapters: [] };
    videoFormat = resolved.videoFormat;
    audioFormat = resolved.audioFormat;
  } else {
    const srcFile = g.gta_sources || 'source-gta.md';
    const channels = parseGtaSources(srcFile);
    const candidates = await discoverLatestGtaVideos(channels, history, { maxVideoSec, pickRandom: Boolean(g.random) });
    if (!candidates.length) {
      die(`no fresh GTA 6 videos found in ${srcFile} (or all latest uploads are already in output/history.json).`);
    }
    console.log(`[gta] found ${candidates.length} fresh GTA 6 video(s) across channels`);

    const topPick = await aiPickVideo(cfg, persona, candidates);
    const queue = [topPick, ...candidates.filter((c) => c.id !== topPick.id)].slice(0, 4);

    let lastErr = null;
    for (const cand of queue) {
      console.log(`[gta] fetching stream, captions & highlight metadata for ${cand.url}`);
      try {
        resolved = await resolveYtVideo(ytdlp, cand.url);
        chosen = cand;
        meta = { title: resolved.title, channel: resolved.channel, duration: resolved.duration, description: resolved.description, heatmap: [], chapters: [] };
        videoFormat = resolved.videoFormat;
        audioFormat = resolved.audioFormat;
        break;
      } catch (e) {
        lastErr = e;
        console.log(`[gta] failed to resolve ${cand.url} (${String(e?.message ?? e).split('\n')[0]}), trying next candidate...`);
      }
    }
    if (!resolved) throw lastErr ?? new Error('could not resolve any candidate GTA 6 video');
  }

  // 3) Compute safe clip window (skips first 2 min & last 2 min)
  const safe = computeSafeWindow(meta.duration, cfg.script_seconds, trimMargin);
  console.log(
    `[gta] safe clip window: ${fmtTime(safe.start)} -> ${fmtTime(safe.end)} of ${fmtTime(meta.duration)}` +
    (safe.scaled ? ` (video shorter than ${fmtTime(trimMargin * 2 + cfg.script_seconds)}; scaled margin to ±${safe.marginApplied}s)` : ' (first 2m & last 2m excluded)')
  );

  if (resolved) {
    // Keep only heatmap peaks and chapters strictly inside [safe.start, safe.end]
    meta.heatmap = resolved.rawHeatmap
      .filter((h) => h.start_time >= safe.start && h.start_time <= safe.end - Math.min(20, cfg.script_seconds))
      .sort((a, b) => (b.value ?? 0) - (a.value ?? 0))
      .slice(0, 3)
      .map((h) => ({ start: Math.round(h.start_time), end: Math.round(h.end_time), score: Number((h.value ?? 0).toFixed(2)) }));

    meta.chapters = resolved.rawChapters
      .filter((c) => c.end_time >= safe.start && c.start_time <= safe.end)
      .slice(0, 25)
      .map((c) => ({ start: Math.max(safe.start, Math.round(c.start_time)), end: Math.min(safe.end, Math.round(c.end_time)), title: c.title }));

    if (meta.heatmap.length) {
      console.log(`[gta] safe-window Most-Replayed peaks: ${meta.heatmap.map((h) => fmtTime(h.start)).join(', ')}`);
    }
  }

  // Open local media proxy + encoder host once (reused for ASR fallback AND 9:16 render)
  const server = await createMediaServer();
  const host = await new EncoderHost().launch({ channel: cfg.encoder_channel ?? null });
  let hasSeparateAudio = false;
  let downloadedVideo = false;
  let downloadedAudio = false;

  try {
    if (chosen.localFile) {
      server.setFile('/video', chosen.localFile, 'video/mp4');
    }

    // 4) Extract timestamped transcript inside [safe.start, safe.end]
    //    Path A: YouTube subtitles / auto-captions (instant)
    //    Path B: On-device phonon-2-web ASR fallback on a LOCAL file (never proxy googlevideo.com)
    let transcriptBlocks = resolved
      ? await fetchYoutubeTranscript(resolved.subtitles, resolved.automaticCaptions, safe.start, safe.end)
      : null;

    if (!transcriptBlocks && !g.gta_no_asr) {
      try {
        let audioRoute = null;
        if (chosen.localFile) {
          audioRoute = `${server.baseUrl}/video`;
        } else if (audioFormat) {
          console.log('[gta] downloading M4A audio locally for ASR (avoids YouTube stream proxy drops)...');
          const dlAudio = path.join(work, `asr_${chosen.id}.m4a`);
          await downloadSingleStream(ytdlp, chosen.url, audioFormat.format_id, dlAudio);
          server.setFile('/audio', dlAudio, 'audio/mp4');
          hasSeparateAudio = true;
          downloadedAudio = true;
          audioRoute = `${server.baseUrl}/audio`;
        } else if (videoFormat) {
          // Muxed-only video: download just the 6-min ASR window locally
          const safeSpan = safe.end - safe.start;
          const anchor = meta.heatmap?.[0]?.start ?? (safe.start + safeSpan * 0.5);
          const wStart = safeSpan > ASR_MAX_WINDOW_SEC
            ? Math.max(safe.start, Math.min(safe.end - ASR_MAX_WINDOW_SEC, anchor - ASR_MAX_WINDOW_SEC / 2))
            : safe.start;
          const wEnd = Math.min(safe.end, wStart + ASR_MAX_WINDOW_SEC);
          console.log(`[gta] downloading ${Math.round(wEnd - wStart)}s local audio slice for ASR...`);
          const dlSlice = path.join(work, `asr_slice_${chosen.id}.mp4`);
          await downloadClipSection(ytdlp, chosen.url, videoFormat.format_id, dlSlice, wStart, wEnd);
          server.setFile('/audio', dlSlice, 'video/mp4');
          // Local slice starts at 0s -> run ASR over [0, wEnd - wStart], then shift timestamps back by +wStart
          const rawBlocks = await transcribeSafeWindowWithAsr(host, `${server.baseUrl}/audio`, 0, wEnd - wStart, []);
          transcriptBlocks = rawBlocks?.map((b) => ({ start: b.start + wStart, end: b.end + wStart, text: b.text })) ?? null;
        }

        if (audioRoute) {
          transcriptBlocks = await transcribeSafeWindowWithAsr(host, audioRoute, safe.start, safe.end, meta.heatmap);
        }
      } catch (asrErr) {
        console.log(`[gta] ASR fallback skipped (${String(asrErr?.message ?? asrErr).split('\n')[0]})`);
      }
    }

    // 5) Gemini picks exact clip_start inside safe window + writes grounded voiceover script + hashtags
    console.log('[gta] planning viral clip timestamp, AI voiceover script & hashtags...');
    const plan = await aiPlanClipAndPackage(cfg, persona, meta, transcriptBlocks, safe, chosen.url);
    const { pkg } = plan;
    const slug = (pkg.slug || `gta6_${chosen.id || Date.now()}`)
      .replace(/[^\w.-]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 50);
    const outPath = g.output || path.join(GTA_OUT_DIR, `${slug}.mp4`);
    const metaPath = outPath.replace(/\.mp4$/i, '') + '.meta.json';
    fs.mkdirSync(path.dirname(path.resolve(outPath)) || '.', { recursive: true });

    const requestedStart = g.gta_start !== undefined ? g.gta_start : plan.clipStart;
    if (plan.clipReason) console.log(`[gta] clip hook @ ${fmtTime(requestedStart)}: ${plan.clipReason}`);

    if (cfg.script_only) {
      fs.writeFileSync(outPath.replace(/\.mp4$/i, '') + '.txt', pkg.script + '\n');
      fs.writeFileSync(metaPath, JSON.stringify({ ...pkg, clip_start: requestedStart, safe_window: safe }, null, 2));
      console.log(`[gta] --script-only saved to ${outPath.replace(/\.mp4$/i, '')}.txt`);
      console.log(`[script] ${pkg.script}`);
      return;
    }

    // 6) Synthesize TTS voiceover + ASR word timestamps for TikTok captions
    const wav = path.join(work, 'speech.wav');
    const vcfg = makeConfig({
      ...cfg, url: '', width: 1080, height: 1920,
      narrate: false, text: pkg.script, start_delay: 0, pause: 0,
    });
    const speech = await narration(vcfg, wav);
    let secs = speech.secs;
    const reelMax = Number(process.env.REEL_MAX_SECONDS ?? '89');
    if (secs > reelMax) {
      console.log(`note: voice-over is ${secs.toFixed(1)}s; capping clip+audio at ${reelMax}s (IG Reel API limit)`);
      secs = reelMax;
    }

    // Hard-lock [clipStart, clipEnd] strictly inside [safe.start, safe.end] (unless --gta-start overrode manually)
    const minBound = g.gta_start !== undefined ? 0 : safe.start;
    const maxBound = g.gta_start !== undefined ? meta.duration : safe.end;
    const clipLen = Math.min(secs, Math.max(2, maxBound - minBound));
    const clipStart = Math.max(minBound, Math.min(requestedStart, Math.max(minBound, maxBound - clipLen)));
    const clipEnd = Math.min(maxBound, clipStart + clipLen);
    secs = clipLen;
    console.log(`[gta] final clip: ${fmtTime(clipStart)} -> ${fmtTime(clipEnd)} (${clipLen.toFixed(1)}s, strictly inside safe window)`);

    const base = outPath.replace(/\.mp4$/i, '');
    if (cfg.keep_files) {
      fs.writeFileSync(base + '.txt', speech.script + '\n');
      fs.copyFileSync(wav, base + '.wav');
    }
    let assData = null;
    if (vcfg.captions && speech.words?.length) {
      assData = buildCaptionData(speech.words, vcfg, 1080, 1920, 0);
      if (cfg.keep_files) fs.writeFileSync(path.join(work, 'captions.ass'), buildAss(speech.words, vcfg, 1080, 1920, 0));
    }

    // 7) Render 1080x1920 9:16 MP4 via mediabunny + WebCodecs
    await host.initEncoder({
      width: 1080, height: 1920, fps: cfg.fps, quality: cfg.quality, bitrate: cfg.bitrate ?? null,
    });
    const fontPath = path.join(ROOT, 'fonts', 'Anton-Regular.ttf');
    if (assData && fs.existsSync(fontPath)) await host.setFont(fs.readFileSync(fontPath));
    if (assData) await host.setCaptions(assData);
    if (cfg.avatar_file) {
      console.log(`[avatar] looping ${path.basename(cfg.avatar_file)} in the ${cfg.avatar_corner} corner`);
      await host.setAvatar(fs.readFileSync(cfg.avatar_file), cfg.avatar_corner, cfg.avatar_size);
    }
    const { pcm, sampleRate } = readWav(wav);
    await host.deliverAudio(pcm, { sampleRate, audioDelay: 0, totalDuration: secs });

    let renderStart = clipStart;
    let renderEnd = clipEnd;
    let useSeparateAudio = hasSeparateAudio && gameAudioGain > 0;

    if (!chosen.localFile && videoFormat && !downloadedVideo) {
      const { probeMedia } = await import('./utils.js');
      const dlClip = path.join(work, `clip_${chosen.id}.mp4`);
      const formatSpec = (audioFormat && gameAudioGain > 0)
        ? `${videoFormat.format_id}+${audioFormat.format_id}`
        : String(videoFormat.format_id);
      const sliceEnd = meta.duration > 0 ? Math.min(meta.duration, clipEnd + 5) : clipEnd + 5;

      let slicedOk = false;
      try {
        console.log(`[gta] downloading local ${Math.ceil(sliceEnd - clipStart)}s clip slice (${fmtTime(clipStart)} -> ${fmtTime(sliceEnd)})...`);
        await downloadClipSection(ytdlp, chosen.url, formatSpec, dlClip, clipStart, sliceEnd);
        const pm = await probeMedia(dlClip);
        if (pm?.video && pm.duration >= Math.max(2, clipLen - 2)) {
          server.setFile('/video', dlClip, 'video/mp4');
          downloadedVideo = true;
          renderStart = 0;
          renderEnd = Math.min(clipLen, pm.duration);
          useSeparateAudio = false; // audio is already muxed & synced inside dlClip
          slicedOk = true;
        }
      } catch (sliceErr) {
        console.log(`[gta] section download unavailable (${String(sliceErr?.message ?? sliceErr).split('\n')[0]}); falling back to full MP4 download...`);
      }

      if (!slicedOk) {
        const dlVideo = path.join(work, `src_${chosen.id}.mp4`);
        await downloadSingleStream(ytdlp, chosen.url, videoFormat.format_id, dlVideo);
        server.setFile('/video', dlVideo, 'video/mp4');
        downloadedVideo = true;
        if (audioFormat && gameAudioGain > 0 && !downloadedAudio) {
          const dlAudio = path.join(work, `src_${chosen.id}.m4a`);
          try {
            await downloadSingleStream(ytdlp, chosen.url, audioFormat.format_id, dlAudio);
            server.setFile('/audio', dlAudio, 'audio/mp4');
            hasSeparateAudio = true;
            downloadedAudio = true;
            useSeparateAudio = true;
          } catch {
            server.clear('/audio');
            hasSeparateAudio = false;
            useSeparateAudio = false;
          }
        }
      }
    }

    const renderArgs = () => ({
      videoUrl: `${server.baseUrl}/video`,
      audioUrl: useSeparateAudio ? `${server.baseUrl}/audio` : null,
      startSec: renderStart, endSec: renderEnd, fps: cfg.fps, width: 1080, height: 1920,
      quality: cfg.quality, bitrate: cfg.bitrate ?? null, fit, gameAudioGain,
    });

    console.log(`[gta] rendering 9:16 clip (${fit}) with captions...`);
    const info = await host.rpc('renderVideoClip', renderArgs());
    await host.rpc('finish');
    await host.pullToFile(outPath);
    console.log(`done: ${info.frames} frames in ${(info.ms / 1000).toFixed(1)}s`);
    await probe(outPath);

    // 8) Save metadata, publish, update history & cleanup
    pkg.clip_start = Number(clipStart.toFixed(2));
    pkg.clip_end = Number(clipEnd.toFixed(2));
    pkg.safe_window = { start: safe.start, end: safe.end };
    pkg.transcript_source = transcriptBlocks ? (resolved ? 'captions_or_asr' : 'asr') : 'metadata';
    pkg.fit = fit;
    pkg.source_title = meta.title;
    pkg.source_channel = meta.channel;
    fs.writeFileSync(metaPath, JSON.stringify(pkg, null, 2));
    console.log(`[gta] metadata saved to ${metaPath}`);
    console.log(`  YT Title : ${pkg.youtube.title}`);
    if (pkg.instagram) console.log(`  IG Hook  : ${pkg.instagram.title}`);

    const publish = await import('./publish.js');
    const errors = [];
    let publishedAny = false;

    if (g.publish === 'youtube' || g.publish === 'all') {
      try {
        const yt = pkg.youtube;
        await publish.uploadYoutubeShort(
          outPath, yt.title, yt.description, yt.hashtags ?? [], g.yt_privacy,
          yt.tags ?? null, yt.category ?? 'gaming',
        );
        publishedAny = true;
      } catch (e) {
        const msg = String(e?.message ?? e).split('\n')[0];
        console.log(`[youtube] ERROR: ${msg}`);
        errors.push(`YouTube: ${msg}`);
      }
    }
    if (g.publish === 'instagram' || g.publish === 'all') {
      try {
        const ig = pkg.instagram;
        await publish.uploadInstagramReel(outPath, ig.title, ig.description, ig.hashtags, true, null);
        publishedAny = true;
      } catch (e) {
        const msg = String(e?.message ?? e).split('\n')[0];
        console.log(`[instagram] ERROR: ${msg}`);
        errors.push(`Instagram: ${msg}`);
      }
    }

    if (!g.gta_file && (publishedAny || g.publish === 'none')) saveHistory(chosen.url);

    if (g.publish !== 'none' && publishedAny && !cfg.keep_files) {
      for (const fpath of [outPath, metaPath]) {
        if (fpath && fs.existsSync(fpath)) {
          fs.rmSync(fpath);
          console.log(`[cleanup] removed ${fpath}`);
        }
      }
    }
    if (errors.length) die(errors.join(' | '));
  } finally {
    await host.close();
    await server.close();
  }
}