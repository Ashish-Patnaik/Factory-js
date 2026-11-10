// images.js - AI image sourcing for --image slideshows.
//
// The AI reads the narration script and picks concrete visual search terms, then images are
// pulled from multiple free sources automatically (no manual searching, no API keys needed
// for the default path):
//
//   wiki      Wikipedia + Wikimedia Commons  -> people, places, science/tech, historical photos
//   openverse Openverse (CC-licensed, aggregates Flickr etc.) -> general topics + stock-style shots
//   ddg       DuckDuckGo Images via duck-duck-scrape -> general web image search, no API key
//   stock     Pexels / Unsplash / Pixabay when an API key is present in the environment
//             (PEXELS_API_KEY / UNSPLASH_ACCESS_KEY / PIXABAY_API_KEY) -> b-roll & stock
//   imdb      IMDb posters + media galleries via the Obscura stealth browser -> movies / TV /
//             games / celebrity topics; silently skipped when Obscura is not installed
//
// Downloads are validated (real decodable image, minimum resolution, sane aspect ratio),
// deduplicated by content hash and saved as output-ordered slide_XXXX.jpg files.
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import { llm } from './gemini.js';
import { findObscura, startObscura, stopProcess } from './scrape.js';
import { chromium } from 'playwright';

const _UA_BROWSER = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36' };
const _UA_BOT = { 'User-Agent': 'AI-Video-Factory/2.0 (automated image research for video slideshows; nodejs)' };
const IMDB_HEADERS = { ..._UA_BROWSER, Referer: 'https://www.imdb.com/' };
const MIN_SHORT_SIDE = 480;   // reject tiny thumbnails
const MAX_RATIO = 2.8;        // reject extreme banners / skyscraper crops
const OVERFETCH = 1.8;        // search for more URLs than needed (some downloads fail)
const TIMEOUT = 20000;

async function get(url, { headers = {}, params = {} } = {}) {
  // GET with a UA fallback: Wikimedia endpoints 403 spoofed browser UAs and 429 on burst,
  // so the bot UA is tried first and both get one retry after a short backoff.
  const qs = new URLSearchParams(params).toString();
  const full = qs ? `${url}?${qs}` : url;
  let delay = 2000;
  for (let i = 0; i < 4; i++) {
    try {
      const res = await fetch(full, { headers: { ...[_UA_BOT, _UA_BROWSER][i % 2], ...headers }, signal: AbortSignal.timeout(TIMEOUT) });
      if (res.status !== 403 && res.status !== 429 || i === 3) return res;
      await new Promise((r) => setTimeout(r, delay));
      delay *= 2;
    } catch (e) {
      if (i === 3) throw e;
      await new Promise((r) => setTimeout(r, delay));
      delay *= 2;
    }
  }
}

const okDims = (w, h) => Boolean(w && h && Math.min(w, h) >= MIN_SHORT_SIDE && w / h <= MAX_RATIO && w / h >= 1 / MAX_RATIO);

// ----------------------------------------------------------------------------------
// AI search-term extraction
// ----------------------------------------------------------------------------------
const TERMS_PROMPT = `You are the visual research director for a {mins}-minute 16:9 YouTube documentary video.
Below is the narration script. Extract 10-16 concrete VISUAL search queries that will find strong,
relevant still images covering the whole story from start to end.
{hint}
Rules:
- Each term: 2-5 words, concrete and searchable (named people, named places, objects, technology, scenes).
- Order matters: early terms must match the opening topics, later terms the closing topics.
- kind must be one of: person, place, technology, concept, entertainment, stock
- entertainment = movies / TV / games / celebrities (searched on IMDb), stock = generic cinematic b-roll.

Return ONLY valid JSON: {{"terms": [{{"term": "...", "kind": "..."}}]}}

Script:
{script}`;

const STOPWORDS = new Set(('a an and are as at be been but by can could for from has have how i in into is it its ' +
  'like may might more most not of on or our out over said says should than that the their them then there ' +
  'these they this those through to too under up very was we were what when where which while who why will ' +
  'with would you your about after all also any because before being between both did do does during each ' +
  'either even every few first get got he her here him his if just make made many much need new no now off ' +
  'once only other same see seen she some still such take takes took two use used using want way well went ' +
  'one three four five video youtube today really actually going gonna thing things lot lots').split(' '));

export function fallbackTerms(script, limit = 12) {
  // No-LLM fallback: named entities + frequent meaningful words from the script.
  const words = script.match(/[A-Za-z][A-Za-z'-]+/g) ?? [];
  const caps = [];
  for (let i = 0; i < words.length - 1; i++) {
    const a = words[i], b = words[i + 1];
    if (a[0] === a[0].toUpperCase() && !STOPWORDS.has(a.toLowerCase()) && a.length > 2) {
      if (b[0] === b[0].toUpperCase() && !STOPWORDS.has(b.toLowerCase()) && b.length > 2) caps.push(`${a} ${b}`);
      else if (!STOPWORDS.has(a.toLowerCase())) caps.push(a);
    }
  }
  const freq = {};
  for (const w of words) {
    const lw = w.toLowerCase();
    if (STOPWORDS.has(lw) || lw.length < 4) continue;
    freq[lw] = (freq[lw] ?? 0) + 1;
  }
  const top = Object.entries(freq).sort((x, y) => y[1] - x[1]).slice(0, Math.floor(limit / 2)).map(([w]) => w);
  const terms = [];
  const seen = new Set();
  for (const t of [...caps, ...top]) {
    if (!seen.has(t.toLowerCase())) {
      seen.add(t.toLowerCase());
      terms.push(t);
    }
  }
  return terms.slice(0, limit).map((term) => ({ term, kind: 'concept' }));
}

export async function aiTerms(cfg, script) {
  // LLM picks the visual search terms; falls back to keyword extraction on any failure.
  try {
    const mins = Math.max(1, Math.round(script.split(/\s+/).length / 150));
    const hint = cfg.entertainment
      ? '\nThis video is about films/TV/anime/comics: prefer named titles, characters, actors, studios and iconic scenes, and mark those terms kind=entertainment.'
      : '';
    let raw = await llm(cfg, TERMS_PROMPT
      .replaceAll('{mins}', String(mins))
      .replaceAll('{hint}', hint)
      .replaceAll('{{"terms"', '{"terms"')
      .replaceAll('{script}', script.slice(0, 14000)));
    raw = raw.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
    const m = raw.match(/\{[\s\S]*\}/);
    const data = JSON.parse(m ? m[0] : raw);
    const terms = [];
    const seen = new Set();
    for (const t of data.terms ?? []) {
      const term = String(t.term ?? '').trim();
      let kind = String(t.kind ?? 'concept').trim().toLowerCase();
      if (!term || !/^[\w '&.,()-]{2,60}$/.test(term)) continue;
      if (!['person', 'place', 'technology', 'concept', 'entertainment', 'stock'].includes(kind)) kind = 'concept';
      if (seen.has(term.toLowerCase())) continue;
      seen.add(term.toLowerCase());
      terms.push({ term, kind });
    }
    if (terms.length) {
      console.log(`[image] AI picked ${terms.length} search terms: ` +
        terms.slice(0, 8).map((t) => t.term).join(', ') + (terms.length > 8 ? '...' : ''));
      return terms;
    }
  } catch (e) {
    console.log(`[image] AI term extraction failed (${String(e?.message ?? e).split('\n')[0]}); using keyword fallback`);
  }
  return fallbackTerms(script);
}

// ----------------------------------------------------------------------------------
// Sources (each returns a list of direct image URLs)
// ----------------------------------------------------------------------------------
export async function searchWiki(term, want = 12) {
  const urls = [];
  try { // precise entity match: the article's own lead image
    const r = await get('https://en.wikipedia.org/api/rest_v1/page/summary/' + encodeURIComponent(term.replaceAll(' ', '_')));
    if (r.ok) {
      const img = (await r.json()).originalimage ?? {};
      if (okDims(img.width, img.height)) urls.push(img.source);
    }
  } catch { /* ignore */ }
  try {
    const r = await get('https://commons.wikimedia.org/w/api.php', {
      params: {
        action: 'query', format: 'json', generator: 'search',
        gsrsearch: term, gsrnamespace: 6, gsrlimit: Math.max(want, 16),
        prop: 'imageinfo', iiprop: 'url|size|mime', iiurlwidth: 1920,
      },
    });
    const pages = (await r.json())?.query?.pages ?? {};
    for (const p of Object.values(pages)) {
      for (const ii of p.imageinfo ?? []) {
        if (!['image/jpeg', 'image/png'].includes(ii.mime)) continue;
        if (!okDims(ii.width, ii.height)) continue;
        urls.push(ii.thumburl ?? ii.url); // 1920px thumb renders faster
      }
    }
  } catch { /* ignore */ }
  return urls.slice(0, want);
}

export async function searchOpenverse(term, want = 12) {
  try {
    const r = await get('https://api.openverse.org/v1/images/', {
      params: { q: term, page_size: Math.min(20, Math.max(want, 12)), mature: 'false' },
    });
    if (!r.ok) return [];
    return (await r.json()).results
      .filter((x) => x.url && okDims(x.width, x.height))
      .map((x) => x.url)
      .slice(0, want);
  } catch {
    return [];
  }
}

// Direct DuckDuckGo image search: VQD token + the i.js JSON endpoint (exactly what DDG's own
// image frontend calls). duck-duck-scrape wraps this same endpoint, but its internal chain is
// fragile (its VQD regex/format checks break on today's DDG responses -> "Cannot read
// properties of undefined (reading 'length')"), so we talk to the endpoint ourselves and keep
// the library only as a fallback.
async function ddgDirect(term, want) {
  const q = encodeURIComponent(term);
  const r1 = await get(`https://duckduckgo.com/?q=${q}&iax=images&ia=images`, { headers: _UA_BROWSER });
  if (!r1.ok) throw new Error(`HTTP ${r1.status} on the DDG search page`);
  const m = String(await r1.text()).match(/vqd=["'](\d+-\d+(?:-\d+)?)["']/);
  if (!m) throw new Error('no vqd token on the DDG search page');
  const r2 = await get(`https://duckduckgo.com/i.js?l=en-us&o=json&q=${q}&vqd=${m[1]}&p=-1&f=,,,&s=0`, {
    headers: { ..._UA_BROWSER, Referer: 'https://duckduckgo.com/', Accept: 'application/json' },
  });
  if (!r2.ok) throw new Error(`HTTP ${r2.status} on the DDG image endpoint`);
  const data = await r2.json();
  return (data?.results ?? []).slice(0, want);
}

export async function searchDdg(term, want = 12) {
  const urls = [];
  const collect = (results) => {
    for (const r of results ?? []) {
      if (!r || typeof r !== 'object') continue;
      const u = String(r.image ?? r.imageUrl ?? '');
      if (!/^https?:\/\//i.test(u)) continue;
      const w = Number(r.width ?? 0), h = Number(r.height ?? 0);
      if (w && h && !okDims(w, h)) continue; // skip tiny thumbnails / extreme banners
      urls.push(u);
      if (urls.length >= want) break;
    }
  };
  // 1) duck-duck-scrape. MUST be searchImages() (search() is web-text search and returns page
  //    results without image URLs) and safeSearch MUST be the SafeSearchType enum value
  //    ({ STRICT: 0, MODERATE: -1, OFF: -2 }) - a raw `safeSearch: 1` throws
  //    "1 is an invalid safe search type!".
  try {
    const { searchImages, SafeSearchType } = await import('duck-duck-scrape');
    collect((await searchImages(term, { safeSearch: SafeSearchType.MODERATE })).results);
  } catch (e) {
    console.log(`[image] ddg library failed (${String(e?.message ?? e).split('\n')[0]}); trying the direct DDG endpoint`);
  }
  // 2) direct endpoint fallback
  if (urls.length < want) {
    try {
      collect(await ddgDirect(term, Math.max(want * 2, 20)));
    } catch (e) {
      console.log(`[image] ddg source failed (${String(e?.message ?? e).split('\n')[0]}); using other sources`);
    }
  }
  return [...new Set(urls)].slice(0, want);
}

function stockProviders() {
  const keys = [];
  if (process.env.PEXELS_API_KEY) keys.push('pexels');
  if (process.env.UNSPLASH_ACCESS_KEY) keys.push('unsplash');
  if (process.env.PIXABAY_API_KEY) keys.push('pixabay');
  return keys;
}

export async function searchStock(term, want = 12) {
  const urls = [];
  try {
    if (process.env.PEXELS_API_KEY) {
      const r = await get('https://api.pexels.com/v1/search', {
        headers: { Authorization: process.env.PEXELS_API_KEY },
        params: { query: term, per_page: Math.max(want, 10), orientation: 'landscape' },
      });
      if (r.ok) urls.push(...((await r.json()).photos ?? []).filter((p) => p.src).map((p) => p.src.large2x));
    }
    if (process.env.UNSPLASH_ACCESS_KEY) {
      const r = await get('https://api.unsplash.com/search/photos', {
        headers: { Authorization: 'Client-ID ' + process.env.UNSPLASH_ACCESS_KEY },
        params: { query: term, per_page: Math.max(want, 10) },
      });
      if (r.ok) urls.push(...((await r.json()).results ?? []).filter((p) => p.urls).map((p) => p.urls.regular + '&fm=jpg&w=1920'));
    }
    if (process.env.PIXABAY_API_KEY) {
      const r = await get('https://pixabay.com/api/', {
        params: { key: process.env.PIXABAY_API_KEY, q: term, image_type: 'photo', per_page: Math.max(want, 10) },
      });
      if (r.ok) urls.push(...((await r.json()).hits ?? []).map((p) => p.largeImageURL).filter(Boolean));
    }
  } catch { /* ignore */ }
  return urls.slice(0, want);
}

// --- IMDb via the Obscura stealth browser (ported from images.py) -----------------
const IMDB_GALLERY_JS = `(limit) => {
  const IMG = /media-amazon\\.com|media-imdb\\.com/i;
  const ITEM = /\\/(?:mediaindex|mediaviewer)\\/rm\\d+/i;
  const JUNK = /1x1|transparent|pixel|sprite|blank|\\.svg/i;
  const best = (ss) => { if (!ss) return null; let b = null, w = -1;
    for (const part of ss.split(',')) { const seg = part.trim().split(/\\s+/); if (!seg[0]) continue;
      let x = 0; if (seg[1]) { if (seg[1].endsWith('w')) x = parseFloat(seg[1]) || 0;
        else if (seg[1].endsWith('x')) x = (parseFloat(seg[1]) || 1) * 1000; }
      if (x > w) { w = x; b = seg[0]; } } return b; };
  const out = [], seen = new Set();
  const push = (u) => { if (!u || !IMG.test(u) || JUNK.test(u) || seen.has(u)) return;
    u = u.replace(/\\._V1_[^/]*?(\\.(?:jpe?g|png|webp))(\\?.*)?$/i, '._V1_$1');  // full resolution
    seen.add(u); out.push(u); };
  for (const a of document.querySelectorAll('a[href]')) {
    if (!ITEM.test(a.getAttribute('href') || '')) continue;
    for (const img of a.querySelectorAll('img'))
      push(best(img.getAttribute('srcset')) || img.currentSrc || img.src || img.getAttribute('src') || '');
  }
  if (!out.length) { const scope = document.querySelector('#section-images, section[data-testid="media-index-page"]');
    if (scope) for (const img of scope.querySelectorAll('img'))
      push(best(img.getAttribute('srcset')) || img.currentSrc || img.src || '');
  }
  return out.slice(0, limit);
}`;

export async function searchImdb(term, want = 10, cfg = null) {
  // IMDb posters/gallery photos through the Obscura stealth browser. Needs obscura;
  // returns [] when it is missing or anything fails (the other sources cover the topic).
  const urls = [];
  try {
    const exe = findObscura(cfg?.obscura_path ?? process.env.OBSCURA_PATH);
    if (!exe) {
      console.log('[image] imdb source skipped (obscura binary not found next to main.js)');
      return [];
    }
    const { proc, port } = await startObscura({ private_net: false }, exe);
    let browser;
    try {
      browser = await chromium.connectOverCDP(`ws://127.0.0.1:${port}`);
      const page = await browser.newPage();
      // IMDb's SPA keeps soft-navigating while it hydrates, which destroys the JS context
      // mid-call -> "page.$: Execution context was destroyed, most likely because of a
      // navigation". So: (1) every read is ONE atomic page.evaluate (no element handles that
      // can die between calls), and (2) each read retries after the navigation settles.
      const evalRetry = async (fn, arg = undefined, tries = 3) => {
        for (let i = 1; ; i++) {
          try { return await page.evaluate(fn, arg); }
          catch (e) {
            if (i >= tries || !/Execution context|destroyed|navigation/i.test(String(e?.message ?? e))) throw e;
            await page.waitForTimeout(1200);
          }
        }
      };
      const q = encodeURIComponent(term);
      await page.goto(`https://www.imdb.com/find/?s=tt&q=${q}`, { waitUntil: 'domcontentloaded', timeout: 30000 });
      await page.waitForTimeout(1500);
      let href = null;
      for (let i = 0; i < 3 && !href; i++) {
        try {
          href = await evalRetry("() => document.querySelector('a[href*=\"/title/tt\"]')?.getAttribute('href') || null");
        } catch { /* page still navigating: wait and retry */ }
        if (!href) await page.waitForTimeout(1200);
      }
      const m = String(href || '').match(/\/title\/(tt\d+)/);
      if (m) {
        await page.goto(`https://www.imdb.com/title/${m[1]}/mediaindex/`, { waitUntil: 'domcontentloaded', timeout: 30000 });
        await page.waitForTimeout(2000);
        for (let i = 0; i < 4 && urls.length < want; i++) {
          urls.push(...await evalRetry(IMDB_GALLERY_JS, want));
          await evalRetry('() => window.scrollBy(0, Math.round(window.innerHeight * 0.9))');
          await page.waitForTimeout(700);
        }
        if (!urls.length) { // no gallery -> the main poster
          const u = await evalRetry(`() => {
            for (const sel of ['img[data-testid="hero-media__poster"]', '.ipc-poster img']) {
              const el = document.querySelector(sel);
              const s = el && (el.currentSrc || el.src || el.getAttribute('src'));
              if (s) return s;
            }
            return null;
          }`);
          if (u) urls.push(u);
        }
      }
    } finally {
      try { await browser?.close(); } catch { /* ignore */ }
      stopProcess(proc);
    }
  } catch (e) {
    console.log(`[image] imdb source failed (${String(e?.message ?? e).split('\n')[0]}); using other sources`);
  }
  return [...new Set(urls)].slice(0, want);
}

export const SEARCH = { wiki: searchWiki, openverse: searchOpenverse, ddg: searchDdg, stock: searchStock, imdb: searchImdb };
// which sources fit which AI term kind (first hit wins, then it rotates for variety)
const KIND_SOURCES = {
  entertainment: ['imdb', 'ddg', 'wiki', 'openverse'],
  person: ['wiki', 'ddg', 'stock', 'openverse'],
  place: ['wiki', 'ddg', 'stock', 'openverse'],
  technology: ['ddg', 'stock', 'openverse', 'wiki'],
  concept: ['ddg', 'openverse', 'wiki', 'stock'],
};

export async function availableSources(wanted = 'all', cfg = null) {
  // Sources that can actually run right now (keys present / packages installed / obscura installed).
  const have = ['wiki', 'openverse'];
  try { await import('duck-duck-scrape'); have.push('ddg'); } catch { /* not installed */ }
  have.push(...stockProviders());
  try { if (findObscura(cfg?.obscura_path ?? null)) have.push('imdb'); } catch { /* ignore */ }
  if (wanted && wanted.trim().toLowerCase() !== 'all') {
    const want = wanted.split(',').map((s) => s.trim().toLowerCase()).filter(Boolean);
    for (const s of want) {
      if (!SEARCH[s]) console.log(`[image] unknown source '${s}' (known: wiki, openverse, ddg, stock, imdb)`);
    }
    const missing = want.filter((s) => !have.includes(s));
    if (missing.length) console.log(`[image] note: no key/binary/package for ${missing.join(', ')}; the other sources still run`);
    return have.filter((s) => want.includes(s));
  }
  return have.length ? have : ['wiki', 'openverse'];
}

// ----------------------------------------------------------------------------------
// Download + validate
// ----------------------------------------------------------------------------------
async function downloadOne(url, filePath) {
  const headers = url.includes('media-amazon') || url.includes('media-imdb') ? IMDB_HEADERS : _UA_BROWSER;
  const res = await fetch(url, { headers, signal: AbortSignal.timeout(TIMEOUT) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = Buffer.from(await res.arrayBuffer());
  if (data.length > 25 * 1024 * 1024) throw new Error('image too large');
  if (data.length < 15 * 1024) throw new Error('file too small'); // placeholders / broken thumbs
  const meta = await sharp(data).metadata(); // throws when not a real decodable image
  if (!okDims(meta.width, meta.height)) throw new Error('bad dimensions');
  await fs.promises.writeFile(filePath, data);
  return { w: meta.width, h: meta.height, md5: crypto.createHash('md5').update(data).digest('hex') };
}

export async function downloadImages(urls, outDir, need, startIndex = 0) {
  // Download + validate + dedupe up to `need` images. Returns saved slide paths.
  await fs.promises.mkdir(outDir, { recursive: true });
  const saved = [];
  const hashes = new Set();
  let next = startIndex;
  const workers = new Array(8).fill(0).map(async () => {
    for (;;) {
      const i = next++;
      if (i >= urls.length || saved.length >= need) return;
      const url = urls[i];
      const filePath = path.join(outDir, `slide_${String(i).padStart(4, '0')}.jpg`);
      try {
        const { md5 } = await downloadOne(url, filePath);
        if (hashes.has(md5)) {
          await fs.promises.rm(filePath, { force: true });
          continue;
        }
        hashes.add(md5);
        saved.push(filePath);
        process.stdout.write(`\r[image] downloaded ${saved.length} usable image(s)`);
        if (saved.length >= need) return;
      } catch {
        await fs.promises.rm(filePath, { force: true });
      }
    }
  });
  await Promise.all(workers);
  console.log();
  return saved;
}

export async function localImages(imageDir, outDir, need) {
  // Use the user's own images from --image-dir (copied + validated, ready for the slideshow).
  if (!fs.existsSync(imageDir)) {
    console.log(`[image] --image-dir not found: ${imageDir}; falling back to AI web search`);
    return [];
  }
  const exts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp'];
  const files = fs.readdirSync(imageDir).filter((f) => exts.some((e) => f.toLowerCase().endsWith(e))).sort();
  if (!files.length) {
    console.log(`[image] no images inside ${imageDir}; falling back to AI web search`);
    return [];
  }
  await fs.promises.mkdir(outDir, { recursive: true });
  const out = [];
  for (const [i, f] of files.entries()) {
    if (out.length >= need) break;
    const dst = path.join(outDir, `slide_${String(i).padStart(4, '0')}.jpg`);
    try {
      const img = sharp(path.join(imageDir, f)).rotate(); // applies EXIF orientation
      const meta = await img.metadata();
      if (!okDims(meta.width, meta.height)) {
        console.log(`[image] skipping ${f} (${meta.width}x${meta.height} is below the quality floor)`);
        continue;
      }
      await img.jpeg({ quality: 92 }).toFile(dst);
      out.push(dst);
    } catch (e) {
      console.log(`[image] skipping ${f}: ${String(e?.message ?? e).split('\n')[0]}`);
    }
  }
  console.log(`[image] using ${out.length} local image(s) from ${imageDir}`);
  return out;
}

// ----------------------------------------------------------------------------------
// Orchestration
// ----------------------------------------------------------------------------------
export async function collect(cfg, script, outDir, need, sources = 'all') {
  // AI terms -> multi-source search -> validated downloads. Returns ordered slide paths.
  const existing = fs.existsSync(outDir)
    ? fs.readdirSync(outDir).filter((f) => /^slide_\d+\.jpg$/.test(f)).sort()
    : [];
  if (existing.length >= need) {
    console.log(`[image] reusing ${existing.length} previously downloaded images`);
    return existing.map((f) => path.join(outDir, f));
  }

  const terms = (await aiTerms(cfg, script)) || fallbackTerms(script);
  const srcs = await availableSources(sources, cfg);
  console.log(`[image] sources: ${srcs.join(', ')}`);

  const queue = []; // (term, source) pairs, rotated per term so every topic hits every fitting source
  for (const t of terms) {
    const pref = KIND_SOURCES[t.kind] ?? KIND_SOURCES.concept;
    const ordered = [...pref.filter((s) => srcs.includes(s)), ...srcs.filter((s) => !pref.includes(s))];
    for (const s of ordered) queue.push([t.term, s]);
  }

  const urls = [];
  const seen = new Set();
  const target = Math.floor(need * OVERFETCH) + 4;
  for (const [term, src] of queue) {
    if (urls.length >= target) break;
    try {
      const found = await SEARCH[src](term, 8, cfg);
      const fresh = found.filter((u) => !seen.has(u));
      fresh.forEach((u) => seen.add(u));
      urls.push(...fresh);
      if (fresh.length) console.log(`[image] ${src}: +${fresh.length} for "${term}" (total ${urls.length})`);
    } catch (e) {
      console.log(`[image] ${src} failed for "${term}": ${String(e?.message ?? e).split('\n')[0]}`);
    }
  }
  if (!urls.length) {
    console.log('[image] no image URLs found anywhere; the slideshow needs at least 2 images');
    return [];
  }
  return downloadImages(urls, outDir, need);
}
