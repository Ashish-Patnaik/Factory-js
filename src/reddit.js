// reddit.js - the --reddit pipeline (the b.mjs integration).
//
//   node main.js --reddit --publish instagram
//   node main.js --reddit --full --publish instagram      (--long/--short work as aliases)
//   node main.js --reddit --video --publish all
//   node main.js --reddit --video --video-style card --publish all
//
// Flow: one random subreddit from sources_reddit.md (or --subreddit <name>, or --post-url) ->
// the hot section -> the AI picks the post it "likes" (persona taste, fresh = not in
// output/history.json) -> a clean screenshot captured with the exact b.mjs technique
// (Obscura/Chromium over CDP, same selectors, same cleaning CSS, same snippet/full logic) ->
// an engaging / opinion-oriented persona caption -> an Instagram FEED image post (not a reel)
// via uploadInstagramImage() (Graph API photo posts need a public image_url, so the PNG is
// auto-hosted on catbox.moe with a tmpfiles.org fallback).
//
// With --video the same post also becomes a 9:16 TTS voice-over video for YouTube Shorts +
// Instagram Reels, in two visual styles:
//   --video-style scroll (default) - the live post page is restyled into the clean b.mjs card
//                                    and the factory's own scroll/encoder pipeline films it;
//   --video-style card             - the screenshot is composed onto a blurred background and
//                                    rendered as a static card through the slideshow encoder.
//
// Dependencies: exactly the b.mjs set (playwright over CDP + Obscura) plus sharp (already a
// factory dependency) and the factory's own Gemini/TTS/ASR/publish modules. Nothing new.
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';
import sharp from 'sharp';
import { findObscura, startObscura, stopProcess } from './scrape.js';
import { checkLlm, llm } from './gemini.js';
import { loadPersona, cleanScript, WPS } from './speech.js';
import { buildCaptionData } from './captions.js';
import { makeConfig } from './config.js';
import { loadHistory, saveHistory } from './agent.js';
import { ROOT, die, validateUrl, probe, parseRedditCookies } from './utils.js';

const REDDIT_OUT_DIR = path.join('output', 'reddit');
const POST_SELECTOR = 'shreddit-post, [data-testid="post-container"], .thing.link';
const CAPTION_MAX = 2200; // Instagram caption hard limit

// ---------------------------------------------------------------------------
// Page-side functions (serialized into the browser by Playwright - keep them self-contained)
// ---------------------------------------------------------------------------

// The b.mjs "click any read-more / show-more toggle" walker (shadow-DOM inclusive).
function clickReadMoreInPost() {
  let clicked = false;
  function clickInTree(root) {
    if (!root) return;
    root.querySelectorAll('button, a').forEach((el) => {
      const txt = (el.textContent || '').trim().toLowerCase();
      if (txt === 'read more' || txt === 'show more' || txt === 'see more') {
        el.click();
        clicked = true;
      }
    });
    root.querySelectorAll('*').forEach((el) => {
      if (el.shadowRoot) clickInTree(el.shadowRoot);
    });
  }
  const post = document.querySelector('shreddit-post, [data-testid="post-container"]');
  if (post) clickInTree(post);
  return clicked;
}

// The b.mjs box probe (with the shadow-root fallback).
function postBox(selector) {
  const el = document.querySelector(selector);
  if (!el) return null;
  let r = el.getBoundingClientRect();
  if (r.width === 0 || r.height === 0) {
    const inner = el.shadowRoot?.querySelector('article, div') || el.firstElementChild;
    if (inner) r = inner.getBoundingClientRect();
  }
  const pad = 16;
  return {
    x: Math.max(0, Math.floor(r.left - pad)),
    y: Math.max(0, Math.floor(r.top + window.scrollY - pad)),
    width: Math.ceil(r.width + pad * 2),
    height: Math.ceil(r.height + pad * 2),
  };
}

// The b.mjs cleaning + styling pass. mode: 'snippet' (truncated text + fade) | 'full'
// (whole post). mobile: adjusts the card to a phone viewport with bigger, readable type
// for the scroll video (the screenshot capture uses desktop sizing exactly like b.mjs).
function cleanAndStylePage(opts) {
  const mode = opts.mode;
  const mobile = Boolean(opts.mobile);

  const removeSelectors = [
    'reddit-header-large',
    'reddit-header-small',
    'shreddit-cookie-banner',
    '#right-sidebar-container',
    '#left-sidebar-container',
    'shreddit-async-loader',
    'pdp-back-button',
    'shreddit-post-overflow-menu',
    'faceplate-dropdown-menu',
    'rpl-dropdown',
    'rpl-tooltip',
    'shreddit-post-action-row',
    '[data-testid="action-row"]',
    '[data-testid="post-overflow-menu"]',
    '[slot="post-insights-panel"]',
    '.read-more-overflow-cover',
    '[class*="read-more-overflow"]',
    '.sr-only',
    '[hidden]',
  ];

  function cleanTree(root) {
    if (!root) return;

    for (const sel of removeSelectors) {
      root.querySelectorAll(sel).forEach((el) => el.remove());
    }

    root.querySelectorAll('button, a, [role="menuitem"]').forEach((el) => {
      const txt = (el.textContent || '').trim();
      const aria = (el.getAttribute('aria-label') || '').trim();
      if (
        ['Back', 'Options', 'Report', 'Close', '×', 'Read more', 'Read less'].includes(txt) ||
        ['Back', 'More options', 'Close', 'Read more'].includes(aria)
      ) {
        if (!el.href || !el.href.includes('/r/')) {
          el.remove();
        }
      }
    });

    // Unclamp inner read-more wrappers so [slot="text-body"] controls snippet vs full cleanly
    root
      .querySelectorAll(
        'read-more-max-height, read-more-truncator, read-more-provider, shreddit-post-text-body, faceplate-expandable-section-helper, [id$="-post-rtjson-content"], .md'
      )
      .forEach((el) => {
        el.style.setProperty('max-height', 'none', 'important');
        el.style.setProperty('height', 'auto', 'important');
        el.style.setProperty('overflow', 'visible', 'important');
        el.style.setProperty('-webkit-line-clamp', 'unset', 'important');
        el.style.setProperty('mask-image', 'none', 'important');
        el.style.setProperty('-webkit-mask-image', 'none', 'important');
      });

    root.querySelectorAll('*').forEach((el) => {
      if (el.shadowRoot) {
        const shadowStyle = document.createElement('style');
        shadowStyle.textContent = `
          :host, div, section, article, slot, read-more-max-height, read-more-truncator {
            max-height: none !important;
            overflow: visible !important;
            -webkit-line-clamp: unset !important;
            mask-image: none !important;
            -webkit-mask-image: none !important;
          }
        `;
        el.shadowRoot.appendChild(shadowStyle);
        cleanTree(el.shadowRoot);
      }
    });
  }

  cleanTree(document);

  document
    .querySelectorAll('shreddit-comment-tree, comment-body-header, #comment-tree')
    .forEach((el) => el.remove());

  const cardCss = mobile
    ? `
        width: calc(100vw - 12px) !important;
        max-width: none !important;
        margin: 10px 6px !important;
        padding: 20px 18px !important;
        border-radius: 16px !important;
      `
    : `
        max-width: 700px !important;
        width: 700px !important;
        margin: 24px auto !important;
        padding: 24px 28px !important;
        border-radius: 18px !important;
      `;
  const titleSize = mobile ? '26px' : '22px';
  const bodySize = mobile ? '17px' : '15px';
  const textBodyCss = mode === 'snippet'
    ? `
        max-height: ${mobile ? 420 : 360}px !important;
        overflow: hidden !important;
        position: relative !important;
      `
    : `
        max-height: none !important;
        height: auto !important;
        overflow: visible !important;
      `;

  const style = document.createElement('style');
  style.textContent = `
    html, body {
      background: #000000 !important;
      margin: 0 !important;
      padding: 0 !important;
      height: auto !important;
      overflow: visible !important;
    }
    shreddit-post, [data-testid="post-container"], .thing.link {
      display: block !important;
      ${cardCss}
      box-sizing: border-box !important;
      background: #ffffff !important;
      color: #1a1a1b !important;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    shreddit-post h1, [slot="title"] {
      font-size: ${titleSize} !important;
      line-height: 1.35 !important;
      margin: 14px 0 10px 0 !important;
      font-weight: 700 !important;
    }
    shreddit-post-flair, [slot="post-flair"] {
      display: inline-block !important;
      margin: 0 0 14px 0 !important;
    }
    shreddit-post-flair a, shreddit-post-flair span,
    [slot="post-flair"] a, [slot="post-flair"] span {
      padding: 4px 12px !important;
      border-radius: 9999px !important;
      display: inline-block !important;
      line-height: 1.4 !important;
    }
    [slot="text-body"] {
      font-size: ${bodySize} !important;
      line-height: 1.55 !important;
      color: #222222 !important;
      ${textBodyCss}
    }
    [slot="text-body"] p {
      margin: 0 0 12px 0 !important;
    }
    [slot="text-body"] p:last-child {
      margin-bottom: 0 !important;
    }
    [slot="credit-bar"] {
      display: flex !important;
      align-items: center !important;
      gap: 10px !important;
      margin-bottom: 4px !important;
    }
  `;
  document.head.appendChild(style);

  // In snippet mode, add a clean fade-out only if the text actually overflows
  if (mode === 'snippet') {
    const textBody = document.querySelector('[slot="text-body"]');
    if (textBody && textBody.scrollHeight > textBody.clientHeight + 4) {
      const fade = document.createElement('div');
      fade.style.cssText =
        'position:absolute;left:0;right:0;bottom:0;height:64px;' +
        'background:linear-gradient(to bottom, rgba(255,255,255,0), #ffffff);' +
        'pointer-events:none;';
      textBody.appendChild(fade);
    }
  }

  window.scrollTo(0, 0);
}

// ---------------------------------------------------------------------------
// Browser session (Obscura over CDP exactly like b.mjs; Chromium fallback)
// ---------------------------------------------------------------------------
async function withSession(cfg, fn) {
  const cookieRaw = cfg.reddit_cookie ?? process.env.REDDIT_SESSION_COOKIE ?? null;
  const exe = findObscura(cfg.obscura_path);
  if (exe) {
    console.log('[reddit] Obscura session (b.mjs technique: Chromium over CDP)');
    const { proc, port } = await startObscura(cfg, exe);
    let browser = null;
    try {
      browser = await chromium.connectOverCDP(`ws://127.0.0.1:${port}`);
      const context = browser.contexts()[0] || (await browser.newContext());
      const cookies = parseRedditCookies(cookieRaw);
      if (cookies.length) {
        try {
          await context.addCookies(cookies);
          console.log('[reddit] session cookie(s) applied - Reddit should treat this session as logged in');
        } catch {
          console.log('[reddit] warning: could not apply the session cookies (continuing without them)');
        }
      }
      const page = await context.newPage();
      page.setDefaultTimeout(Math.max(cfg.timeout, 30) * 1000);
      return await fn(page);
    } finally {
      if (browser) { try { await browser.close(); } catch { /* already gone */ } }
      stopProcess(proc);
    }
  }
  console.log('[reddit] Obscura not found; using bundled Chromium ' +
    '(drop obscura.exe into obscura/ for pages that block headless Chrome)');
  const browser = await chromium.launch({ headless: true });
  try {
    const context = await browser.newContext({ viewport: { width: 1280, height: 1000 } });
    const cookies = parseRedditCookies(cookieRaw);
    if (cookies.length) {
      try { await context.addCookies(cookies); } catch { /* best effort */ }
    }
    const page = await context.newPage();
    page.setDefaultTimeout(Math.max(cfg.timeout, 30) * 1000);
    return await fn(page);
  } finally {
    try { await browser.close(); } catch { /* ignore */ }
  }
}

// ---------------------------------------------------------------------------
// Reddit data layer (2026 edition). Reddit's wall, in order of severity:
//   a. .json URLs NAVIGATED to serve the SPA shell ("Skip to main content...").
//   b. Non-browser JSON requests get HTTP 403.
//   c. Suspicious sessions get a self-running JS challenge: the page redirects to
//      ...?solution=...&js_challenge=1&jsc_token=... A REAL browser (Obscura)
//      passes it automatically - it runs the JS, sets a cookie and lands on the
//      content. The ONLY wrong move is interrupting it with another goto, which
//      restarts the whole flow (that was the "interrupted by another navigation"
//      + 45s timeout loop).
//   d. Hard edge blocks: "You've been blocked by network security" (HTTP 403).
//   e. old.reddit is LOGIN-ONLY since mid-2026 (listing + .json both demand login)
//      - it is not usable logged-out, so there is no old.reddit fallback anywhere.
// All JSON is therefore fetched IN-PAGE (fetch() inside the reddit tab: session
// cookies, real browser TLS, same origin), and a REDDIT_SESSION_COOKIE /
// --reddit-cookie session (logged in) skips the entire wall - see
// parseRedditCookies() in utils.js.
// ---------------------------------------------------------------------------
const NAV_TIMEOUT = 45000;
const CHALLENGE_URL_RE = /[?&]js_challenge=1|[?&]jsc_token=/;
const CHALLENGE_WAIT_MS = () => Number(process.env.REDDIT_CHALLENGE_WAIT_MS ?? 25000);

async function gotoReddit(page, url, { attempts = 3 } = {}) {
  let lastErr = null;
  for (let i = 0; i < attempts; i++) {
    // The last try only waits for 'commit' - a hung font/ad can no longer kill the run.
    const waitUntil = i === attempts - 1 ? 'commit' : 'domcontentloaded';
    try {
      return await page.goto(url, { waitUntil, timeout: NAV_TIMEOUT });
    } catch (e) {
      const msg = String(e?.message ?? e);
      // "Navigation to X is interrupted by another navigation to Y": Reddit's JS
      // challenge redirect took over. That is the wall WORKING - let it land
      // instead of restarting it with a fresh goto.
      const interrupted = /interrupted by another navigation|ERR_ABORTED/i.test(msg);
      const target = (msg.match(/another navigation to\s+"?([^"\s,]+)"?/i) ?? [])[1] ?? '';
      if (interrupted && /reddit\.com/i.test(target)) {
        console.log('[reddit] the js_challenge redirect took over the navigation; letting it land');
        await page.waitForLoadState('domcontentloaded', { timeout: 20000 }).catch(() => {});
        await page.waitForTimeout(1500);
        if (/reddit\.com/i.test(page.url())) return null; // callers treat null as "unknown status"
      }
      lastErr = e;
      if (i < attempts - 1) {
        console.log(`[reddit] page load failed (${msg.split('\n')[0]}), retrying (attempt ${i + 2} of ${attempts})...`);
        await page.waitForTimeout(1500 + i * 1000).catch(() => {});
      }
    }
  }
  throw new Error(`could not open ${url} after ${attempts} attempt(s): ${String(lastErr?.message ?? lastErr).split('\n')[0]}`);
}

// Page-state probe for the challenge classifier.
async function pageState(page) {
  return page.evaluate(() => ({
    url: location.href,
    text: String(document.body ? document.body.innerText : '').trim(),
    scripts: document.scripts.length,
  })).catch(() => null);
}

function classifyPageState(state) {
  if (!state) return 'gone'; // page closed/crashed
  const head = state.text.slice(0, 600);
  if (/blocked by network security|you'?ve been blocked|log in to use old reddit/i.test(head)) return 'blocked';
  if (CHALLENGE_URL_RE.test(state.url)) return 'challenge';
  if (!state.text && state.scripts > 0) return 'challenge'; // script-only stub body
  return 'loaded';
}

// Waits out the js_challenge WITHOUT navigating (navigating restarts it). A real
// browser resolves it in a few seconds, sets the cookie and lands on the content.
async function resolveRedditChallenge(page, timeoutMs = null) {
  const deadline = Date.now() + (timeoutMs ?? CHALLENGE_WAIT_MS());
  let logged = false;
  while (true) {
    const kind = classifyPageState(await pageState(page));
    if (kind === 'loaded') return true;
    if (kind !== 'challenge') return false; // gone | blocked | login wall
    if (!logged) {
      console.log('[reddit] Reddit js_challenge in progress; waiting for it to resolve (no navigation - navigating would restart it)...');
      logged = true;
    }
    if (Date.now() > deadline) return false;
    await page.waitForTimeout(1000);
  }
}

function parseRedditJsonText(text, url) {
  const body = String(text ?? '').trim();
  if (!body || !/^[[{]/.test(body)) {
    throw new Error(`non-JSON response (${body.slice(0, 60) || 'empty'}) from ${url}`);
  }
  return JSON.parse(body);
}

// fetch() INSIDE the reddit tab. credentials: 'same-origin' sends the session
// cookies on same-origin calls and stays CORS-clean (reddit .json sends ACAO: *)
// when the tab happens to be on the other host or on about:blank.
async function fetchRedditJsonInPage(page, url) {
  const res = await page.evaluate(async (u) => {
    try {
      const r = await fetch(u, { headers: { Accept: 'application/json' }, credentials: 'same-origin' });
      return { status: r.status, text: await r.text() };
    } catch (e) {
      return { status: 0, text: '', error: String((e && e.message) || e) };
    }
  }, url);
  if (res.error) throw new Error(`in-page fetch failed: ${res.error}`);
  if (res.status >= 400) throw new Error(`HTTP ${res.status} from ${url}`);
  return parseRedditJsonText(res.text, url);
}

// Fallback: the browser context's request client (shares the session cookies).
async function fetchRedditJsonViaRequest(page, url) {
  const resp = await page.context().request.get(url, { headers: { Accept: 'application/json' } });
  if (resp.status() >= 400) throw new Error(`HTTP ${resp.status()} from ${url}`);
  return parseRedditJsonText(await resp.text(), url);
}

// Tries every URL with two strategies: in-page fetch (same origin) -> context
// request (shares the session cookies). Returns the first parsed JSON.
async function fetchRedditJson(page, urls) {
  const list = Array.isArray(urls) ? urls : [urls];
  let lastErr = null;
  for (const url of list) {
    for (const how of ['in-page', 'context']) {
      try {
        return how === 'in-page'
          ? await fetchRedditJsonInPage(page, url)
          : await fetchRedditJsonViaRequest(page, url);
      } catch (e) {
        lastErr = e;
      }
    }
  }
  throw lastErr ?? new Error('no JSON URL to try');
}

function normalizePost(d) {
  return {
    title: String(d.title ?? '').trim(),
    permalink: d.permalink ? `https://www.reddit.com${d.permalink}` : '',
    selftext: String(d.selftext ?? '').trim(),
    score: Number(d.score ?? 0),
    numComments: Number(d.num_comments ?? 0),
    author: String(d.author ?? 'unknown'),
    flair: String(d.link_flair_text ?? ''),
    isText: Boolean(d.is_self),
    subreddit: String(d.subreddit_name_prefixed ?? d.subreddit ?? ''),
  };
}

// Page-side collectors (serialized into the browser - keep them self-contained).
function collectNewReddit() {
  const out = [];
  for (const el of document.querySelectorAll('shreddit-post')) {
    const t = (el.querySelector('[slot="title"], h1')?.textContent || '').replace(/\s+/g, ' ').trim();
    const permalink = el.getAttribute('permalink');
    if (t && permalink) {
      out.push({
        title: t,
        permalink: 'https://www.reddit.com' + permalink,
        selftext: '',
        score: Number(el.getAttribute('score') ?? 0),
        numComments: Number(el.getAttribute('comment-count') ?? 0),
        author: el.getAttribute('author') ?? 'unknown',
        flair: (el.getAttribute('post-flair-text') ?? '').trim(),
        isText: false,
        subreddit: el.getAttribute('subreddit-prefixed-name') ?? '',
      });
    }
  }
  return out;
}

// The new-UI hot feed is virtualized: only ~3 posts exist in the DOM until you
// scroll. A few rounds of scroll + collect pull in 15-30 posts.
async function collectHotFromPage(page, collector) {
  let items = await page.evaluate(collector).catch(() => []);
  for (let i = 0; items.length < 15 && i < 6; i++) {
    await page.evaluate(() => window.scrollBy(0, Math.round(window.innerHeight * 1.8))).catch(() => {});
    await page.waitForTimeout(700);
    items = await page.evaluate(collector).catch(() => []);
  }
  await page.evaluate(() => window.scrollTo(0, 0)).catch(() => {});
  await page.waitForTimeout(300);
  return items;
}

async function fetchHotPosts(page, subredditUrl) {
  const base = subredditUrl.replace(/\/+$/, '');

  const resp = await gotoReddit(page, `${base}/hot/`);
  const settled = await resolveRedditChallenge(page);
  const navStatus = resp ? resp.status() : 0;

  // First choice: the .json listing fetched in-page (navigating to it directly
  // is exactly what produced the "Skip to main content" / 403 failures).
  let lastErr = null;
  if (settled && navStatus < 400) {
    try {
      const j = await fetchRedditJson(page, [`${base}/hot.json?limit=30&raw_json=1`]);
      const posts = (j?.data?.children ?? [])
        .map((c) => c?.data ?? {})
        .filter((d) => d.title && !d.stickied && !d.over_18)
        .map(normalizePost)
        .filter((p) => p.permalink);
      if (posts.length) {
        console.log(`[reddit] hot section: ${posts.length} post(s) via JSON`);
        return posts;
      }
      lastErr = new Error('hot.json returned no posts');
    } catch (e) {
      lastErr = e;
    }
  } else {
    lastErr = new Error(settled ? `HTTP ${navStatus} on the hot page` : 'the js_challenge never resolved');
  }
  console.log(`[reddit] hot.json failed (${String(lastErr?.message ?? lastErr).split('\n')[0]}); reading the hot page HTML`);

  if (settled) {
    // The hot page is already open - just scroll its virtualized feed.
    const items = await collectHotFromPage(page, collectNewReddit);
    if (items.length) {
      console.log(`[reddit] hot section: ${items.length} post(s) via HTML`);
      return items;
    }
  }

  throw new Error(
    `could not read the hot section of ${base} - Reddit's anti-bot wall stopped this session. ` +
    'What works: (1) simply re-run in a few minutes (the wall is throttled, not permanent), ' +
    '(2) run with a logged-in session: --reddit-cookie "<token2 cookie value from your browser>" ' +
    'or the REDDIT_SESSION_COOKIE env var (logged-in sessions skip the wall entirely), ' +
    '(3) try a different network if this IP is heavily flagged');
}

async function fetchPostDetail(page, permalink) {
  // Full selftext + the top comments: they feed the caption and the video script.
  const out = { title: '', selftext: '', comments: [] };
  let jsonErr = null;
  try {
    if (!/reddit\.com/i.test(page.url())) {
      // Not on a reddit origin (e.g. --post-url from a blank tab): open the post
      // first so the .json fetch below is same-origin with session cookies.
      await gotoReddit(page, permalink);
      await resolveRedditChallenge(page);
    }
    const clean = permalink.replace(/\/+$/, '');
    const j = await fetchRedditJson(page, [`${clean}.json?limit=12&raw_json=1&sort=top`]);
    const postData = j?.[0]?.data?.children?.[0]?.data ?? {};
    out.title = String(postData.title ?? '').trim();
    if (postData.selftext) out.selftext = String(postData.selftext).trim();
    for (const c of j?.[1]?.data?.children ?? []) {
      if (c?.kind !== 't1') continue;
      const body = String(c.data?.body ?? '').trim();
      if (!body || body === '[deleted]' || body === '[removed]' || body.length < 25) continue;
      out.comments.push({ body, score: Number(c.data?.score ?? 0) });
      if (out.comments.length >= 3) break;
    }
    if (out.comments.length) console.log(`[reddit] fetched post details + ${out.comments.length} top comment(s)`);
  } catch (e) {
    jsonErr = e;
    console.log(`[reddit] json details failed (${String(e?.message ?? e).split('\n')[0]}); reading the post page HTML`);
    try {
      const scraped = await scrapePostDetailFromPage(page, permalink);
      if (scraped.title || scraped.selftext || scraped.comments.length) {
        out.title = out.title || scraped.title;
        out.selftext = out.selftext || scraped.selftext;
        if (!out.comments.length) out.comments = scraped.comments;
        console.log(`[reddit] post details scraped from the post page HTML` +
          (scraped.comments.length ? ` (+${scraped.comments.length} comment(s))` : ''));
      }
    } catch {
      console.log(`[reddit] could not fetch post details (${String(jsonErr?.message ?? jsonErr).split('\n')[0]}); continuing with the listing data`);
    }
  }
  return out;
}

// Page-side post-detail scraper for the shreddit UI (serialized - self-contained).
// Used when the .json detail endpoint is walled; the post PAGE itself usually renders.
function scrapeDetailPage() {
  const out = { title: '', selftext: '', comments: [] };
  const post = document.querySelector('shreddit-post');
  if (post) {
    const t = post.querySelector('[slot="title"]')?.textContent ?? post.getAttribute('title') ?? '';
    out.title = String(t).replace(/\s+/g, ' ').trim();
    out.selftext = String(post.querySelector('[slot="text-body"]')?.innerText ?? '').trim();
  }
  if (!out.title) out.title = String(document.querySelector('h1')?.textContent ?? '').trim();
  for (const c of document.querySelectorAll('shreddit-comment')) {
    const body = String(c.querySelector('[slot="comment"]')?.innerText ?? '').trim();
    if (body && body.length >= 25 && body !== '[deleted]' && body !== '[removed]') {
      out.comments.push({ body, score: Number(c.getAttribute('score') ?? 0) });
      if (out.comments.length >= 3) break;
    }
  }
  return out;
}

async function scrapePostDetailFromPage(page, permalink) {
  if (!/\/comments\//.test(page.url())) {
    await gotoReddit(page, permalink);
    await resolveRedditChallenge(page);
    await page.waitForTimeout(1500);
  }
  return page.evaluate(scrapeDetailPage);
}

// Polls for the shreddit post element while any js_challenge resolves in place.
async function waitForPostElement(page, timeoutMs = 40000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await page.evaluate((sel) => !!document.querySelector(sel), POST_SELECTOR).catch(() => false)) return true;
    const kind = classifyPageState(await pageState(page));
    if (kind === 'blocked' || kind === 'gone') return false;
    await page.waitForTimeout(kind === 'challenge' ? 1200 : 800);
  }
  return false;
}

function subredditFromPermalink(permalink) {
  const m = String(permalink).match(/reddit\.com\/r\/([^/]+)\//i);
  return m ? `r/${m[1]}` : 'reddit';
}

// ---------------------------------------------------------------------------
// Sources file + subreddit helpers
// ---------------------------------------------------------------------------
export function parseSubredditFile(file = 'sources_reddit.md') {
  if (!fs.existsSync(file)) {
    die(`Subreddit sources file not found: ${file}. Create it with one subreddit URL per line ` +
      '(see sources_reddit.md), or pass --subreddit <name> / --post-url <url>.');
  }
  const urls = [];
  for (const line of fs.readFileSync(file, 'utf8').split(/\r?\n/)) {
    const l = line.trim();
    if (!l || l.startsWith('#')) continue;
    for (const m of l.matchAll(/https?:\/\/[^\s)>\]]+/g)) {
      if (!/reddit\.com\/r\//.test(m[0])) {
        console.log(`[reddit] skipping non-subreddit URL: ${m[0]}`);
        continue;
      }
      urls.push(m[0]);
    }
  }
  if (!urls.length) die(`No subreddit URLs (https://www.reddit.com/r/<name>/) found inside ${file}`);
  return urls;
}

export function subredditUrl(input) {
  // 'developersIndia' | 'r/developersIndia' | 'https://www.reddit.com/r/developersIndia/'
  const s = String(input ?? '').trim().replace(/^\/+/, '');
  if (/^https?:\/\//i.test(s)) {
    const u = validateUrl(s);
    if (!/reddit\.com\/r\//.test(u)) die(`--subreddit must be a subreddit URL like https://www.reddit.com/r/<name>/ (got: ${s})`);
    return u;
  }
  const name = s.replace(/^r\//i, '').replace(/\/+$/, '');
  if (!name || /[/?#]/.test(name)) die(`invalid subreddit: ${input} (use a name like developersIndia or r/developersIndia)`);
  return `https://www.reddit.com/r/${name}/`;
}

function postSlug(permalink, fallbackTitle) {
  // /r/x/comments/<id>/<slug>/ -> "<slug>-<id>" (short, filesystem-safe)
  try {
    const m = new URL(permalink).pathname.match(/\/comments\/([^/]+)(?:\/([^/]+))?/);
    if (m) {
      const slug = (m[2] || m[1]).replace(/[^\w-]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 60);
      return `${slug || 'post'}-${m[1]}`.slice(0, 80);
    }
  } catch { /* fall through */ }
  return (String(fallbackTitle ?? 'post').replace(/[^\w-]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 60)) ||
    `post-${Date.now()}`;
}

// ---------------------------------------------------------------------------
// AI steps (Gemini via src/gemini.js, persona.md voice)
// ---------------------------------------------------------------------------
function extractJson(raw) {
  let s = String(raw ?? '').trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
  const m = s.match(/\{[\s\S]*\}/);
  return JSON.parse(m ? m[0] : s);
}

async function aiPickPost(cfg, persona, posts, subredditUrl_) {
  const listing = posts.slice(0, 30).map((p, i) => {
    const body = p.selftext ? p.selftext.slice(0, 500).replace(/\s+/g, ' ') : '(no text body - image/link post)';
    return `${i}. [${p.title}](${p.permalink}) score:${p.score} comments:${p.numComments} flair:"${p.flair}"\n   ${body}`;
  }).join('\n');
  const prompt = `You are the autonomous content director for an Instagram account that posts screenshots of Reddit posts.

Creator Persona (follow its taste):
${persona}

Subreddit: ${subredditUrl_} (hot section right now)

Candidate posts:
${listing}

Pick the ONE post that will make the best Instagram image post today:
1. Self-contained text whose screenshot ALONE tells a story, hot take, confession, life lesson or wild fact.
2. Strong hook in the title; the body keeps delivering (real stories, opinions, career/dev insight for a tech audience).
3. Avoid: pure link/image/video posts with no text, polls, mod announcements, inside jokes that need thread context, anything NSFW.
Return ONLY valid JSON: {"pick": <index>, "reason": "one sentence"}`;
  const data = extractJson(await llm(cfg, prompt));
  let idx = Number(data.pick);
  if (!Number.isInteger(idx) || idx < 0 || idx >= posts.length) idx = 0;
  console.log(`[reddit] AI selection reason: ${data.reason ?? 'best hook on the list'}`);
  return posts[idx];
}

async function pickWithFallback(cfg, persona, posts, subredditUrl_) {
  try {
    return await aiPickPost(cfg, persona, posts, subredditUrl_);
  } catch (e) {
    console.log(`[reddit] AI pick failed (${String(e?.message ?? e).split('\n')[0]}); falling back to the top-scored text post`);
    const sorted = [...posts].sort((a, b) => (Number(b.isText) - Number(a.isText)) || (b.score - a.score));
    return sorted[0] ?? posts[0];
  }
}

async function aiCaption(cfg, persona, post, comments) {
  const topComments = comments.slice(0, 3)
    .map((c) => `- (${c.score} points) ${c.body.slice(0, 300).replace(/\s+/g, ' ')}`)
    .join('\n') || '(none)';
  const body = post.selftext.slice(0, 2500) || '(the screenshot shows only the title)';
  const prompt = `You run an Instagram account that posts screenshots of Reddit posts. Write the caption for today's post.

=== CREATOR PERSONA & VOICE (FOLLOW STRICTLY) ===
${persona}

=== THE REDDIT POST ===
Subreddit: ${post.subreddit}
Title: ${post.title}
Score: ${post.score} upvotes, ${post.numComments} comments
Body:
${body}

Top comments:
${topComments}

Caption rules:
1. First line: a scroll-stopping hook (bold statement, hot take or sharp question) in the persona's voice.
2. Then 2-4 short lines: your engaging, opinion-oriented reaction to what the screenshot shows - not a dry summary.
3. Then one question that drives comments.
4. No emojis, no markdown, no "source:" lines - the screenshot itself is the content. The caption body must stay under 1600 characters.
5. Return the hashtags ONLY in the hashtags array (8-12 targeted niche hashtags, lowercase, each starting with #) - never inside the caption.

Return ONLY valid JSON: {"caption": "the caption text with \\n line breaks", "hashtags": ["#tag1", "#tag2"]}`;
  const data = extractJson(await llm(cfg, prompt));
  const tags = Array.isArray(data.hashtags)
    ? data.hashtags.map((h) => (String(h).startsWith('#') ? String(h) : `#${h}`))
    : [];
  let caption = String(data.caption ?? '').trim();
  if (tags.length) caption = `${caption}\n\n${tags.join(' ')}`.trim();
  return caption.slice(0, CAPTION_MAX);
}

async function aiVideoScript(cfg, persona, post, comments) {
  // SUMMARY + HOT TAKE: compress the post, then give the persona's own sharp take.
  const maxWords = Math.floor(cfg.script_seconds * WPS);
  const body = post.selftext ? post.selftext.slice(0, 3500) : `(title only) ${post.title}`;
  const top = comments[0]?.body?.slice(0, 400) ?? '';
  const prompt = `You write the spoken voice-over for a vertical Reddit-story video (YouTube Shorts / Instagram Reels). The post's screenshot stays on screen the whole time.

=== CREATOR PERSONA & VOICE (FOLLOW STRICTLY) ===
${persona}

=== THE REDDIT POST ===
${post.title} (${post.subreddit}, ${post.score} upvotes)
${body}
${top ? `\nTop comment: ${top}` : ''}

Style: SUMMARY + HOT TAKE. In 2-3 sentences summarize what the post says, then give YOUR sharp, contrarian take on it in the persona's voice. End with one punchy question or takeaway line that sparks comments.

Strict rules:
- Spoken English only, max ${maxWords} words (under ${cfg.script_seconds} seconds). No markdown, no emojis, no hashtags, never read URLs aloud.
- Output ONLY the spoken script text.`;
  return cleanScript(await llm(cfg, prompt), maxWords);
}

async function aiVideoPackage(cfg, persona, post, script) {
  const prompt = `You are a short-form video creator and social media strategist packaging a finished vertical video for YouTube Shorts + Instagram Reels.

=== CREATOR PERSONA (FOLLOW STRICTLY) ===
${persona}

=== VIDEO SUMMARY (spoken voice-over) ===
${script}

=== SOURCE ===
${post.permalink} (${post.subreddit})

Return ONLY valid JSON:
{
  "slug": "short_kebab_case_filename_slug",
  "youtube": {
    "title": "punchy curiosity-gap title under 70 chars",
    "description": "2-3 sentence high-retention description matching the persona. Include source: ${post.permalink}",
    "tags": ["5-8 lowercase search keyword phrases viewers would type"],
    "hashtags": ["#Shorts", "#ai", "#programming", "#tech"],
    "category": "tech"
  },
  "instagram": {
    "title": "scroll-stopping first line hook for the Reel caption",
    "description": "engaging multi-line caption body in the persona's voice + a question at the end to drive comments",
    "hashtags": ["#reels", "#coding", "#ai", "#tech", "#softwareengineer", "#programming"]
  }
}`;
  const pkg = extractJson(await llm(cfg, prompt));
  const yt = pkg.youtube ?? {};
  if (!Array.isArray(yt.tags) || !yt.tags.length) {
    yt.tags = (yt.hashtags ?? []).map((h) => String(h).replace(/^#/, ''));
  }
  pkg.youtube = yt;
  pkg.instagram = pkg.instagram ?? null;
  pkg.url = post.permalink;
  pkg.reddit_post = post.subreddit;
  pkg.created_at = new Date().toISOString().replace('T', ' ').slice(0, 19);
  pkg.long_form = false;
  return pkg;
}

// ---------------------------------------------------------------------------
// Screenshot capture - the 1:1 b.mjs port
// ---------------------------------------------------------------------------
async function capturePost(page, postUrl, style, outPath) {
  await page.setViewportSize({ width: 1280, height: 1000 });
  await page.emulateMedia({ colorScheme: 'light' }).catch(() => { /* older CDP targets */ });

  // Retrying navigation, then a challenge-aware wait: the cookie from the listing
  // step usually makes this instant; if Reddit serves a js_challenge here too, we
  // wait it out before declaring the post missing (never navigate mid-challenge).
  await gotoReddit(page, postUrl);

  if (!(await waitForPostElement(page, 40000))) {
    const title = await page.title().catch(() => '(unknown)');
    const debugPath = path.join(REDDIT_OUT_DIR, 'debug.png');
    await page.screenshot({ path: debugPath }).catch(() => { /* best effort */ });
    throw new Error(
      `Post element not found.\n` +
      `  URL: ${postUrl}\n` +
      `  Title: ${title}\n` +
      `  (Reddit is walling this session - see the hot-section error fixes: re-run later or use --reddit-cookie)\n` +
      `  Saved debug screenshot to: ${debugPath}`);
  }

  await page.waitForTimeout(1500);

  // In full mode, click any "Read more" / "Show more" toggle first
  if (style === 'full') {
    const clickedReadMore = await page.evaluate(clickReadMoreInPost).catch(() => false);
    if (clickedReadMore) await page.waitForTimeout(500);
  }

  // Clean Reddit UI clutter & apply the snippet/full styling (b.mjs, desktop sizing)
  await page.evaluate(cleanAndStylePage, { mode: style, mobile: false });

  const getBox = () => page.evaluate(postBox, POST_SELECTOR);
  let box = await getBox();
  if (!box || box.width === 0 || box.height === 0) {
    throw new Error('Could not determine post dimensions.');
  }

  const targetHeight = Math.max(1000, box.y + box.height + 60);
  if (targetHeight > 1000) {
    await page.setViewportSize({ width: 1280, height: targetHeight });
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(200);
    box = await getBox();
  }

  await page.screenshot({ path: outPath, clip: box });
  // The clip is the card box + a uniform 16px black pad, so the card sits at (16,16).
  const card = { x: 16, y: 16, width: Math.max(1, box.width - 32), height: Math.max(1, box.height - 32) };
  console.log(`[reddit] saved [${style}] screenshot (${box.width}x${box.height}) to: ${outPath}`);
  return { path: outPath, width: box.width, height: box.height, card };
}

// Restyles the LIVE post page for the scroll video - called from main.js run() via the
// cfg.reddit_style hook, right before the scroll/recording pass.
export async function styleRedditPage(page, cfg) {
  const style = cfg.reddit_full ? 'full' : 'snippet';
  const mobile = Boolean(cfg.mobile);
  console.log(`[reddit] styling the live post page for a ${mobile ? 'mobile' : 'desktop'} scroll video (style: ${style})`);
  try {
    await page.waitForSelector(POST_SELECTOR, { state: 'attached', timeout: 15000 });
  } catch {
    throw new Error('reddit post element not found on the page - the video would film a broken page. ' +
      'If Reddit keeps blocking the browser, put obscura.exe into obscura/ (it is picked automatically).');
  }
  await page.waitForTimeout(1500);
  if (style === 'full') {
    const clickedReadMore = await page.evaluate(clickReadMoreInPost).catch(() => false);
    if (clickedReadMore) await page.waitForTimeout(500);
  }
  await page.evaluate(cleanAndStylePage, { mode: style, mobile });
}

// ---------------------------------------------------------------------------
// --video-style card: compose the 9:16 frame (blurred cover bg + rounded card) with sharp
// ---------------------------------------------------------------------------
async function composeCardFrame(screenshotPath, card, outPath, width, height) {
  // width x height: 1080x1920 for the video, 1080x1350 (4:5, the IG feed portrait limit) for images.
  const cardBuf = await sharp(screenshotPath)
    .extract({ left: card.x, top: card.y, width: card.width, height: card.height })
    .resize({ width: Math.round(width * 0.85), height: Math.round(height * 0.9), fit: 'inside', withoutEnlargement: false })
    .png().toBuffer();
  const meta = await sharp(cardBuf).metadata();
  const cw = meta.width, ch = meta.height;
  const radius = Math.max(16, Math.round(cw * 0.03));
  const mask = Buffer.from(
    `<svg width="${cw}" height="${ch}"><rect x="0" y="0" width="${cw}" height="${ch}" rx="${radius}" ry="${radius}"/></svg>`);
  const rounded = await sharp(cardBuf).composite([{ input: mask, blend: 'dest-in' }]).png().toBuffer();
  const bg = await sharp(screenshotPath)
    .resize(width, height, { fit: 'cover' })
    .blur(35)
    .modulate({ brightness: 0.45 })
    .png().toBuffer();
  await sharp(bg)
    .composite([{ input: rounded, left: Math.round((width - cw) / 2), top: Math.round((height - ch) / 2) }])
    .png().toFile(outPath);
}

async function renderCardVideo(cfg, opts) {
  // Static-card render through the factory's slideshow encoder (mediabunny + WebCodecs).
  const wav = path.join(opts.work, 'speech.wav');
  const vcfg = makeConfig({
    url: '', width: 1080, height: 1920, fps: opts.fps, quality: opts.quality,
    bitrate: opts.bitrate ?? null, output: opts.output, start_delay: 0, pause: 0,
    script_seconds: cfg.script_seconds, narrate: false, text: opts.script,
    captions: opts.captions, cap_style: opts.cap_style, cap_color: opts.cap_color,
    cap_font: opts.cap_font, cap_size: opts.cap_size, cap_position: opts.cap_position,
    cap_words: opts.cap_words, cap_chars: opts.cap_chars, cap_caps: opts.cap_caps,
    kenburns: opts.kenburns, slide_seconds: 6, slide_fit: 'exact',
    avatar_file: opts.avatar_file, avatar_corner: opts.avatar_corner, avatar_size: opts.avatar_size,
    voice: opts.voice, tts_model: opts.tts_model, tts_quant: opts.tts_quant, tts_lang: opts.tts_lang,
    persona_file: cfg.persona_file, keep_files: opts.keep_files,
    encoder_channel: opts.encoder_channel ?? null,
    reddit_cookie: cfg.reddit_cookie ?? null,
  });

  console.log('[reddit] synthesizing the voice-over (summary + hot take)...');
  const { narration } = await import('./speech.js');
  const res = await narration(vcfg, wav);
  const secs = res.secs;
  const reelMax = Number(process.env.REEL_MAX_SECONDS ?? '89');
  let totalS = secs;
  if (totalS > reelMax) {
    console.log(`note: voice-over is ${secs.toFixed(1)}s; cutting video+audio at the ${reelMax}s reel cap`);
    totalS = reelMax;
  }
  if (cfg.keep_files) {
    fs.copyFileSync(wav, opts.output.replace(/\.mp4$/i, '') + '.wav');
    fs.writeFileSync(opts.output.replace(/\.mp4$/i, '') + '.txt', res.script + '\n');
    console.log(`[out] script + voice-over saved next to ${opts.output}`);
  }

  let assData = null;
  if (vcfg.captions && res.words?.length) {
    assData = buildCaptionData(res.words, vcfg, 1080, 1920, 0);
  }
  const { EncoderHost } = await import('./encoder/host.js');
  const slides = await import('./slides.js');
  const host = await new EncoderHost().launch({ channel: vcfg.encoder_channel ?? null });
  try {
    await host.initEncoder({ width: 1080, height: 1920, fps: vcfg.fps, quality: vcfg.quality, bitrate: vcfg.bitrate ?? null });
    const fontPath = path.join(ROOT, 'fonts', 'Anton-Regular.ttf');
    if (fs.existsSync(fontPath)) await host.setFont(fs.readFileSync(fontPath));
    if (assData) await host.setCaptions(assData);
    if (vcfg.avatar_file) {
      console.log(`[avatar] looping ${path.basename(vcfg.avatar_file)} in the ${vcfg.avatar_corner} corner over the card`);
      await host.setAvatar(fs.readFileSync(vcfg.avatar_file), vcfg.avatar_corner, vcfg.avatar_size);
    }
    const { readWav } = await import('./tts.js');
    const { pcm, sampleRate } = readWav(wav);
    await host.deliverAudio(pcm, { sampleRate, audioDelay: 0, totalDuration: totalS });
    console.log('[reddit] rendering the card video (static card + captions)...');
    const info = await slides.renderSlideshow(host, vcfg, [opts.framePath], { assData, totalS, work: opts.work });
    await host.rpc('finish'); // finalize output + attach audio
    await host.pullToFile(opts.output);
    console.log(`done: ${info.frames} frames in ${(info.ms / 1000).toFixed(0)}s`);
  } finally {
    await host.close();
  }
  await probe(opts.output);
}

// ---------------------------------------------------------------------------
// Orchestrator - called from main.js when --reddit is passed
// ---------------------------------------------------------------------------
export async function runRedditFlow(cfg, g, work, hooks = {}) {
  const style = g.reddit_full ? 'full' : 'snippet';
  fs.mkdirSync(REDDIT_OUT_DIR, { recursive: true });
  console.log(`[reddit] mode: ${g.reddit_video ? 'video (TTS voice-over)' : 'screenshot post'} | ` +
    `screenshot style: ${style} | publish: ${g.publish}`);

  if (g.post_url) {
    if (!/reddit\.com\/.+\/comments\//.test(g.post_url)) {
      die('--post-url must be a reddit post URL like https://www.reddit.com/r/<sub>/comments/<id>/<slug>/');
    }
  }
  checkLlm();

  // 1) Resolve the post: --post-url, or random subreddit -> hot section -> AI pick.
  const history = loadHistory();
  let subreddit = null;
  let post = null;
  if (g.post_url) {
    post = {
      permalink: validateUrl(g.post_url), title: '', selftext: '', score: 0, numComments: 0,
      author: 'unknown', flair: '', isText: true, subreddit: '',
    };
    subreddit = 'direct post URL';
    if (history.has(post.permalink)) console.log('[reddit] note: this post is already in output/history.json (posting it again anyway - --post-url is explicit)');
  } else {
    const file = g.subreddits || 'sources_reddit.md';
    const subs = parseSubredditFile(file);
    if (g.subreddit) {
      subreddit = subredditUrl(g.subreddit);
      console.log(`[reddit] pinned subreddit: ${subreddit}`);
    } else {
      subreddit = subs[Math.floor(Math.random() * subs.length)];
      console.log(`[reddit] random subreddit from ${file}: ${subreddit}`);
    }
  }

  // 2) One browser session does everything: hot listing, post details, the screenshot.
  const persona = await loadPersona(cfg.persona_file);
  if (!persona) console.log('[reddit] note: persona.md missing/unreadable - the AI will use a neutral voice');
  let captured = null;
  await withSession(cfg, async (page) => {
    if (!g.post_url) {
      const hot = await fetchHotPosts(page, subreddit);
      const fresh = hot.filter((p) => !history.has(p.permalink));
      if (!fresh.length) {
        die('every hot post on this subreddit is already in output/history.json - ' +
          'add more subreddits to sources_reddit.md, use --subreddit <name>, or clear the history file');
      }
      console.log(`[reddit] ${fresh.length} fresh hot post(s) (${hot.length - fresh.length} already posted)`);
      post = await pickWithFallback(cfg, persona, fresh, subreddit);
    }
    const detail = await fetchPostDetail(page, post.permalink);
    post = {
      ...post,
      title: detail.title || post.title,
      selftext: detail.selftext || post.selftext,
      comments: detail.comments,
      subreddit: post.subreddit || subredditFromPermalink(post.permalink),
    };
    console.log(`[reddit] post: "${(post.title || post.permalink).slice(0, 90)}" (${post.subreddit}, ${post.score} upvotes, ${post.numComments} comments)`);
    const slug = postSlug(post.permalink, post.title);
    captured = await capturePost(page, post.permalink, style, path.join(REDDIT_OUT_DIR, `${slug}-${style}.png`));
  });

  const slug = postSlug(post.permalink, post.title);

  // 3) Screenshot-post mode: persona caption -> Instagram FEED image post.
  if (!g.reddit_video) {
    console.log('[reddit] writing the persona caption (engaging / opinion-oriented)...');
    const caption = await aiCaption(cfg, persona, post, post.comments ?? []);
    const base = path.join(REDDIT_OUT_DIR, `${slug}-${style}`);
    fs.writeFileSync(base + '.caption.txt', caption + '\n');
    const meta = {
      slug, style, url: post.permalink, subreddit: post.subreddit, title: post.title,
      score: post.score, comments: post.numComments, screenshot: captured.path,
      caption, created_at: new Date().toISOString().replace('T', ' ').slice(0, 19),
    };
    fs.writeFileSync(base + '.meta.json', JSON.stringify(meta, null, 2));
    console.log(`[reddit] caption (${caption.length} chars) saved to ${base}.caption.txt`);
    console.log('--- Instagram caption ---');
    console.log(caption);
    console.log('-------------------------');

    let publishedAny = false;
    const errors = [];
    if (g.publish === 'youtube') {
      console.log('note: YouTube has no image-post API here - a screenshot post goes to Instagram only. ' +
        'Use --publish instagram, or --reddit --video --publish all for YouTube Shorts.');
    }
    if (g.publish === 'instagram' || g.publish === 'all') {
      if (g.publish === 'all') {
        console.log('note: --publish all on a screenshot post means Instagram (the image). ' +
          'Run --reddit --video --publish all for YouTube + Instagram video.');
      }
      try {
        const publish = await import('./publish.js');
        await publish.uploadInstagramImage(captured.path, caption, g.image_url ?? process.env.IG_IMAGE_URL ?? null);
        publishedAny = true;
      } catch (e) {
        const msg = String(e?.message ?? e).split('\n')[0];
        console.log(`[instagram] ERROR: ${msg}`);
        errors.push(`Instagram: ${msg}`);
      }
    }
    if (publishedAny || g.publish === 'none') saveHistory(post.permalink);
    if (errors.length) die(errors.join(' | '));
    return { post, screenshot: captured.path, caption };
  }

  // 4) Video mode: summary + hot-take script -> 9:16 render -> metadata -> publish.
  console.log('[reddit] writing the voice-over script (summary + hot take)...');
  const script = await aiVideoScript(cfg, persona, post, post.comments ?? []);
  const vBase = g.output ? String(g.output).replace(/\.mp4$/i, '') : path.join(REDDIT_OUT_DIR, slug);
  if (cfg.script_only) {
    fs.writeFileSync(vBase + '.txt', script + '\n');
    console.log(`[out] script saved to ${vBase}.txt`);
    console.log(`[script] ${script}`);
    return { post, script };
  }
  const vOut = vBase + '.mp4';

  if ((g.video_style ?? 'scroll') === 'card') {
    const framePath = path.join(REDDIT_OUT_DIR, `${slug}-${style}-card.png`);
    console.log('[reddit] composing the 9:16 card frame (blurred background + rounded screenshot)...');
    await composeCardFrame(captured.path, captured.card, framePath, 1080, 1920);
    console.log(`[reddit] card frame ready: ${framePath}`);
    await renderCardVideo(cfg, {
      script, framePath, output: vOut, work,
      fps: cfg.fps, quality: cfg.quality, bitrate: cfg.bitrate,
      captions: cfg.captions, cap_style: cfg.cap_style, cap_color: cfg.cap_color,
      cap_font: cfg.cap_font, cap_size: cfg.cap_size, cap_position: cfg.cap_position,
      cap_words: cfg.cap_words, cap_chars: cfg.cap_chars, cap_caps: cfg.cap_caps,
      avatar_file: cfg.avatar_file, avatar_corner: cfg.avatar_corner, avatar_size: cfg.avatar_size,
      voice: cfg.voice, tts_model: cfg.tts_model, tts_quant: cfg.tts_quant, tts_lang: cfg.tts_lang,
      kenburns: g.kenburns_given ? g.kenburns : false, // static by default: text must stay readable
      encoder_channel: cfg.encoder_channel, keep_files: cfg.keep_files,
    });
  } else {
    console.log('[reddit] building the scroll video from the styled post page...');
    const exe = findObscura(cfg.obscura_path);
    const browser = exe ? 'obscura' : 'chromium';
    if (!exe) console.log('[reddit] Obscura not found; the render will use bundled Chromium');
    const vcfg = makeConfig({
      url: post.permalink, width: 1080, height: 1920, viewport: null, mobile: true, long: false,
      duration: 20, fps: cfg.fps, speed: null, quality: cfg.quality, bitrate: cfg.bitrate ?? null,
      output: vOut, wait: 1, start_delay: 0, pause: 0, scroll_mode: 'smooth',
      pause_every: 0, pause_for: 1, browser, mode: browser === 'obscura' ? 'stepped' : 'virtual',
      capture: 'auto', warmup: true, headed: false, channel: cfg.channel ?? null, timeout: cfg.timeout,
      narrate: false, text: script, enhance: false, prompt: cfg.prompt ?? '',
      persona_file: cfg.persona_file, entertainment: false, script_seconds: cfg.script_seconds,
      script_only: false, scraper: cfg.scraper, obscura_path: cfg.obscura_path,
      private_net: cfg.private_net, llm_model: cfg.llm_model, voice: cfg.voice,
      tts_model: cfg.tts_model, tts_quant: cfg.tts_quant, tts_lang: cfg.tts_lang,
      whisper_model: 'tiny', captions: cfg.captions, cap_style: cfg.cap_style,
      cap_color: cfg.cap_color, cap_font: cfg.cap_font, cap_size: cfg.cap_size,
      cap_position: cfg.cap_position, cap_words: cfg.cap_words, cap_chars: cfg.cap_chars,
      cap_caps: cfg.cap_caps, separate: cfg.separate, keep_files: cfg.keep_files,
      avatar_file: cfg.avatar_file, avatar_corner: cfg.avatar_corner, avatar_size: cfg.avatar_size,
      image: false, image_dir: null, image_limit: 0, slide_seconds: 6, image_sources: 'all',
      kenburns: true, encoder_channel: cfg.encoder_channel ?? null,
      reddit_cookie: cfg.reddit_cookie ?? null,
      reddit_style: true, reddit_full: g.reddit_full,
    });
    await hooks.renderVideo?.(vcfg);
  }

  console.log('[reddit] generating the video metadata package (YouTube + Instagram)...');
  const pkg = await aiVideoPackage(cfg, persona, post, script);
  const metaPath = vOut.replace(/\.mp4$/i, '') + '.meta.json';
  fs.writeFileSync(metaPath, JSON.stringify(pkg, null, 2));
  console.log(`[reddit] metadata saved to ${metaPath}`);
  console.log(`  YT Title : ${pkg.youtube.title}`);
  if (pkg.instagram) console.log(`  IG Hook  : ${pkg.instagram.title}`);

  const publish = await import('./publish.js');
  const errors = [];
  let publishedAny = false;
  if (g.publish === 'youtube' || g.publish === 'all') {
    try {
      const yt = pkg.youtube;
      await publish.uploadYoutubeShort(vOut, yt.title, yt.description, yt.hashtags ?? [], g.yt_privacy,
        yt.tags ?? null, yt.category ?? null);
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
      await publish.uploadInstagramReel(vOut, ig.title, ig.description, ig.hashtags, true, null);
      publishedAny = true;
    } catch (e) {
      const msg = String(e?.message ?? e).split('\n')[0];
      console.log(`[instagram] ERROR: ${msg}`);
      errors.push(`Instagram: ${msg}`);
    }
  }

  if (publishedAny || g.publish === 'none') saveHistory(post.permalink);

  if (g.publish !== 'none' && publishedAny && !cfg.keep_files) {
    for (const fpath of [vOut, metaPath]) {
      if (fs.existsSync(fpath)) {
        fs.rmSync(fpath);
        console.log(`[cleanup] removed ${fpath}`);
      }
    }
  }
  if (errors.length) die(errors.join(' | '));
  return { post, script, pkg, video: vOut };
}

// Exposed for the network regression test (scripts/test_reddit_net.mjs).
export { gotoReddit, fetchHotPosts, fetchPostDetail, collectNewReddit, waitForPostElement, resolveRedditChallenge };