// publish.js - publish generated videos to YouTube Shorts and Instagram Reels.
//
// Instagram hardening (the famous recurring failure, kept from the Python factory and now
// also prevented at the encoder level):
//   * The Graph API hard-caps Reels at 90 s even though the in-app limit is 3 minutes; longer
//     files upload fine and then die in Meta's transcoder with ProcessingFailedError
//     (retriable: False) AFTER all bytes are uploaded. -> The factory caps short videos at
//     REEL_MAX_SECONDS (89 s) BEFORE rendering, and _igPrepare trims/heals as a last resort.
//   * Meta's transcoder needs H.264 yuv420p LIMITED range + an audio track at 44.1 kHz.
//     -> The mediabunny encoder emits exactly that (yuv420p tv, AAC 44.1 kHz), and healing
//       re-checks before upload.
//   * A failed transcode poisons that container forever: re-POSTing bytes into the same one
//     can never succeed. -> Every retry creates a FRESH container.
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { probeMedia } from './utils.js';

// --- YouTube config ---
const CLIENT_SECRET_FILE = process.env.YT_CLIENT_SECRET_FILE ?? 'client_secrets.json';
const TOKEN_FILE = process.env.YT_TOKEN_FILE ?? 'token.json';
const YT_SCOPES = ['https://www.googleapis.com/auth/youtube.upload'];
const YT_CATEGORY_ID = '28'; // Science & Tech (default; per-video category comes from the agent package)
const YT_CATEGORIES = {
  tech: '28', science: '28', education: '27', entertainment: '24', comedy: '23', news: '25',
  film: '1', animation: '1', 'film & animation': '1', games: '20', gaming: '20', music: '10',
};
const YT_TAGS_BUDGET = 480; // YouTube caps the tags field at 500 chars; stay under it

// --- Instagram config ---
const IG_ACCESS_TOKEN = (process.env.IG_ACCESS_TOKEN ?? 'YOUR_ACCESS_TOKEN_HERE').trim();
const IG_USER_ID = (process.env.IG_USER_ID ?? 'YOUR_IG_USER_ID_HERE').trim();
// v21.0 is nearing Meta's deprecation window; v23.0 is current and confirmed working with
// resumable reel uploads (override with IG_GRAPH_VERSION in .env if you must pin one).
const GRAPH_VERSION = process.env.IG_GRAPH_VERSION ?? 'v23.0';
const BASE_URL = `https://graph.facebook.com/${GRAPH_VERSION}`;
const IG_MAX_DURATION = Number(process.env.IG_MAX_DURATION ?? '90');
export { IG_MAX_DURATION };
// Resumable upload protocol (Meta docs): ONE request carries the whole file -
// offset: 0 + file_size: N headers + all N bytes as the body. The endpoint validates
// bytes_received == file_size - offset per request and answers PartialRequestError
// ("Partial request (did not match length of file)") otherwise - so fixed-size chunk
// bodies (the YouTube protocol, NOT this one) can never validate. IG_CHUNK_SIZE is
// legacy, kept only for the igChunkPlan unit tests.
const IG_CHUNK_SIZE = Number(process.env.IG_CHUNK_SIZE ?? 8 * 1024 * 1024);
const IG_UPLOAD_ATTEMPTS = Number(process.env.IG_UPLOAD_ATTEMPTS ?? '3');

// ---------------------------------------------------------------------------
// Shared HTTP helper
// ---------------------------------------------------------------------------
async function jsonFetch(url, opts = {}) {
  const res = await fetch(url, opts);
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON body */ }
  return { status: res.status, data, text: await res.text().catch(() => '') };
}

// ---------------------------------------------------------------------------
// YouTube OAuth (hand-rolled installed-app flow: no googleapis dependency, ~150 lines)
// ---------------------------------------------------------------------------
function loadClientSecret() {
  if (!fs.existsSync(CLIENT_SECRET_FILE)) {
    throw new Error(`Missing ${CLIENT_SECRET_FILE} for YouTube OAuth`);
  }
  const doc = JSON.parse(fs.readFileSync(CLIENT_SECRET_FILE, 'utf8'));
  const key = doc.installed ?? doc.web;
  if (!key) throw new Error(`${CLIENT_SECRET_FILE} has no installed/web client block`);
  return key;
}

async function refreshToken(client, refreshToken_) {
  const res = await jsonFetch('https://oauth2.googleapis.com/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      client_id: client.client_id,
      client_secret: client.client_secret,
      refresh_token: refreshToken_,
      grant_type: 'refresh_token',
    }),
  });
  if (res.status !== 200 || !res.data?.access_token) {
    throw new Error(`YouTube token refresh failed: ${res.text?.slice(0, 200)}`);
  }
  return res.data;
}

function saveToken(tok) {
  fs.writeFileSync(TOKEN_FILE, JSON.stringify(tok, null, 2));
}

function loadToken() {
  try { return JSON.parse(fs.readFileSync(TOKEN_FILE, 'utf8')); } catch { return null; }
}

async function getYtCredentials() {
  const client = loadClientSecret();
  let tok = loadToken();
  if (tok?.refresh_token) {
    try {
      const fresh = await refreshToken(client, tok.refresh_token);
      tok = { ...tok, ...fresh };
      saveToken(tok);
      return tok.access_token;
    } catch (e) {
      console.log(`[youtube] token refresh failed, re-authorizing: ${e.message}`);
      tok = null;
    }
  }
  if (!tok?.access_token) {
    // On CI there is no browser to sign in with: the interactive flow below would burn
    // its 5-minute timeout and kill the upload. Fail fast with the actual fix instead.
    if (process.env.CI || process.env.GITHUB_ACTIONS) {
      throw new Error(
        'YouTube OAuth needs a one-time interactive browser sign-in, which cannot run on GitHub Actions. ' +
        'Re-authorize locally once (delete token.json, run any upload), then base64 the fresh token.json ' +
        'into the YT_TOKEN_JSON_B64 repo secret - its refresh_token then renews itself on every CI run.',
      );
    }
    // Installed-app flow: tiny localhost server catches the redirect, browser opens consent.
    const port = await new Promise((ok) => {
      const probe = http.createServer();
      probe.listen(0, '127.0.0.1', () => { const p = probe.address().port; probe.close(() => ok(p)); });
    });
    const redirectUri = `http://127.0.0.1:${port}`;
    const authUrl = new URL('https://accounts.google.com/o/oauth2/v2/auth');
    authUrl.searchParams.set('client_id', client.client_id);
    authUrl.searchParams.set('redirect_uri', redirectUri);
    authUrl.searchParams.set('response_type', 'code');
    authUrl.searchParams.set('scope', YT_SCOPES.join(' '));
    authUrl.searchParams.set('access_type', 'offline');
    authUrl.searchParams.set('prompt', 'consent');
    console.log('[youtube] opening the browser for Google sign-in ...');
    console.log(`[youtube] if nothing opens, visit:\n  ${authUrl.toString()}`);
    const { spawn } = await import('node:child_process');
    const open = process.platform === 'win32' ? spawn('cmd', ['/c', 'start', '', authUrl.toString()], { detached: true, stdio: 'ignore' })
      : process.platform === 'darwin' ? spawn('open', [authUrl.toString()], { detached: true, stdio: 'ignore' })
        : spawn('xdg-open', [authUrl.toString()], { detached: true, stdio: 'ignore' });
    open.unref();
    const code = await new Promise((resolve, reject) => {
      const srv = http.createServer((req, res) => {
        const u = new URL(req.url, redirectUri);
        if (u.searchParams.get('code')) {
          res.writeHead(200, { 'Content-Type': 'text/html' });
          res.end('<h1>YouTube authorized. You can close this tab.</h1>');
          resolve(u.searchParams.get('code'));
        } else {
          res.writeHead(400);
          res.end('missing code');
        }
      });
      srv.listen(port, '127.0.0.1');
      setTimeout(() => reject(new Error('YouTube OAuth timed out waiting for the browser redirect')), 5 * 60000);
    });
    const res = await jsonFetch('https://oauth2.googleapis.com/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        client_id: client.client_id,
        client_secret: client.client_secret,
        code,
        grant_type: 'authorization_code',
        redirect_uri: redirectUri,
      }),
    });
    if (res.status !== 200 || !res.data?.access_token) {
      throw new Error(`YouTube token exchange failed: ${res.text?.slice(0, 200)}`);
    }
    tok = res.data;
    saveToken(tok);
  }
  return tok.access_token;
}

// Turn whatever callers/LLMs provide into clean YouTube fields. Accepted shapes:
//   tags:    keyword array | comma-separated string | null (fall back to hashtags)
//            | legacy single-object call form { tags, category } (main.js used to pass this)
//   hashtags: array | comma/space-separated string like "#ai #tech"
// Returns the hashtag list for the description plus the 500-char-budgeted keyword array.
export function normalizeYtTags(tags, hashtags, isShort) {
  if (tags && typeof tags === 'object' && !Array.isArray(tags)) {
    tags = tags.tags ?? null; // legacy { tags, category } options object passed positionally
  }
  const hashList = (Array.isArray(hashtags) ? hashtags : String(hashtags ?? '').split(/[,\s]+/))
    .map((h) => String(h).trim()).filter(Boolean)
    .map((h) => (h.startsWith('#') ? h : `#${h}`));
  if (isShort && !hashList.some((h) => h.toLowerCase() === '#shorts')) hashList.push('#Shorts');
  const src = tags == null ? hashList
    : Array.isArray(tags) ? tags
      : String(tags).split(',');
  const keywords = [];
  let total = 0;
  for (const t of src.map((x) => String(x).replace(/^#/, '').trim()).filter(Boolean)) {
    if (total + t.length + 1 > YT_TAGS_BUDGET) break;
    keywords.push(t);
    total += t.length + 1;
  }
  return { hashList, keywords };
}

export async function uploadYoutubeShort(videoPath, title, description, hashtags, privacy = 'public', tags = null, category = null) {
  // Upload a video to YouTube: as a Short when it is <= 3 min, as a regular long-form video
  // otherwise (--long). SEO keyword `tags` drive search for long-form uploads.
  if (!fs.existsSync(videoPath)) throw new Error(`Video file not found: ${videoPath}`);

  const meta = await probeMedia(videoPath).catch(() => null);
  const dur = meta?.duration ?? 0;
  const isShort = dur > 0 ? dur <= 180 : true;

  const { hashList, keywords } = normalizeYtTags(tags, hashtags, isShort);
  const fullDescription = `${description.trim()}\n\n${hashList.join(' ')}`;

  const cat = YT_CATEGORIES[String(category ?? '').trim().toLowerCase()] ?? YT_CATEGORY_ID;
  const accessToken = await getYtCredentials();

  // Resumable upload: initiate, then PUT 8 MB chunks (same protocol/behavior as the Python client).
  const initRes = await fetch('https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${accessToken}`,
      'Content-Type': 'application/json; charset=UTF-8',
      'X-Upload-Content-Length': String(fs.statSync(videoPath).size),
      'X-Upload-Content-Type': 'video/mp4',
    },
    body: JSON.stringify({
      snippet: {
        title: title.slice(0, 100),
        description: fullDescription,
        tags: keywords,
        categoryId: cat,
        defaultLanguage: 'en',
      },
      status: { privacyStatus: privacy, selfDeclaredMadeForKids: false },
    }),
  });
  if (initRes.status !== 200) {
    throw new Error(`YouTube initiate failed (${initRes.status}): ${(await initRes.text()).slice(0, 200)}`);
  }
  const uploadUri = initRes.headers.get('location');
  if (!uploadUri) throw new Error('YouTube initiate returned no upload URI (missing Location header)');

  console.log(`[youtube] uploading ${videoPath} as '${title.slice(0, 60)}' ` +
    `(${isShort ? 'Short' : 'long-form'}, category ${cat}, ${keywords.length} tags)...`);
  const fileBuf = fs.readFileSync(videoPath);
  const CHUNK = 8 * 1024 * 1024;
  let offset = 0;
  let response = null;
  while (offset < fileBuf.length) {
    const end = Math.min(offset + CHUNK, fileBuf.length);
    const chunk = fileBuf.subarray(offset, end);
    const isLast = end >= fileBuf.length;
    const res = await fetch(uploadUri, {
      method: 'PUT',
      headers: {
        'Content-Length': String(chunk.length),
        'Content-Range': `bytes ${offset}-${end - 1}/${fileBuf.length}`,
      },
      body: chunk,
    });
    if (res.status === 308) {
      offset = Number(res.headers.get('range')?.split('-')[1] ?? end);
      console.log(`[youtube] uploaded ${Math.floor(offset / fileBuf.length * 100)}%`);
      continue;
    }
    response = { status: res.status, body: await res.json().catch(() => null) };
    if (res.status !== 200) throw new Error(`YouTube upload failed (${res.status}): ${JSON.stringify(response.body).slice(0, 300)}`);
    break;
  }
  if (!response) { // final empty PUT for exact multiples of the chunk size
    const res = await fetch(uploadUri, {
      method: 'PUT',
      headers: { 'Content-Range': `bytes */${fileBuf.length}`, 'Content-Length': '0' },
    });
    response = { status: res.status, body: await res.json().catch(() => null) };
  }
  const videoId = response.body?.id;
  if (!videoId) throw new Error(`YouTube upload returned no video id: ${JSON.stringify(response.body).slice(0, 200)}`);
  const url = isShort ? `https://www.youtube.com/shorts/${videoId}` : `https://youtu.be/${videoId}`;
  console.log(`[youtube] upload complete! URL: ${url}`);
  return url;
}

// ---------------------------------------------------------------------------
// Instagram
// ---------------------------------------------------------------------------
function igCheck(res) {
  if (res.status !== 200 || (res.data && res.data.error)) {
    throw new Error(`Instagram API error: ${JSON.stringify(res.data ?? res.text?.slice(0, 200))}`);
  }
  return res.data;
}

// --- chunk plan + rupload response parsing (exported for unit tests) ---
// Byte plan for the chunked resumable upload: [{ offset, size }, ...] covering exactly
// fileSize bytes.
export function igChunkPlan(fileSize, chunkSize = IG_CHUNK_SIZE) {
  const plan = [];
  for (let off = 0; off < fileSize; off += chunkSize) {
    plan.push({ offset: off, size: Math.min(chunkSize, fileSize - off) });
  }
  return plan;
}

// The rupload endpoint answers success as {"success":true,"message":"Upload successful."}
// and failures in TWO documented shapes: top-level debug_info
// ({"debug_info":{"retriable":false,"type":"ProcessingFailedError","message":"..."}})
// and Graph-style {"error":{...,"debug_info":{...}}}. This normalizes both so retry
// decisions are made on the REAL error type. (The old code only read the nested shape,
// got type='' and mis-classified every rupload ProcessingFailedError as a hard
// rejection - attempts 2..N never ran.)
export function parseRuploadResult(data2, httpStatus) {
  const dbg = data2?.debug_info ?? data2?.error?.debug_info ?? null;
  const type = String(dbg?.type ?? data2?.error?.type ?? '');
  const message = String(dbg?.message ?? data2?.error?.message ?? '');
  let nested = '';
  if (message.startsWith('{')) {
    try { nested = String(JSON.parse(message)?.error?.message ?? ''); } catch { /* not nested JSON */ }
  }
  const retriable = dbg?.retriable === true;
  const authFail = /unauthorized|invalid oauth|access.?token|invalid token|session has been/i.test(`${message} ${nested}`);
  const ok = httpStatus === 200 && !dbg && !(data2 && data2.error);
  return { ok, type, message, nested, retriable, authFail };
}

async function igProbe(videoPath) {
  const m = await probeMedia(videoPath).catch(() => null);
  if (!m || !m.video) return { path: videoPath, meta: m };
  const audioDesc = m.audio ? `${m.audio.codec}/${m.audio.sampleRate}Hz` : 'MISSING';
  console.log(`[instagram] probe: ${m.video.codec} ${m.video.width}x${m.video.height} @${m.video.fps.toFixed(2)} fps | ` +
    `audio ${audioDesc} | ${m.duration.toFixed(1)}s, ${(m.size / 1e6).toFixed(1)} MB`);
  return { path: videoPath, meta: m };
}

// Diagnose the reel-spec violations that make Meta's transcoder bail with
// ProcessingFailedError (retriable: False) AFTER the bytes are already uploaded:
//   - duration > 90 s (the Graph API's hard reel cap)
//   - missing audio track
// The Node factory's encoder produces compliant files by construction, so healing is the
// exception; the check itself always prints the exact stream properties (as in the Python era).
async function igPrepare(videoPath, encoderFactory = null) {
  const { path: p, meta } = await igProbe(videoPath);
  if (!meta?.video) return p;
  const trim = meta.duration > IG_MAX_DURATION;
  const silent = !meta.audio;
  if (!trim && !silent) return p;
  const fixes = [];
  if (trim) fixes.push(`cut ${meta.duration.toFixed(1)}s -> ${(IG_MAX_DURATION - 1.5).toFixed(1)}s (API reel cap is 90s)`);
  if (silent) fixes.push('add silent AAC track (reels need an audio stream)');
  if (!encoderFactory) {
    console.log(`[instagram] auto-heal needed (${fixes.join('; ')}) but no encoder available; ` +
      'the factory\'s own renders are already compliant - upload will use the file as-is');
    return p;
  }
  console.log(`[instagram] auto-healing: ${fixes.join('; ')}`);
  const fixed = p.replace(/\.mp4$/i, '') + '.ig.mp4';
  await encoderFactory({ input: p, output: fixed, maxDuration: IG_MAX_DURATION - 1.5 });
  return fixed;
}

export async function uploadInstagramReel(videoPath, title, description, hashtags, shareToFeed = true, encoderFactory = null) {
  // Upload a local video to Instagram Reels via Graph API resumable upload.
  if (!fs.existsSync(videoPath)) throw new Error(`Video file not found: ${videoPath}`);
  if (!IG_ACCESS_TOKEN || IG_ACCESS_TOKEN === 'YOUR_ACCESS_TOKEN_HERE') {
    throw new Error('Instagram credentials missing. Set IG_ACCESS_TOKEN and IG_USER_ID env vars.');
  }

  const hashtagStr = Array.isArray(hashtags)
    ? hashtags.map((h) => (h.startsWith('#') ? h : `#${h}`)).join(' ')
    : String(hashtags);
  const caption = `${title.trim()}\n\n${description.trim()}\n\n${hashtagStr}`.trim();

  // Diagnose + auto-heal BEFORE touching the API (90s cap, silent audio).
  videoPath = await igPrepare(videoPath, encoderFactory);

  let err = null;
  let containerId = null;
  // Meta's ingest layer intermittently fails transfers that are 100% spec-compliant
  // ("1363008 OIL Error[FILE_NOT_FOUND]" / ProcessingFailedError) - Meta's own forums
  // confirm it is server-side, and the same file often succeeds on a later attempt.
  // So: N attempts, escalating waits, a FRESH container every time, and byte-exact
  // chunked transfers.
  const waits = [20000, 60000];
  for (let attempt = 1; attempt <= IG_UPLOAD_ATTEMPTS; attempt++) {
    err = null;
    // Step 1: Create container -- freshly on EVERY attempt. If Meta's transcoder failed
    // the previous container, re-POSTing bytes into the same one can never succeed.
    const created = await jsonFetch(`${BASE_URL}/${IG_USER_ID}/media`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        media_type: 'REELS',
        upload_type: 'resumable',
        caption: caption.slice(0, 2200),
        share_to_feed: String(shareToFeed),
        access_token: IG_ACCESS_TOKEN,
      }),
    });
    const data = igCheck(created);
    containerId = data.id;
    const uploadUri = data.uri;
    console.log(`[instagram] container created (attempt ${attempt}/${IG_UPLOAD_ATTEMPTS}): ${containerId}`);

    // Step 2: Resumable upload per Meta's official protocol
    // (developers.facebook.com/documentation/instagram-platform/content-publishing/
    // resumable-uploads): ONE request carries the whole file - Authorization: OAuth,
    // offset: 0, file_size: N headers, body = all N bytes. The endpoint validates
    // bytes_received == file_size - offset PER REQUEST and answers PartialRequestError
    // "Partial request (did not match length of file)" otherwise - which is why fixed
    // 8 MB chunk bodies always stalled at byte 0: an 8 MB body against a 32 MB
    // file_size can never validate, and Meta's retriable:true flag is a trap (retrying
    // the same partial body can never succeed). The documented "resumable" part means
    // RESUME: if a transfer is interrupted, query bytes_transferred and re-send ONLY
    // the remaining bytes with that offset (still exactly file_size - offset). Each
    // body is a Buffer slice, so undici sends an exact Content-Length. Do NOT mix in
    // X-Entity-* headers here (separate protocol; mixing broke the Python-era attempts).
    const fileBuf = fs.readFileSync(videoPath);
    const fileSize = fileBuf.length;
    console.log(`[instagram] uploading ${videoPath} (${(fileSize / 1e6).toFixed(1)} MB) as one resumable request (auto-resume from bytes_transferred on interruption)`);
    let offset = 0;
    let resumes = 0;
    while (offset < fileSize) {
      const body = fileBuf.subarray(offset); // ALL remaining bytes: exactly file_size - offset
      let chunkOk = false;
      let lastChunkErr = null;
      let authErr = null;
      for (let chunkTry = 1; chunkTry <= 3 && !chunkOk; chunkTry++) {
        let parsed;
        try {
          const r = await fetch(uploadUri, {
            method: 'POST',
            headers: {
              Authorization: `OAuth ${IG_ACCESS_TOKEN}`,
              offset: String(offset),
              file_size: String(fileSize),
            },
            body,
            signal: AbortSignal.timeout(600000),
          });
          parsed = parseRuploadResult(await r.json().catch(() => null), r.status);
        } catch (e) {
          parsed = { ok: false, type: 'NetworkError', message: String(e?.message ?? e), nested: '', retriable: true, authFail: false };
        }
        if (parsed.ok) { chunkOk = true; break; }
        if (parsed.authFail) {
          authErr = new Error('Instagram auth rejected (check IG_ACCESS_TOKEN - expired, wrong token type, ' +
            `or missing instagram_content_publish): ${parsed.nested || parsed.message}`);
          break;
        }
        lastChunkErr = `${parsed.type || 'unknown'}: ${parsed.message.slice(0, 160)}`;
        console.log(`[instagram] rupload @${offset} try ${chunkTry}/3 failed (retriable=${parsed.retriable}): ${lastChunkErr}`);
        if (!parsed.retriable && parsed.type !== 'ProcessingFailedError') break; // hard rejection
        if (chunkTry < 3) await new Promise((r2) => setTimeout(r2, 3000 * chunkTry));
      }
      // A bad or expired token never fixes itself, so fail fast with a readable message
      // instead of burning the remaining attempts (the docs' own failure sample shows
      // ProcessingFailedError can WRAP "unauthorized user request").
      if (authErr) throw authErr;
      if (!chunkOk) {
        // Documented resume path (docs Step 5): ask Meta how many bytes actually landed
        // and re-send only the remainder. If Meta already holds the whole file, the error
        // response was a lie - treat the transfer as complete. If Meta holds LESS than
        // our offset, the container is dead -> the next attempt gets a FRESH container.
        let have = 0;
        try {
          const st = await jsonFetch(`${BASE_URL}/${containerId}?fields=video_status&access_token=${encodeURIComponent(IG_ACCESS_TOKEN)}`);
          have = Number(st.data?.video_status?.uploading_phase?.bytes_transferred ?? 0);
        } catch { /* diagnostics only */ }
        if (have >= fileSize) {
          console.log(`[instagram] Meta already holds all ${fileSize} bytes (bytes_transferred=${have}); continuing`);
          break;
        }
        if (have > offset && resumes < 3) {
          console.log(`[instagram] Meta holds ${have} bytes (> our failed offset ${offset}); resuming with the remaining ${fileSize - have}`);
          offset = have;
          resumes++;
          continue;
        }
        err = `resumable upload stalled at byte ${offset}/${fileSize}: ${lastChunkErr ?? 'unknown error'}`;
        break;
      }
      offset = fileSize;
      console.log(`[instagram] uploaded ${offset}/${fileSize} bytes (100%)`);
    }
    if (err) {
      console.log(`[instagram] upload attempt ${attempt}/${IG_UPLOAD_ATTEMPTS} failed: ${err}`);
    } else {
      console.log('[instagram] upload accepted: all bytes transferred');
      break; // success - leave the attempt loop and poll processing below
    }
    // Best-effort diagnosis: the container's video_status carries Meta's REAL rejection
    // reason (error code + message + how many bytes actually landed) when the raw rupload
    // response is just a generic ProcessingFailedError. Makes server-side rejections
    // actually debuggable.
    try {
      const st = await jsonFetch(`${BASE_URL}/${containerId}?fields=status_code,video_status&access_token=${encodeURIComponent(IG_ACCESS_TOKEN)}`);
      const vs = st.data?.video_status ?? {};
      const metaErrs = [
        ...(vs.uploading_phase?.errors ?? []),
        ...(vs.processing_phase?.errors ?? []),
      ].map((e) => `${e.code ?? ''} ${e.message ?? ''}`.trim()).filter(Boolean);
      const bt = vs.uploading_phase?.bytes_transferred;
      const xfer = bt != null ? `, bytes_transferred=${bt}` : '';
      if (metaErrs.length) {
        console.log(`[instagram] container ${containerId} (${st.data?.status_code ?? '?'}${xfer}): Meta reports: ${metaErrs.join(' | ')}`);
      }
    } catch { /* diagnostics only - never mask the upload error */ }
    if (attempt < IG_UPLOAD_ATTEMPTS) {
      // Transient/processing failure: the next attempt starts with a FRESH container.
      await new Promise((r) => setTimeout(r, waits[attempt - 1] ?? 60000));
    }
  }
  if (err) throw new Error(`Instagram upload failed: ${err}`);

  // Step 3: Wait until ready
  let waited = 0;
  const timeout = 600, interval = 5;
  while (waited < timeout) {
    const st = igCheck(await jsonFetch(`${BASE_URL}/${containerId}?fields=status_code,status&access_token=${encodeURIComponent(IG_ACCESS_TOKEN)}`));
    const status = st.status_code;
    console.log(`[instagram] processing status: ${status}`);
    if (status === 'FINISHED') break;
    if (status === 'ERROR' || status === 'EXPIRED') throw new Error(`Instagram processing failed: ${JSON.stringify(st)}`);
    await new Promise((r) => setTimeout(r, interval * 1000));
    waited += interval;
  }
  if (waited >= timeout) throw new Error('Instagram timed out waiting for video processing');

  // Step 4: Publish
  const pub = igCheck(await jsonFetch(`${BASE_URL}/${IG_USER_ID}/media_publish`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ creation_id: containerId, access_token: IG_ACCESS_TOKEN }),
  }));
  console.log(`[instagram] published! Media ID: ${pub.id}`);
  return pub.id;
}
