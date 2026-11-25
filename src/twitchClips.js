// twitchClips.js - Twitch Helix API discovery + fast local MP4 clip download
// (twitch-dl primary, pure-Node Twitch GQL fallback; zero cookies, zero Azure IP bans).
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const TWITCH_GQL_CLIENT_ID = 'kimne78kx3ncx6brgo4mv6wki5h1ko';
const CLIP_ACCESS_TOKEN_HASH = '36b89d2507fce29e5ca551df756d27c1cfe079e2609642b4390aa4c35796eb11';

export function hasTwitchCredentials() {
  return Boolean(
    (process.env.TWITCH_CLIENT_ID ?? '').trim() &&
    (process.env.TWITCH_CLIENT_SECRET ?? '').trim(),
  );
}

export function extractTwitchClipSlug(url) {
  const s = String(url ?? '').trim();
  const m = s.match(/clips\.twitch\.tv\/([A-Za-z0-9_-]+)/i)
    ?? s.match(/twitch\.tv\/[^/]+\/clip\/([A-Za-z0-9_-]+)/i);
  return m ? m[1] : null;
}

export function canonicalTwitchClipUrl(slug) {
  return `https://clips.twitch.tv/${slug}`;
}

async function getTwitchAppToken(clientId, clientSecret) {
  const res = await fetch('https://id.twitch.tv/oauth2/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      client_id: clientId,
      client_secret: clientSecret,
      grant_type: 'client_credentials',
    }),
    signal: AbortSignal.timeout(20000),
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    throw new Error(`Twitch OAuth failed (HTTP ${res.status}): ${txt.slice(0, 160)}`);
  }
  const { access_token } = await res.json();
  if (!access_token) throw new Error('Twitch OAuth returned no access_token');
  return access_token;
}

export async function fetchTrendingGtaClips({
  games = (process.env.TWITCH_GAMES || 'Grand Theft Auto VI,Grand Theft Auto V')
    .split(',').map((s) => s.trim()).filter(Boolean),
  lookbackHours = Number(process.env.TWITCH_LOOKBACK_HOURS ?? '48'),
  minDuration = Number(process.env.TWITCH_MIN_DURATION ?? '18'),
  maxDuration = Number(process.env.TWITCH_MAX_DURATION ?? '60'),
  language = process.env.TWITCH_LANGUAGE ?? 'en',
  history = new Set(),
  pickRandom = false,
} = {}) {
  const clientId = (process.env.TWITCH_CLIENT_ID ?? '').trim();
  const clientSecret = (process.env.TWITCH_CLIENT_SECRET ?? '').trim();
  if (!clientId || !clientSecret) {
    throw new Error('Missing TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET env vars');
  }

  const token = await getTwitchAppToken(clientId, clientSecret);
  const headers = {
    'Client-Id': clientId,
    Authorization: `Bearer ${token}`,
  };

  // 1. Resolve Twitch game IDs dynamically
  const gameParams = new URLSearchParams();
  for (const name of games) gameParams.append('name', name);
  const gameRes = await fetch(`https://api.twitch.tv/helix/games?${gameParams}`, {
    headers,
    signal: AbortSignal.timeout(20000),
  });
  if (!gameRes.ok) throw new Error(`Twitch /helix/games failed (HTTP ${gameRes.status})`);
  const { data: gameList = [] } = await gameRes.json();
  if (!gameList.length) throw new Error(`No Twitch games matched: ${games.join(', ')}`);

  // Preserve priority order from `games` (e.g. GTA VI first, GTA V second)
  const order = new Map(games.map((g, i) => [g.toLowerCase(), i]));
  gameList.sort((a, b) => (order.get(a.name.toLowerCase()) ?? 99) - (order.get(b.name.toLowerCase()) ?? 99));

  const seen = new Set();
  const isPosted = (clip) => {
    const canon = canonicalTwitchClipUrl(clip.id);
    return history.has(canon) || history.has(clip.url) || history.has(clip.id);
  };

  // 2. Pull fresh clips (widen lookback from lookbackHours -> 168h if needed)
  const windows = [...new Set([lookbackHours, Math.max(lookbackHours, 168)])];
  const collected = [];

  for (const hrs of windows) {
    const startedAt = new Date(Date.now() - hrs * 3600 * 1000).toISOString();
    for (const game of gameList) {
      const clipParams = new URLSearchParams({
        game_id: game.id,
        started_at: startedAt,
        first: '40',
      });
      const clipRes = await fetch(`https://api.twitch.tv/helix/clips?${clipParams}`, {
        headers,
        signal: AbortSignal.timeout(20000),
      });
      if (!clipRes.ok) {
        console.log(`[twitch] /helix/clips HTTP ${clipRes.status} for ${game.name}; skipping`);
        continue;
      }
      const { data = [] } = await clipRes.json();
      for (const c of data) {
        if (!c?.id || seen.has(c.id) || isPosted(c)) continue;
        if (language && c.language && c.language !== language) continue;
        const dur = Number(c.duration ?? 0);
        if (dur < minDuration || dur > maxDuration) continue;
        seen.add(c.id);
        collected.push({
          source: 'twitch',
          id: c.id,
          url: canonicalTwitchClipUrl(c.id),
          embedUrl: c.url,
          title: String(c.title ?? '').trim() || `${game.name} clip`,
          channel: c.broadcaster_name || 'Twitch',
          gameName: game.name,
          duration: dur,
          views: `${Number(c.view_count ?? 0).toLocaleString('en-US')} views`,
          viewCount: Number(c.view_count ?? 0),
          published: c.created_at ? c.created_at.slice(0, 10) : 'recent',
        });
      }
    }
    if (collected.length >= 6) break;
  }

  // Sort by view count descending; if --random is passed, keep top 25 viral clips and shuffle lightly
  collected.sort((a, b) => b.viewCount - a.viewCount);
  if (pickRandom && collected.length > 5) {
    const topPool = collected.slice(0, 20).sort(() => Math.random() - 0.5);
    return [...topPool, ...collected.slice(20)];
  }
  return collected;
}

function findTwitchDl() {
  const isWin = os.platform() === 'win32';
  const names = isWin ? ['twitch-dl.exe', 'twitch-dl'] : ['twitch-dl'];
  const isFile = (p) => { try { return fs.statSync(p).isFile(); } catch { return false; } };
  const dirs = (process.env.PATH ?? '').split(path.delimiter).filter(Boolean);
  if (!isWin) dirs.push(path.join(os.homedir(), '.local', 'bin'), '/usr/local/bin', '/usr/bin');
  for (const d of dirs) {
    for (const n of names) {
      const p = path.join(d, n);
      if (isFile(p)) return p;
    }
  }
  return null;
}

function runBin(bin, args, timeoutMs = 120000) {
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
      if (code === 0) return resolve(Buffer.concat(out).toString('utf8'));
      const msg = (Buffer.concat(err).toString('utf8') || Buffer.concat(out).toString('utf8'))
        .trim().split('\n').slice(-2).join(' | ');
      reject(new Error(msg || `exit ${code}`));
    });
  });
}

// Pure-Node fallback: ask Twitch's public web GQL for the signed MP4 CDN URL and stream to disk.
async function downloadViaTwitchGql(slug, outPath) {
  const gqlRes = await fetch('https://gql.twitch.tv/gql', {
    method: 'POST',
    headers: {
      'Client-ID': TWITCH_GQL_CLIENT_ID,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      operationName: 'VideoAccessToken_Clip',
      variables: { slug },
      extensions: {
        persistedQuery: { version: 1, sha256Hash: CLIP_ACCESS_TOKEN_HASH },
      },
    }),
    signal: AbortSignal.timeout(20000),
  });
  if (!gqlRes.ok) throw new Error(`Twitch GQL HTTP ${gqlRes.status}`);
  const payload = await gqlRes.json();
  const clip = payload?.data?.clip;
  const qualities = clip?.videoQualities ?? [];
  const sig = clip?.playbackAccessToken?.signature;
  const token = clip?.playbackAccessToken?.value;
  if (!qualities.length || !sig || !token) {
    throw new Error(`Twitch GQL returned no playable MP4 qualities for clip ${slug}`);
  }

  // Pick highest quality <= 1080p (sorted descending by numeric resolution + frameRate)
  const best = [...qualities]
    .filter((q) => q?.sourceURL)
    .sort((a, b) =>
      (parseInt(b.quality, 10) - parseInt(a.quality, 10)) ||
      ((b.frameRate ?? 0) - (a.frameRate ?? 0)))[0];

  const sep = best.sourceURL.includes('?') ? '&' : '?';
  const signedUrl = `${best.sourceURL}${sep}sig=${encodeURIComponent(sig)}&token=${encodeURIComponent(token)}`;

  const dl = await fetch(signedUrl, { signal: AbortSignal.timeout(120000) });
  if (!dl.ok) throw new Error(`Twitch CDN MP4 fetch failed (HTTP ${dl.status})`);
  const buf = Buffer.from(await dl.arrayBuffer());
  if (buf.length < 50_000) throw new Error(`downloaded clip too small (${buf.length} bytes)`);
  fs.writeFileSync(outPath, buf);
  return outPath;
}

export async function downloadTwitchClip(slug, outPath) {
  fs.mkdirSync(path.dirname(path.resolve(outPath)) || '.', { recursive: true });
  if (fs.existsSync(outPath)) fs.rmSync(outPath, { force: true });

  const bin = findTwitchDl();
  if (bin) {
    for (const args of [
      ['download', slug, '--quality', 'source', '--output', outPath],
      ['download', slug, '--output', outPath],
    ]) {
      try {
        await runBin(bin, args, 120000);
        if (fs.existsSync(outPath) && fs.statSync(outPath).size > 50_000) {
          console.log(`[twitch] downloaded ${slug} via twitch-dl (${(fs.statSync(outPath).size / 1e6).toFixed(1)} MB)`);
          return outPath;
        }
      } catch (e) {
        console.log(`[twitch] twitch-dl attempt failed (${String(e?.message ?? e).split('\n')[0]})`);
      }
    }
    console.log('[twitch] falling back to direct Twitch GQL MP4 download...');
  }

  await downloadViaTwitchGql(slug, outPath);
  console.log(`[twitch] downloaded ${slug} via GQL CDN (${(fs.statSync(outPath).size / 1e6).toFixed(1)} MB)`);
  return outPath;
}