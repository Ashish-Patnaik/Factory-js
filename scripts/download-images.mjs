#!/usr/bin/env node
// download-images.mjs - standalone multi-source AI image extractor (the port of v.py +
// the images.py CLI): wiki / openverse / ddg / stock / imdb.
//
//   node scripts/download-images.mjs --search "donald trump, white house" --limit 20
//   node scripts/download-images.mjs --script output/my_video.txt --limit 40
//   node scripts/download-images.mjs --search "attack on titan" --limit 25 --sources ddg,imdb
import fs from 'node:fs';
import path from 'node:path';
import { availableSources, collect, downloadImages, SEARCH } from '../src/images.js';
import { makeConfig } from '../src/config.js';

function arg(flag) {
  const i = process.argv.indexOf(flag);
  return i !== -1 && i + 1 < process.argv.length ? process.argv[i + 1] : null;
}

const search = arg('--search');
const scriptFile = arg('--script');
const limit = parseInt(arg('--limit') ?? '10', 10);
const out = arg('--out') ?? 'images';
const sources = arg('--sources') ?? 'all';

if (!search && !scriptFile) {
  console.error('pass --search "term, term" or --script <file>');
  process.exit(1);
}

const cfg = makeConfig({ obscura_path: process.env.OBSCURA_PATH ?? null });

let terms = null;
if (search) {
  terms = search.split(',').map((t) => t.trim()).filter(Boolean).map((term) => ({ term, kind: 'concept' }));
} else {
  const script = (scriptFile === '-' ? fs.readFileSync(0, 'utf8') : fs.readFileSync(scriptFile, 'utf8')).trim() || 'general topic';
  const { aiTerms, fallbackTerms } = await import('../src/images.js');
  terms = (await aiTerms(cfg, script)) || fallbackTerms(script);
}

const srcs = await availableSources(sources, cfg);
console.log(`[image] sources: ${srcs.join(', ')}`);
const urls = [];
const seen = new Set();
const target = Math.floor(limit * 1.8) + 4;
outer: for (const t of terms) {
  for (const s of srcs) {
    if (urls.length >= target) break outer;
    try {
      const found = await SEARCH[s](t.term, 8, cfg);
      const fresh = found.filter((u) => !seen.has(u));
      fresh.forEach((u) => seen.add(u));
      urls.push(...fresh);
      if (fresh.length) console.log(`[image] ${s}: +${fresh.length} for "${t.term}" (total ${urls.length})`);
    } catch (e) {
      console.log(`[image] ${s} failed for "${t.term}": ${String(e?.message ?? e).split('\n')[0]}`);
    }
  }
}
if (!urls.length) {
  console.error('no images found');
  process.exit(1);
}
const saved = await downloadImages(urls, path.resolve(out), limit);
console.log(`done: ${saved.length} image(s) in ${path.resolve(out)}`);
