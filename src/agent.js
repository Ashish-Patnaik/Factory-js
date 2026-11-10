// agent.js - autonomous AI agent: surfs source hubs, selects trending URLs, writes a
// persona-aligned script + social metadata package.
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { pageLinks, pageText } from './scrape.js';
import { checkLlm, llm } from './gemini.js';
import { cleanScript, loadPersona, WPS } from './speech.js';
import { validateUrl } from './utils.js';

const HISTORY_FILE = path.join('output', 'history.json');

export function loadHistory() {
  try {
    return new Set(JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf8')));
  } catch {
    return new Set();
  }
}

export function saveHistory(url) {
  fs.mkdirSync(path.dirname(HISTORY_FILE) || '.', { recursive: true });
  const hist = [...loadHistory()];
  if (!hist.includes(url)) hist.push(url);
  fs.writeFileSync(HISTORY_FILE, JSON.stringify(hist.slice(-500), null, 2) + '\n');
}

export function parseSources(path_ = 'sources.md') {
  if (!fs.existsSync(path_)) {
    throw new Error(`Sources file not found: ${path_}. Create ${path_} with one hub URL per line.`);
  }
  const urls = [];
  for (const line of fs.readFileSync(path_, 'utf8').split(/\r?\n/)) {
    const l = line.trim();
    if (!l || l.startsWith('#')) continue;
    for (const m of l.matchAll(/https?:\/\/[^\s)>\]]+/g)) urls.push(validateUrl(m[0]));
  }
  if (!urls.length) throw new Error(`No valid http/https URLs found inside ${path_}`);
  return urls;
}

function extractJson(raw) {
  let s = String(raw ?? '').trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
  const m = s.match(/\{[\s\S]*\}/);
  return JSON.parse(m ? m[0] : s);
}

export async function discoverBestUrl(cfg, sourcesFile = 'sources.md', pickRandom = false) {
  // Surf hub URLs in sources.md (or just 1 random hub if --random), collect nested links, and pick the best story.
  checkLlm();
  let hubs = parseSources(sourcesFile);
  const history = loadHistory();
  const persona = await loadPersona(cfg.persona_file);

  if (pickRandom) {
    hubs = [...hubs].sort(() => Math.random() - 0.5);
    console.log(`[agent] --random active: picked random source hub -> ${hubs[0]}`);
  }

  const candidates = [];
  const seen = new Set(history);
  for (const hub of hubs) {
    console.log(`[agent] surfing hub: ${hub}`);
    let nested = [];
    try {
      nested = await pageLinks(cfg, hub);
    } catch (e) {
      console.log(`[agent] hub unreachable (${String(e?.message ?? e).split('\n')[0]}), skipping to the next hub`);
      continue;
    }
    let added = 0;
    for (const item of nested) {
      const u = item.url;
      if (!seen.has(u)) {
        seen.add(u);
        candidates.push(item);
        added += 1;
      }
      if (added >= 25) break; // cap per hub so token usage stays tiny
    }
    if (!nested.length && !seen.has(hub)) candidates.push({ url: hub, title: hub, source_hub: hub });

    // If --random is set, stop as soon as we have fresh candidates from the single randomly chosen hub
    if (pickRandom && candidates.length) break;
  }

  if (!candidates.length) {
    throw new Error('No new candidate URLs found across sources (or all have been posted already).');
  }

  const listing = candidates.slice(0, 120)
    .map((c, i) => `${i}. [${c.title}](${c.url}) (from ${c.source_hub})`).join('\n');

  const longform = cfg.script_seconds > 90; // --long mode
  const fmt = longform ? 'long-form 16:9 YouTube video (5-7 minutes)' : 'vertical short-form video (YouTube Shorts & Instagram Reels)';
  const view = longform ? 'desktop view' : 'mobile view';

  let arena, focus;
  if (cfg.entertainment) {
    arena = 'entertainment (films, TV shows, anime, comics)';
    focus = '1. Represent a genuinely exciting entertainment story: film/TV/anime/comics news, trailers, plot ' +
      'moments, character arcs, release dates, fan theories or strong opinions worth sharing.\n' +
      '2. Have rich story text on the landing page (plot summary, moments, quotes, opinions) so there is ' +
      'plenty of material to narrate.\n' +
      '3. Prefer recognizable titles and stories that can naturally be connected to a positive mindset, ' +
      'good habits or a clear lesson for the viewer.';
  } else {
    arena = 'tech';
    focus = '1. Represent a genuinely exciting, newly trending repo, tool, engineering deep-dive, major AI/tech ' +
      'story, or a big tech-meets-entertainment story (AI in films, streaming tech, anime industry...).\n' +
      `2. Have rich visual/text content on the landing page that will look great when scrolled in ${view}.\n` +
      '3. Avoid generic homepages, login walls, or thin discussion threads without substance.';
  }

  const prompt = `You are an autonomous ${arena} content director scouting for the most viral, high-signal topic for a ${fmt}.

Creator Persona:
${persona}

Candidate Links Discovered Today:
${listing}

Select the top 3 best links that:
${focus}

Return ONLY valid JSON in this exact format:
{"picks": [0, 5, 12], "reason": "One sentence explaining why the #1 pick is the best hook."}`;

  const data = extractJson(await llm(cfg, prompt));
  let picks = (data.picks ?? []).filter((p) => Number.isInteger(p) && p >= 0 && p < candidates.length);
  if (!picks.length) picks = [0];

  console.log(`[agent] AI selection reason: ${data.reason ?? 'Top trending topic'}`);
  for (const idx of picks) {
    const chosen = candidates[idx];
    const url = chosen.url;
    console.log(`[agent] inspecting selected URL: ${url} (${chosen.title})`);
    try {
      const content = await pageText(cfg, url);
      if (content.length >= 250) return { url, title: chosen.title, content };
      console.log(`[agent] page had too little readable text (${content.length} chars), trying next pick...`);
    } catch (e) {
      console.log(`[agent] failed to read ${url} (${e}), trying next pick...`);
    }
  }

  // All AI picks were unreadable (empty pages or anti-bot challenges): walk a few more
  // candidates linearly instead of dying on the first pick.
  const tried = new Set(picks.map((p) => candidates[p]));
  for (const c of candidates.filter((x) => !tried.has(x)).slice(0, 12)) {
    try {
      const content = await pageText(cfg, c.url);
      if (content.length >= 250) {
        console.log(`[agent] AI picks unreadable; fell back to ${c.url} (${c.title})`);
        return { url: c.url, title: c.title, content };
      }
    } catch { /* challenged or empty - try the next candidate */ }
  }
  throw new Error('no readable candidate found (anti-bot challenges or empty pages everywhere) - ' +
    'add more sources to sources.md or drop the protected ones');
}

export async function generateContentPackage(cfg, url, pageContent) {
  // Generate the spoken script + social metadata using persona.md.
  // Long-form (--long) packages are built for YouTube long-form reach only (no Instagram block);
  // short packages also include the Instagram Reels caption.
  checkLlm();
  const persona = await loadPersona(cfg.persona_file);
  const maxWords = Math.floor(cfg.script_seconds * WPS);
  const extra = cfg.prompt ? `\nCreator Extra Instructions: ${cfg.prompt}\n` : '';
  const ent = cfg.entertainment;

  const longform = cfg.script_seconds > 90; // --long mode
  let role, lengthRule, ytTitleRule, ytDescRule, ytTagsRule, storyRules = '';
  if (longform) {
    const mins = Math.max(1, Math.round(cfg.script_seconds / 60));
    let structure;
    if (ent) {
      role = 'autonomous long-form YouTube creator of film/TV/anime video essays and social media strategist';
      structure = '- Structure it like a video essay: hook in sentence 1, then walk through the story\'s key ' +
        'moments, characters and opinions in a natural order with smooth transitions, connect the ' +
        'story to a positive mindset, good habit or clear lesson the viewer can take away, then a ' +
        'closing takeaway and CTA.';
    } else {
      role = 'autonomous long-form YouTube video creator and social media strategist';
      structure = '- Structure it like a mini documentary: strong hook, deep body that walks through the ' +
        'page\'s key points in a natural order with smooth transitions, then a takeaway and CTA.';
    }
    lengthRule = `- The narration script MUST run for about ${mins} minutes when spoken ` +
      `(MAXIMUM ${maxWords} words, target ${Math.floor(maxWords * 0.85)} words).\n${structure}`;
    ytTitleRule = 'SEO-friendly long-form YouTube title under 100 chars that names the main keyword/title (no #Shorts)';
    ytDescRule = `Search-optimized 4-6 sentence description for maximum reach: hook line first, what the ` +
      `viewer will learn, the main keywords used naturally, ending with 'Source: ${url}'`;
    ytTagsRule = '8-12 lowercase search keyword phrases (2-4 words each) a viewer would type to find this video, ordered by search volume';
  } else {
    role = 'autonomous short-form video creator and social media strategist';
    lengthRule = `- The narration script MUST be spoken in under ${cfg.script_seconds} seconds ` +
      `(MAXIMUM ${maxWords} words, target ${Math.floor(maxWords * 0.85)} words).`;
    ytTitleRule = 'Punchy curiosity-gap title under 70 chars';
    ytDescRule = `2-3 sentence high-retention description matching the persona. Include source: ${url}`;
    ytTagsRule = '5-8 lowercase search keyword phrases viewers would type';
  }

  let ytHashtagsJson, igTitleRule, igHashtagsJson;
  if (ent) {
    ytHashtagsJson = longform ? '["#anime", "#film", "#entertainment"]' : '["#Shorts", "#anime", "#film", "#entertainment"]';
    igTitleRule = 'Scroll-stopping first line hook naming the film/show/anime for the Reel caption';
    igHashtagsJson = '["#reels", "#anime", "#film", "#tvshows", "#movies", "#entertainment", "#comicbooks", "#otaku"]';
    storyRules = '- Tell it like a story: name the film/show/anime/comic and its characters, walk through the ' +
      'key moments vividly, and share a clear opinion.\n' +
      '- Connect the story to something useful: a mindset, habit or lesson the viewer can apply ' +
      '(keep it natural, not preachy).\n';
  } else {
    ytHashtagsJson = longform
      ? '["#AI", "#SoftwareEngineering", "#OpenSource", "#Coding", "#Tech"]'
      : '["#Shorts", "#AI", "#SoftwareEngineering", "#OpenSource", "#Coding"]';
    igTitleRule = 'Scroll-stopping first line hook for the Reel caption';
    igHashtagsJson = '["#reels", "#coding", "#ai", "#tech", "#softwareengineer", "#opensource", "#buildinpublic"]';
  }

  const igBlock = longform ? '' : `,
  "instagram": {
    "title": "${igTitleRule}",
    "description": "Engaging multi-line caption body written in the persona's voice explaining why this matters + a question at the end to drive comments.",
    "hashtags": ${igHashtagsJson}
  }`;

  const prompt = `You are an ${role}.

=== CREATOR PERSONA & STYLE (FOLLOW STRICTLY) ===
${persona}

=== STRICT NARRATION & SCRIPT GUIDELINES ===
${lengthRule}
- Open with an immediate, scroll-stopping hook in sentence 1.
${storyRules}- Write plain spoken English ONLY for TTS: NO markdown, NO bullet points, NO emojis, NO hashtags, NO stage directions, and NEVER read out URLs.
- Keep sentences short, punchy, and natural when spoken aloud.
${extra}
=== SOURCE WEBPAGE (${url}) ===
${pageContent}

Produce the complete content package as a single valid JSON object with these exact keys:
{
  "slug": "short_kebab_case_filename_slug",
  "script": "The plain-text spoken TTS narration script under ${maxWords} words...",
  "youtube": {
    "title": "${ytTitleRule}",
    "description": "${ytDescRule}",
    "tags": ["${ytTagsRule}"],
    "hashtags": ${ytHashtagsJson},
    "category": "tech or entertainment - exactly one, whichever fits THIS story"
  }${igBlock}
}
Return ONLY the JSON object, with no surrounding markdown or commentary.`;

  const pkg = extractJson(await llm(cfg, prompt));
  pkg.script = cleanScript(pkg.script ?? '', maxWords);
  const yt = pkg.youtube ?? {};
  if (!Array.isArray(yt.tags) || !yt.tags.length) {
    yt.tags = (yt.hashtags ?? []).map((h) => String(h).replace(/^#/, ''));
  }
  pkg.youtube = yt;
  pkg.instagram = longform ? null : (pkg.instagram ?? null);
  pkg.url = url;
  pkg.created_at = new Date().toISOString().replace('T', ' ').slice(0, 19);
  pkg.long_form = longform;
  return pkg;
}
