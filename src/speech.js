// speech.js - AI script (Gemini) -> TTS (phonon-tts wasm, on-device) -> ASR (phonon-2-web, word timestamps).
import fs from 'node:fs';
import { Word, ensureFont } from './captions.js';
import { checkLlm, llm } from './gemini.js';
import { PhononTTS, readWav } from './tts.js';
import * as asr from './asr.js';

export const WPS = 2.5; // initial guess: spoken words per second (re-measured after the first TTS pass)
const ttsCache = new Map();

export { checkLlm };

export async function loadPersona(path = 'persona.md') {
  try {
    return fs.readFileSync(path, 'utf8').trim();
  } catch {
    return '';
  }
}

export function cleanScript(text, maxWords = null) {
  // Strip markdown/emoji/labels, flatten to one paragraph, cap length at a sentence boundary.
  let t = String(text ?? '');
  t = t.replace(/```[\s\S]*?```/g, ' ');
  t = t.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1');                       // [text](url) -> text
  t = t.replace(/^\s*(#{1,6}|[-*+>]|\d+[.)])\s+/gm, '');               // headings, bullets, quotes
  t = t.replace(/^\s*(script|narration|voiceover|hook)\s*:\s*/gim, '');
  t = t.replace(/[*_`#~|]/g, '');
  t = t.replace(/[\u{1F000}-\u{1FFFF}\u{2600}-\u{27BF}\u{FE0F}]/gu, ''); // emoji
  t = t.replace(/\s+/g, ' ').trim();
  if (maxWords && t.split(' ').length > maxWords) {
    const cut = t.split(' ').slice(0, maxWords).join(' ');
    let end = -1;
    for (const c of '.!?') end = Math.max(end, cut.lastIndexOf(c));
    t = end > cut.length * 0.6 ? cut.slice(0, end + 1) : cut.replace(/[,;:]$/, '') + '.';
  }
  return t;
}

export const RULES =
  'Strict Narration Guidelines:\n' +
  '1. Open with a scroll-stopping hook in the first sentence (bold claim, surprising fact, or sharp question).\n' +
  '2. Deliver high-signal, concrete information in short spoken sentences.\n' +
  '3. End with a brief, punchy call to action or question.\n' +
  '4. Write plain spoken English ONLY: no markdown, no bullet points, no headings, no emojis, no hashtags, ' +
  'no stage directions, no speaker labels, and NEVER read out URLs or code syntax.\n' +
  '5. HARD LENGTH CAP: Use at most {words} words (strictly under {secs} seconds when spoken). ' +
  'Output ONLY the spoken script text.';

export async function writeScript(cfg, page, words) {
  const rules = RULES.replaceAll('{words}', String(words)).replaceAll('{secs}', String(cfg.script_seconds));
  const persona = await loadPersona(cfg.persona_file ?? 'persona.md');
  const personaBlock = persona ? `\n\nFollow this Creator Persona & Voice strictly:\n${persona}\n` : '';
  const extra = cfg.prompt ? `\nExtra instructions from the creator: ${cfg.prompt}` : '';
  const ent = cfg.entertainment;
  let p;
  if (cfg.script_seconds > 90) { // --long: long-form YouTube narration (5-7 minutes)
    const mins = Math.max(1, Math.round(cfg.script_seconds / 60));
    let task;
    if (ent) {
      task = `You are a long-form YouTube scriptwriter for film, TV and anime video essays. Write a spoken ` +
        `narration of about ${mins} minutes (aim for ${words} words) for a 16:9 YouTube video that brings ` +
        `the story below to life: hook in the first sentence, then walk through the plot, characters, ` +
        `key moments and your honest opinions one by one in a natural order with smooth transitions, ` +
        `connect the story to a positive mindset, good habit or clear lesson the viewer can use, and ` +
        `end with a takeaway and a call to action. It must stay engaging for the full ${mins} minutes.`;
    } else {
      task = `You are a long-form YouTube scriptwriter. Write a spoken narration of about ${mins} minutes ` +
        `(aim for ${words} words) for a 16:9 desktop video that walks the viewer through the topic below ` +
        `in depth: hook in the first sentence, then cover the most interesting points one by one in a ` +
        `natural order with smooth transitions between topics, and end with a clear takeaway and a call ` +
        `to action. It must stay engaging for the full ${mins} minutes.`;
    }
    p = cfg.text
      ? `${task} Rewrite the text below into this long-form narration, keeping its meaning and facts.` +
        `${personaBlock}\n${rules}${extra}\n\nText:\n${cfg.text}`
      : `${task}${personaBlock}\n${rules}${extra}\n\nPage content:\n${page}`;
  } else if (cfg.text) { // --enhance
    p = `You are a short-form video scriptwriter. Rewrite the text below into an engaging narration for a vertical ` +
      `video, keeping its meaning and facts.${personaBlock}\n${rules}${extra}\n\nText:\n${cfg.text}`;
  } else if (ent) { // --entertainment short: film/show/anime story for Shorts & Reels
    p = `You are a short-form video scriptwriter for film, TV and anime content. Write a narration for a vertical ` +
      `video (TikTok/Reels/Shorts) about the story below: name the title, hit the most exciting moment, plot ` +
      `twist or opinion fast, and land one quick takeaway tied to a mindset, habit or lesson.` +
      `${personaBlock}\n${rules}${extra}\n\nPage content:\n${page}`;
  } else {
    p = `You are a short-form video scriptwriter. Write a narration for a vertical video (TikTok/Reels/Shorts) that ` +
      `explains the web page below and gives the most interesting, useful takeaway from it.${personaBlock}\n${rules}${extra}` +
      `\n\nPage content:\n${page}`;
  }
  return cleanScript(await llm(cfg, p), words);
}

export async function synthesize(cfg, text, wav) {
  const key = [cfg.tts_model, cfg.tts_lang, cfg.tts_quant].join('|');
  if (!ttsCache.has(key)) ttsCache.set(key, await PhononTTS.load(cfg));
  const tts = ttsCache.get(key);
  return tts.save(wav, text, { voice: cfg.voice });
}

// transcribe the generated voice-over into word timings (phonon-2-web)
export async function transcribe(cfg, wavPath) {
  const { pcm, sampleRate } = readWav(wavPath);
  const words = await asr.transcribe({ pcm, sampleRate });
  return words.map((w) => new Word(w.text, w.start, w.end));
}

// narration(cfg, wav) -> { script, secs, words }
export async function narration(cfg, wav) {
  let ai = cfg.narrate || cfg.enhance;
  if (ai) checkLlm();
  let page = null;
  if (cfg.narrate && !cfg.text) {
    const { pageText } = await import('./scrape.js');
    console.log(`[ai] visiting ${cfg.url}`);
    page = await pageText(cfg);
  }
  let budget = Math.floor(cfg.script_seconds * WPS);
  let script = '';
  let secs = 0;
  let words = [];
  for (let attempt = 1; attempt <= 2; attempt++) {
    if (ai) {
      console.log(`[ai] writing script with ${cfg.llm_model} (<= ${budget} words)`);
      script = await writeScript(cfg, page, budget);
    } else {
      script = cleanScript(cfg.text);
    }
    if (!script) throw new Error('the AI returned an empty script');
    console.log(`[script] ${script}`);
    if (cfg.script_only) return { script, secs: 0.0, words: [] };
    secs = await synthesize(cfg, script, wav);
    console.log(`[tts] ${secs.toFixed(1)}s of speech (${script.split(/\s+/).length} words)`);
    if ((ai || cfg.text) && secs > cfg.script_seconds && attempt === 1) {
      // Shorten automatically if over the length cap
      budget = Math.floor(cfg.script_seconds * script.split(/\s+/).length / secs * 0.92);
      console.log(`[ai] speech is ${secs.toFixed(1)}s (> ${cfg.script_seconds}s cap), shortening to ~${budget} words`);
      ai = true;
      if (!page && !cfg.text) cfg.text = script;
      continue;
    }
    if ((ai || cfg.text) && cfg.long && attempt === 1 && secs < cfg.script_seconds * 0.7) {
      // --long: the script landed well under the 5-7 min target; expand it once
      budget = Math.floor(cfg.script_seconds * script.split(/\s+/).length / Math.max(secs, 1.0) * 0.97);
      console.log(`[ai] speech is only ${secs.toFixed(1)}s (target ${cfg.script_seconds}s), expanding to ~${budget} words`);
      cfg.text = script;
      ai = true;
      continue;
    }
    if (secs > cfg.script_seconds) {
      console.log(`warning: narration is ${Math.round(secs)}s, longer than the ${cfg.script_seconds}s target`);
    }
    break;
  }
  if (!cfg.captions) return { script, secs, words: [] };
  await ensureFont(cfg.cap_font); // make sure Anton is on disk before the encoder burns captions
  words = await transcribe(cfg, wav);
  if (!words.length) throw new Error('ASR found no speech in the generated audio');
  console.log(`[asr] ${words.length} words timed`);
  return { script, secs, words };
}
