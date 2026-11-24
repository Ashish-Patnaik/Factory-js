// gemini.js - LLM access via @google/genai with automatic retry + OpenRouter fallback.
import { GoogleGenAI } from '@google/genai';
import { sleep } from './utils.js';

let client = null;
const GEMINI_FALLBACK_MODELS = ['gemini-2.5-flash', 'gemini-3.8-flash'];
const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';
const DEFAULT_OPENROUTER_MODEL = 'baidu/ernie-5.5-flash-preview:free';

function hasGeminiKey() {
  return Boolean(process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY);
}

function hasOpenRouterKey() {
  return Boolean(process.env.OPENROUTER_API_KEY);
}

export function checkLlm() {
  if (!hasGeminiKey() && !hasOpenRouterKey()) {
    console.error('error: No LLM API key found. Set GEMINI_API_KEY and/or OPENROUTER_API_KEY.');
    console.log('error: No LLM API key found (set GEMINI_API_KEY and/or OPENROUTER_API_KEY)');
    process.exit(1);
  }
}

function getClient() {
  if (!client) client = new GoogleGenAI({});
  return client;
}

function firstLine(e) {
  return String(e?.message ?? e).split('\n')[0];
}

function isTransientError(msg) {
  return /429|500|502|503|504|UNAVAILABLE|RESOURCE_EXHAUSTED|rate.?limit|high demand|overloaded|timeout|fetch failed|ECONNRESET|ETIMEDOUT/i.test(msg);
}

function isModelUnavailable(msg) {
  return /not found|unsupported|invalid model|does not exist|not supported/i.test(msg);
}

async function callGeminiOnce(ai, model, prompt, useInteractions = true) {
  if (useInteractions) {
    try {
      const interaction = await ai.interactions.create({ model, input: prompt });
      const text = interaction.output_text ?? interaction.text ?? '';
      if (text) return text;
    } catch {
      // Fall through to generateContent
    }
  }
  const res = await ai.models.generateContent({ model, contents: prompt });
  const text = res.text ?? '';
  if (!text) throw new Error('empty generateContent response');
  return text;
}

async function callGeminiWithRetry(cfg, prompt) {
  const ai = getClient();
  const models = [...new Set([cfg.llm_model, ...GEMINI_FALLBACK_MODELS].filter(Boolean))];
  let lastErr = '';

  for (let mIdx = 0; mIdx < models.length; mIdx++) {
    const model = models[mIdx];
    const maxAttempts = mIdx === 0 ? 3 : 1;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        const text = await callGeminiOnce(ai, model, prompt, mIdx === 0 && attempt === 1);
        if (mIdx > 0) console.log(`[ai] primary Gemini model failed; used fallback Gemini model ${model}`);
        return text;
      } catch (e) {
        lastErr = firstLine(e);
        if (isModelUnavailable(lastErr)) break; // try next Gemini model immediately
        if (isTransientError(lastErr) && attempt < maxAttempts) {
          const waitMs = attempt * 3000;
          console.log(`[ai] Gemini transient error (${lastErr}); retrying in ${waitMs / 1000}s...`);
          await sleep(waitMs);
          continue;
        }
        break;
      }
    }
  }
  throw new Error(lastErr || 'Gemini request failed');
}

function stripReasoningTags(text) {
  // Some reasoning models wrap chain-of-thought in <think>...</think>
  return String(text ?? '').replace(/<think>[\s\S]*?<\/think>\s*/gi, '').trim();
}

async function callOpenRouter(prompt) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error('OPENROUTER_API_KEY not set');

  const model = process.env.OPENROUTER_MODEL || DEFAULT_OPENROUTER_MODEL;
  const headers = {
    'Authorization': `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
    'X-Title': 'AI Content Factory',
  };
  if (process.env.OPENROUTER_REFERER) headers['HTTP-Referer'] = process.env.OPENROUTER_REFERER;

  let lastErr = '';
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const resp = await fetch(OPENROUTER_URL, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model,
          messages: [{ role: 'user', content: prompt }],
          temperature: 0.7,
        }),
        signal: AbortSignal.timeout(90_000),
      });

      const raw = await resp.text();
      let data = null;
      try { data = JSON.parse(raw); } catch { /* non-JSON error body */ }

      if (!resp.ok) {
        const msg = data?.error?.message || `HTTP ${resp.status}: ${raw.slice(0, 200)}`;
        throw new Error(msg);
      }

      const content = stripReasoningTags(data?.choices?.[0]?.message?.content ?? '');
      if (!content) throw new Error('empty OpenRouter response');
      return content;
    } catch (e) {
      lastErr = firstLine(e);
      if (attempt < 3 && isTransientError(lastErr)) {
        const waitMs = attempt * 3000;
        console.log(`[ai] OpenRouter transient error (${lastErr}); retrying in ${waitMs / 1000}s...`);
        await sleep(waitMs);
        continue;
      }
      break;
    }
  }
  throw new Error(`OpenRouter request failed (${model}): ${lastErr}`);
}

export async function llm(cfg, prompt) {
  let geminiErr = null;

  if (hasGeminiKey()) {
    try {
      return await callGeminiWithRetry(cfg, prompt);
    } catch (e) {
      geminiErr = firstLine(e);
      if (hasOpenRouterKey()) {
        const orModel = process.env.OPENROUTER_MODEL || DEFAULT_OPENROUTER_MODEL;
        console.log(`[ai] Gemini unavailable (${geminiErr}); falling back to OpenRouter (${orModel})`);
      }
    }
  }

  if (hasOpenRouterKey()) {
    return await callOpenRouter(prompt);
  }

  throw new Error(`Gemini request failed: ${geminiErr || 'GEMINI_API_KEY not set'}`);
}