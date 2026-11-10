// gemini.js - LLM access via the official @google/genai SDK.
// Primary path uses the interactions API exactly as documented at
// aistudio.google.com/docs/get-started?cqlanguage=javascript:
//   const ai = new GoogleGenAI({});
//   const interaction = await ai.interactions.create({ model, input });
//   interaction.output_text
// with a fallback to ai.models.generateContent for older/other account shapes.
import { GoogleGenAI } from '@google/genai';

let client = null;
const FALLBACK_MODEL = 'gemini-3.8-flash'; // used if the configured model name is rejected

export function checkLlm() {
  if (!(process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY)) {
    console.error('error: Gemini API key missing. Get one at https://aistudio.google.com/apikey then run:');
    console.error('  PowerShell: $env:GEMINI_API_KEY="your_key"     (permanent: setx GEMINI_API_KEY "your_key")');
    console.log('error: Gemini API key missing. Get one at https://aistudio.google.com/apikey (set GEMINI_API_KEY)');
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

export async function llm(cfg, prompt) {
  const ai = getClient();
  try {
    const interaction = await ai.interactions.create({ model: cfg.llm_model, input: prompt });
    const text = interaction.output_text ?? interaction.text ?? '';
    if (text) return text;
    throw new Error('empty interactions response');
  } catch (e1) {
    // Interactions may be unavailable for some account/SDK shapes, or the model name may not
    // exist: fall back to generateContent, then to the current documented flash model.
    try {
      const res = await ai.models.generateContent({ model: cfg.llm_model, contents: prompt });
      const text = res.text ?? '';
      if (text) return text;
      throw new Error('empty generateContent response');
    } catch (e2) {
      const msg = firstLine(e2);
      if (/not found|unsupported|invalid model|does not exist|not supported/i.test(msg)) {
        try {
          const res = await ai.models.generateContent({ model: FALLBACK_MODEL, contents: prompt });
          if (res.text) {
            console.log(`[ai] model '${cfg.llm_model}' unavailable (${msg}); used ${FALLBACK_MODEL}`);
            return res.text;
          }
        } catch { /* fall through */ }
      }
      throw new Error(`Gemini request failed: ${msg || firstLine(e1)}`);
    }
  }
}
