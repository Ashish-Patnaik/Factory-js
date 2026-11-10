# AI Content Factory — Node.js Edition

Scrolling-website video & autonomous AI content factory. **A 1:1 port of the Python factory to
Node.js** — same CLI, same pipeline, same output quality — rebuilt to be **lighter, faster and
ffmpeg-free**:

| Concern | Python factory | Node.js factory (this port) |
| --- | --- | --- |
| Rendering / muxing | ffmpeg `image2pipe` + libx264 `medium` (slow, external binary) | **mediabunny + WebCodecs** inside a headless encoder page — hardware-accelerated H.264, pure JS MP4 muxing, no subprocess |
| Instagram killer (`ProcessingFailedError`) | fixed post-hoc via ffmpeg re-encode | **impossible by construction**: frames are converted RGBA→I420 with explicit **limited (tv) range**, AAC 44.1 kHz always muxed, short renders hard-capped at 89 s (`REEL_MAX_SECONDS`) before the API ever sees the file, fresh container per upload attempt |
| TTS | `ptts` (Python runtime, model download) | **phonon-tts**'s own WebAssembly engine running inside Node (`phonon-tts` npm package) |
| ASR | faster-whisper `tiny` | **phonon-2-web** (Rust/WASM port of NVIDIA Parakeet-TDT 0.6B v3) with exact word timestamps — prebuilt engine ships in `vendor/` |
| LLM | `google-genai` (Python) | **`@google/genai`** (`ai.interactions.create` → `output_text`, with `models.generateContent` fallback) |
| Stealth browser | Obscura via CDP | same (Obscura over CDP via `playwright-core`) — drop `obscura.exe` + `obscura-worker.exe` into `obscura/` |
| Media probing | ffprobe | **mediabunny** (pure JS) |
| Captions | .ass burned by ffmpeg | same TikTok pop/karaoke engine, burned on an OffscreenCanvas (the .ass writer is kept for `--keep-files`) |

## Install

```powershell
# Node.js 20.11+ required (Node 22 LTS recommended)
npm install
npx playwright install chromium     # page-capture browser (Chrome/Edge also fine)
npm run doctor                      # verify everything
```

Optional:
- **Obscura** — for sites that block headless Chrome: put `obscura.exe` + `obscura-worker.exe`
  into the `obscura/` folder (or set `OBSCURA_PATH` / `--obscura-path`).
  <https://github.com/h4ckf0r0day/obscura/releases>
- **ASR rebuild** — a prebuilt phonon-2-web engine ships in `vendor/phonon-2-web/web/pkg-st`.
  If it is missing: `npm run setup:asr` (needs stable Rust once).

## API keys / credentials

`.env`, `env.txt` or `insta env.txt` in the factory folder are auto-loaded (same rules as the
Python factory; `export ` and `$env:` prefixes are stripped):

| Variable | Used for |
| --- | --- |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | AI scripts, metadata, image search terms — <https://aistudio.google.com/apikey> |
| `IG_ACCESS_TOKEN`, `IG_USER_ID`, `IG_GRAPH_VERSION` | Instagram Reels publishing (Graph API) |
| `YT_CLIENT_SECRET_FILE`, `YT_TOKEN_FILE` | YouTube OAuth (`client_secrets.json`, token stored as `token.json`) |
| `PEXELS_API_KEY` / `UNSPLASH_ACCESS_KEY` / `PIXABAY_API_KEY` | optional stock photo sources |
| `REEL_MAX_SECONDS` (default 89) | short-video cap that keeps renders under Instagram's 90 s API reel limit |
| `OBSCURA_PATH` | explicit path to the Obscura binary |

## Usage (identical CLI to the Python factory)

```text
node main.js --auto                                        discover URL, render 9:16 mobile video + metadata
node main.js --auto --long                                 autonomous long-form: 5-7 min 16:9 YouTube video
node main.js --auto --long --image                         narrated image slideshow instead of a scroll
node main.js --auto --long --image --avatar                slideshow + looping talking-character gif
node main.js --entertainment                               film/TV/anime/comics story -> Shorts & Reels
node main.js --entertainment --long                        entertainment video essay -> 5-7 min
node main.js --auto --publish all                          scout -> render -> post YT Shorts & IG Reels
node main.js https://example.com demo --mobile --narrate   single URL in mobile view with AI voice-over
node main.js https://example.com demo --long               long-form: 5-7 min script, 16:9 slow-scroll
node main.js https://example.com demo                      plain desktop scroll video
```

Run `node main.js --help` for every flag (all Python flags exist, including `--mode
virtual|stepped|realtime`, `--browser obscura`, `--avatar*`, `--cap-*`, `--separate`,
`--keep-files`, `--publish`, `--yt-privacy`, ...).

Handy scripts:

```text
npm run doctor                       # environment + codec + credentials check
npm run setup:asr                    # (re)build the phonon-2-web ASR engine
node scripts/download-images.mjs --search "attack on titan" --limit 25 --sources ddg,imdb
```

## How a render works (and why it's fast)

1. **Script** — Gemini (via `@google/genai`) writes the narration in your `persona.md` voice;
   over-length scripts are automatically shortened (under-length `--long` scripts expanded).
2. **Voice** — phonon-tts WASM synthesizes 24 kHz speech fully on-device (`--voice`, `--tts-*`).
3. **Captions** — phonon-2-web WASM transcribes with word-level timestamps (fully on-device),
   `captions.js` groups words into TikTok-style lines.
4. **Capture** — Playwright drives Chromium (or Obscura over CDP): virtual-time frame-exact
   capture (default), stepped capture for Obscura, or realtime screencast.
5. **Encode** — every frame is composed on an OffscreenCanvas in the encoder page (fit/pad on
   white, captions, avatar gif) and encoded straight to **H.264 yuv420p (limited range)** via
   WebCodecs while **mediabunny** muxes MP4 (+faststart) with **AAC 44.1 kHz** audio
   (native WebCodecs AAC on Chrome/Edge, automatic WASM AAC encoder elsewhere).
   The final MP4 streams back to Node — no ffmpeg, no temp frame pipes.

On a typical laptop this renders a 60 s Short significantly faster than real time
(hardware encoding), with a fraction of the Python factory's disk churn.

## Instagram publishing (the ProcessingFailedError story)

Meta's transcoder rejects reels that violate its ingest spec **after** all bytes are uploaded —
the error says `ProcessingFailedError, retriable: False`. This factory defends in depth:

1. **By construction** — encoder output is H.264 yuv420p limited range, AAC 44.1 kHz, and short
   videos are capped at `REEL_MAX_SECONDS` (89 s) before rendering.
2. **Before upload** — the file is probed again (mediabunny) and auto-healed (trim over-length,
   report missing audio) when needed.
3. **Upload protocol** — resumable single-shot upload with the exact documented header set
   (`Authorization`, `offset`, `file_size`), fresh container per attempt (a transcoder-failed
   container can never succeed), retry only on transient/processing failures, status polling,
   then `media_publish`.

## Project layout

```
main.js                    CLI + pipeline (port of main.py)
src/
  config.js                Config + RATIOS/SPEEDS (config.py)
  utils.js                 env loading, die, URLs, mediabunny probe (utils.py)
  captions.js              Word/groupWords/.ass + canvas caption data (captions.py)
  scroll.js                scroll planning math (scroll.py)
  speech.js                narration pipeline: script -> TTS -> ASR (speech.py)
  gemini.js                @google/genai client
  tts.js                   phonon-tts WASM adapter (Node)
  asr.js                   phonon-2-web WASR adapter (Node)
  scrape.js                page text/links via Obscura/Chromium (scrape.py)
  browser.js               Browser + ObscuraBrowser, popup killer, virtual-time shim (browser.py)
  recorder.js              virtual/stepped/realtime capture + Pacer (recorder.py)
  slides.js                slideshow timeline planning (slides.py)
  images.js                multi-source AI image sourcing (images.py)
  agent.js                 autonomous agent: hubs -> pick -> package (agent.py)
  publish.js               YouTube + Instagram publishing (publish.py)
  encoder/
    host.js                Node driver for the encoder page
    session.js             one encode job (frames + audio + pull)
    page.js                browser-side composition + mediabunny/WebCodecs encode
scripts/                   setup-asr, doctor, download-images
vendor/phonon-2-web/       ASR engine (prebuilt web/pkg-st + upstream source)
fonts/Anton-Regular.ttf    caption font
persona.md, sources.md, sources_entertainment.md   reused as-is from the Python factory
```

## Notes

- `--whisper-model` is accepted for compatibility but ignored: ASR is phonon-2-web now.
- `--tts-quant q4k` maps to `q8` (the wasm engine ships q8/f32).
- The default voice (`Easton`) uses the checkpoint's default voice embedding unless the
  requested voice exists in the model repo.
- `output/history.json` prevents re-posting the same URL, exactly like the Python factory.
