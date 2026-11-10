// config.js - the factory Config (1:1 port of config.py defaults).
export const RATIOS = { '9:16': [1080, 1920], '16:9': [1920, 1080], '1:1': [1080, 1080] };
export const SPEEDS = { slow: 150, normal: 300, fast: 600 }; // CSS px per second

export function defaultConfig(overrides = {}) {
  return {
    url: '',
    // --- scroller video
    width: 1920,
    height: 1080,
    viewport: null,        // [w, h] | null (defaults to output size, or 360x640 in --mobile)
    mobile: false,         // emulate a phone (mobile viewport, touch, UA, high-DPI)
    long: false,           // --long: 16:9 long-form YouTube (5-7 min script, slow scroll locked to the audio)
    duration: 20.0,        // scroll seconds (auto-fit to narration when narrating)
    fps: 30,
    speed: null,           // px/s; overrides duration
    quality: 18,           // video quality (CRF-like 0-51; lower = better)
    bitrate: null,
    output: 'output/video.mp4',
    wait: 4.0,
    start_delay: 1.0,
    pause: 1.5,
    scroll_mode: 'smooth',
    pause_every: 0,
    pause_for: 1.0,
    browser: 'chromium',   // chromium | obscura
    mode: 'virtual',       // virtual (chromium: page-snapshot tiles w/ live fallback) | tiles | stepped | realtime
    capture: 'auto',
    warmup: false,
    headed: false,
    channel: null,
    timeout: 45,
    // --- narration & agent
    narrate: false,        // AI reads the page and writes the script
    text: null,            // or: speak this text
    enhance: false,        // let the AI rewrite --text into a hook script
    prompt: '',
    persona_file: 'persona.md',
    entertainment: false,  // --entertainment: films/TV/anime/comics story mode
    script_seconds: 60,
    script_only: false,
    scraper: 'obscura',    // obscura (falls back to chromium) | chromium
    obscura_path: null,
    private_net: false,
    llm_model: 'gemini-3.5-flash-lite',
    voice: 'Easton',
    tts_model: 'hf://gradium/phonon-31m-en',
    tts_quant: 'q8',       // phonon-tts wasm offers q8 | f32 (q4k requests are mapped to q8)
    tts_lang: 'en',
    whisper_model: 'tiny', // compat flag: phonon-2-web replaces whisper (value is ignored)
    // --- talking avatar (--avatar)
    avatar_file: null,
    avatar_corner: 'bottom-right',
    avatar_size: 26.0,
    // --- image slideshow (--image, long videos only)
    image: false,
    image_dir: null,
    image_limit: 0,
    slide_seconds: 6.0,
    image_sources: 'all',
    kenburns: true,
    // --- captions
    captions: true,
    cap_style: 'pop',      // pop | karaoke
    cap_color: '#FFD700',
    cap_font: 'Anton',
    cap_size: 100,
    cap_position: 'low',   // high | mid | low
    cap_words: 4,
    cap_chars: 0,
    cap_caps: true,
    separate: false,
    keep_files: false,
    ...overrides,
  };
}

export class Config {}
export function makeConfig(obj) { return Object.assign(new Config(), defaultConfig(), obj); }