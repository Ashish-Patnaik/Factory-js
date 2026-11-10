#!/usr/bin/env node
// main.js - Scrolling website video & autonomous AI content factory (Node.js edition).
// 1:1 port of main.py: same CLI surface, same pipeline, no ffmpeg anywhere - rendering runs
// on mediabunny + WebCodecs inside a headless encoder page, TTS/ASR run on-device (phonon).
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { RATIOS, SPEEDS, makeConfig } from './src/config.js';
import { POS } from './src/captions.js';
import { buildAss, buildCaptionData } from './src/captions.js';
import { narration } from './src/speech.js';
import { Browser, ObscuraBrowser } from './src/browser.js';
import { maxScroll, plan, planLoop, warmUp } from './src/scroll.js';
import { die, probe, validateUrl, probeLine, ev } from './src/utils.js';
import { recordVirtual, recordStepped, recordRealtime, recordTiled } from './src/recorder.js';
import { EncoderSession } from './src/encoder/session.js';

// fileURLToPath (not .pathname): on Windows pathname is '/D:/Factory%20js' - percent-encoded
// with a leading slash, which silently breaks every path built from ROOT (fonts, avatar scan).
const ROOT = path.dirname(fileURLToPath(import.meta.url));
const FONTS = path.join(ROOT, 'fonts');
const MOBILE_WIDTH = 360; // CSS px width for phone emulation (360x640 @ 3x DPR = 1080x1920 native 9:16)

// ---------------------------------------------------------------------------
// CLI (mirrors main.py's argparse, flag for flag)
// ---------------------------------------------------------------------------
const EXAMPLES = `Examples:
  node main.js --auto                                        discover URL, render 9:16 mobile video + metadata
  node main.js --auto --long                                 autonomous long-form: 5-7 min 16:9 YouTube video
  node main.js --auto --long --image                         same, but the AI builds a narrated image slideshow
  node main.js --auto --long --image --avatar                slideshow + looping talking-character gif
  node main.js --auto --long --avatar                        same, with a looping talking-character gif in the corner
  node main.js --entertainment                             entertainment story (films/anime/TV/comics) -> Shorts & Reels
  node main.js --entertainment --long                      entertainment video essay -> 5-7 min YouTube long-form
  node main.js --auto --publish all                          autonomous end-to-end: scout -> render -> post YT & IG
  node main.js https://example.com demo --mobile --narrate   single URL in mobile view with AI voice-over
  node main.js https://example.com demo --long               long-form: 5-7 min script, 16:9 slow-scroll video
  node main.js https://example.com demo                      plain desktop scroll video`;

const HELP = `
Scrolling website video & autonomous AI content factory.

${EXAMPLES}

autonomous AI agent:
  --auto                    autonomous mode: surf sources.md, pick best trending URL, write script + metadata in persona.md tone, render --mobile 9:16 video
  --long                    long-form YouTube mode: 5-7 min narration script (default 360s, max 600), 16:9 1080p desktop render, slow scroll that runs or ping-pong loops until the audio ends (ignores --mobile)
  --entertainment           entertainment mode: surf sources_entertainment.md (film, TV, anime & comics hubs), pick the best story and write the script in persona.md tone - generates a story + script for short videos, and combines with --long for 5-7 min YouTube video essays
  --random                  pick 1 random hub from sources.md instead of surfing all of them (saves tokens & scraping time)
  --sources <file>          file with seed/hub URLs to surf (default: sources.md)
  --persona <file>          creator persona & style file (default: persona.md)
  --publish <choice>        automatically publish finished video to YouTube Shorts, Instagram Reels, or both (all) [none|youtube|instagram|all]
  --yt-privacy <choice>     YouTube Short privacy status when publishing (default: public) [public|unlisted|private]

scroller video:
  --mobile                  load and scroll the site in mobile view (phone viewport, touch, mobile UA, 9:16 default)
  --ratio <choice>          [9:16|16:9|1:1]
  --width <n>               
  --height <n>              
  --viewport <WxH>          browser viewport (default: output size, or 360x640 with --mobile)
  --duration <f>            scroll seconds (auto-fit to the voice-over when narrating) (default: 20)
  --fps <choice>            [30|60] (default: 30)
  --speed <v>               slow|normal|fast or custom px/s
  --quality <n>             video quality 0-51 (lower = better) (default: 18)
  --bitrate <v>             e.g. 8M (overrides --quality)
  --output <p>              output path (default output/<name>.mp4)
  --headless                run without a window (default)
  --headed                  show the browser window while recording
  --channel <choice>        use installed Chrome/Edge instead of bundled Chromium [chrome|msedge]
  --wait <f>                settle seconds after load (default: 1)
  --pause <f>               hold at bottom (s); default 1.5, or 0 when narrating
  --start-delay <f>         hold at top before scrolling/speech (s); default 1, or 0 when narrating
  --scroll-mode <choice>    [constant|smooth] (default: smooth)
  --pause-every <n>         hold every PX scrolled
  --pause-for <f>           hold length for --pause-every (s)
  --browser <choice>        browser that loads the page (default chromium). obscura = for sites that block headless Chrome [chromium|obscura]
  --obscura                 shortcut for --browser obscura
  --mode <choice>           virtual = page-snapshot tiles + live virtual-time fallback (chromium, default - heavy URLs render ~20x faster); tiles = snapshot only; stepped = exact scroll, static pages (obscura default); realtime = live capture [virtual|tiles|stepped|realtime]
  --capture <choice>        [auto|screencast|screenshot]
  --warmup                  pre-scroll once to load lazy images
  --timeout <n>             (default: 45)

talking avatar:
  --avatar                  overlay a small looping gif (your talking character) in the bottom-right corner; it loops until the render ends -- with --long / narration that is exactly when the speech ends, so the character 'talks' for the whole video and stops with it
  --avatar-file <path>      gif to use (default: auto-detect avatar*.gif, then any .gif, in the factory root folder)
  --avatar-corner <choice>  which corner to pin the gif to (default: bottom-right) [bottom-right|bottom-left|top-right|top-left]
  --avatar-size <f>         avatar box width as % of the video's short side (default 26 = ~280px on 1080p)

image slideshow (with --long):
  --image                   --long only: instead of scrolling the page, the AI extracts images that match the script content from multiple sources (Wikipedia/Commons, Openverse, Pexels/Unsplash/Pixabay stock, IMDb for movies/TV via Obscura), fits them all cleanly to 16:9 and renders a Ken Burns slideshow with crossfades that runs exactly until the audio ends
  --image-dir <path>        use your own images from this folder instead of AI web search
  --image-limit <n>         max images to download (default: auto = audio length / --slide-seconds + margin)
  --slide-seconds <f>       target seconds each image stays on screen (default 6)
  --image-sources <list>    comma list of sources: wiki,openverse,ddg,stock,imdb (default: all available; stock needs PEXELS_API_KEY/UNSPLASH_ACCESS_KEY/PIXABAY_API_KEY, imdb needs Obscura)
  --kenburns / --no-kenburns  slow zoom/pan motion on every slide (default on)

narration (omit all of these for a plain scroll video):
  --narrate                 AI visits the page, writes a hook script, speaks it
  --text <t>                speak this text instead of an AI script
  --text-file <p>           read the text to speak from a file
  --enhance                 AI rewrites --text/--text-file into an engaging hook script
  --prompt <t>              extra instructions for the AI (tone, focus, audience...)
  --script-seconds <n>      AI script length target (default 60; with --long: 360, up to 600)
  --script-only             print/save the script and stop (no audio/video)
  --scraper <choice>        [obscura|chromium] (default: obscura)
  --obscura-path <p>        path to obscura(.exe); also used by --browser obscura
  --allow-private-network   let Obscura read localhost/LAN URLs
  --llm-model <m>           (default: gemini-3.5-flash-lite)
  --voice <v>               (default: Easton)
  --tts-model <m>           (default: hf://gradium/phonon-31m-en)
  --tts-quant <q>           (default: q8; q4k maps to q8)
  --tts-lang <l>            (default: en)
  --whisper-model <m>       compat only: ASR is phonon-2-web (Parakeet); the value is ignored

captions:
  --captions / --asr / --no-asr   ASR + TikTok captions on/off (--no-asr = voice-over and scroller only, no transcription)
  --cap-style <choice>      [pop|karaoke] (default: pop)
  --cap-color <hex>         highlight colour (default: #FFD700)
  --cap-font <f>            (default: Anton)
  --cap-size <n>            font size for a 1080-px short side (scales with the video) (default: 100)
  --cap-position <choice>   [high|low|mid] (default: low)
  --cap-words <n>           max words per caption line (default: 4)
  --cap-chars <n>           extra max characters per line (0 = auto-fit to frame width)
  --no-caps                 don't UPPERCASE captions
  --separate                don't mux the voice-over: save a silent video plus output/<name>.wav
  --keep-files              also keep the script (.txt), voice-over (.wav) and work files
  --encoder-channel <c>     browser channel for the encoder page (default: auto = chrome|msedge|chromium)
`;

function parse() {
  const argv = process.argv.slice(2);
  const g = {
    sources: 'sources.md', persona: 'persona.md', publish: 'none', yt_privacy: 'public',
    duration: 20, fps: 30, quality: 18, wait: 1, pause: null, start_delay: null,
    scroll_mode: 'smooth', pause_every: 0, pause_for: 1, browser: 'chromium', mode: null,
    capture: 'auto', timeout: 45, avatar_corner: 'bottom-right', avatar_size: 26.0,
    image_limit: 0, slide_seconds: 6.0, image_sources: 'all', kenburns: true,
    script_seconds: 60, scraper: 'obscura', llm_model: 'gemini-3.5-flash-lite',
    voice: 'Easton', tts_model: 'hf://gradium/phonon-31m-en', tts_quant: 'q4k', tts_lang: 'en',
    whisper_model: 'tiny', captions: true, cap_style: 'pop', cap_color: '#FFD700',
    cap_font: 'Anton', cap_size: 100, cap_position: 'low', cap_words: 4, cap_chars: 0,
    no_caps: false, separate: false, keep_files: false,
  };
  let url = null, name = null;
  const choicesErr = (flag, v, allowed) =>
    die(`${flag} must be one of: ${allowed.join(', ')} (got: ${v})`);

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) die(`missing value for ${a}`);
      return argv[++i];
    };
    switch (a) {
      case '-h': case '--help': console.log(HELP); process.exit(0); break;
      // positional (the first non-flag is url, second is name; --url is a hidden alias)
      default: {
        if (a.startsWith('--')) {
          switch (a) {
            case '--url': url = next(); break;
            case '--auto': g.auto = true; break;
            case '--long': g.long = true; break;
            case '--entertainment': g.entertainment = true; break;
            case '--random': g.random = true; break;
            case '--sources': g.sources = next(); break;
            case '--persona': g.persona = next(); break;
            case '--publish': g.publish = next(); if (!['none', 'youtube', 'instagram', 'all'].includes(g.publish)) choicesErr(a, g.publish, ['none', 'youtube', 'instagram', 'all']); break;
            case '--yt-privacy': g.yt_privacy = next(); if (!['public', 'unlisted', 'private'].includes(g.yt_privacy)) choicesErr(a, g.yt_privacy, ['public', 'unlisted', 'private']); break;
            case '--mobile': g.mobile = true; break;
            case '--ratio': g.ratio = next(); if (!RATIOS[g.ratio]) choicesErr(a, g.ratio, Object.keys(RATIOS)); break;
            case '--width': g.width = parseInt(next(), 10); break;
            case '--height': g.height = parseInt(next(), 10); break;
            case '--viewport': g.viewport = next(); break;
            case '--duration': g.duration = parseFloat(next()); break;
            case '--fps': g.fps = parseInt(next(), 10); if (![30, 60].includes(g.fps)) choicesErr(a, g.fps, [30, 60]); break;
            case '--speed': g.speed = next(); break;
            case '--quality': g.quality = parseInt(next(), 10); break;
            case '--bitrate': g.bitrate = next(); break;
            case '--output': g.output = next(); break;
            case '--headless': g.headless = true; break;
            case '--headed': g.headed = true; break;
            case '--channel': g.channel = next(); if (!['chrome', 'msedge'].includes(g.channel)) choicesErr(a, g.channel, ['chrome', 'msedge']); break;
            case '--wait': g.wait = parseFloat(next()); break;
            case '--pause': g.pause = parseFloat(next()); break;
            case '--start-delay': g.start_delay = parseFloat(next()); break;
            case '--scroll-mode': g.scroll_mode = next(); if (!['constant', 'smooth'].includes(g.scroll_mode)) choicesErr(a, g.scroll_mode, ['constant', 'smooth']); break;
            case '--pause-every': g.pause_every = parseInt(next(), 10); break;
            case '--pause-for': g.pause_for = parseFloat(next()); break;
            case '--browser': g.browser = next(); if (!['chromium', 'obscura'].includes(g.browser)) choicesErr(a, g.browser, ['chromium', 'obscura']); break;
            case '--obscura': g.browser = 'obscura'; break;
            case '--mode': g.mode = next(); if (!['virtual', 'tiles', 'stepped', 'realtime'].includes(g.mode)) choicesErr(a, g.mode, ['virtual', 'tiles', 'stepped', 'realtime']); break;
            case '--capture': g.capture = next(); if (!['auto', 'screencast', 'screenshot'].includes(g.capture)) choicesErr(a, g.capture, ['auto', 'screencast', 'screenshot']); break;
            case '--warmup': g.warmup = true; break;
            case '--timeout': g.timeout = parseInt(next(), 10); break;
            case '--avatar': g.avatar = true; break;
            case '--avatar-file': g.avatar_file = next(); break;
            case '--avatar-corner': g.avatar_corner = next(); if (!['bottom-right', 'bottom-left', 'top-right', 'top-left'].includes(g.avatar_corner)) choicesErr(a, g.avatar_corner, ['bottom-right', 'bottom-left', 'top-right', 'top-left']); break;
            case '--avatar-size': g.avatar_size = parseFloat(next()); break;
            case '--image': g.image = true; break;
            case '--image-dir': g.image_dir = next(); break;
            case '--image-limit': g.image_limit = parseInt(next(), 10); break;
            case '--slide-seconds': g.slide_seconds = parseFloat(next()); break;
            case '--image-sources': g.image_sources = next(); break;
            case '--kenburns': g.kenburns = true; break;
            case '--no-kenburns': g.kenburns = false; break;
            case '--narrate': g.narrate = true; break;
            case '--text': g.text = next(); break;
            case '--text-file': g.text_file = next(); break;
            case '--enhance': g.enhance = true; break;
            case '--prompt': g.prompt = next(); break;
            case '--script-seconds': g.script_seconds = parseInt(next(), 10); break;
            case '--script-only': g.script_only = true; break;
            case '--scraper': g.scraper = next(); if (!['obscura', 'chromium'].includes(g.scraper)) choicesErr(a, g.scraper, ['obscura', 'chromium']); break;
            case '--obscura-path': g.obscura_path = next(); break;
            case '--allow-private-network': g.allow_private_network = true; break;
            case '--llm-model': g.llm_model = next(); break;
            case '--voice': g.voice = next(); break;
            case '--tts-model': g.tts_model = next(); break;
            case '--tts-quant': g.tts_quant = next(); break;
            case '--tts-lang': g.tts_lang = next(); break;
            case '--whisper-model': g.whisper_model = next(); break;
            case '--captions': case '--asr': g.captions = true; break;
            case '--no-captions': case '--no-asr': g.captions = false; break;
            case '--cap-style': g.cap_style = next(); if (!['pop', 'karaoke'].includes(g.cap_style)) choicesErr(a, g.cap_style, ['pop', 'karaoke']); break;
            case '--cap-color': g.cap_color = next(); break;
            case '--cap-font': g.cap_font = next(); break;
            case '--cap-size': g.cap_size = parseInt(next(), 10); break;
            case '--cap-position': g.cap_position = next(); if (!(g.cap_position in POS)) choicesErr(a, g.cap_position, Object.keys(POS)); break;
            case '--cap-words': g.cap_words = parseInt(next(), 10); break;
            case '--cap-chars': g.cap_chars = parseInt(next(), 10); break;
            case '--no-caps': g.no_caps = true; break;
            case '--separate': g.separate = true; break;
            case '--keep-files': g.keep_files = true; break;
            case '--encoder-channel': g.encoder_channel = next(); break;
            default: die(`unknown argument: ${a}\n${EXAMPLES}`);
          }
        } else if (url === null) url = a;
        else if (name === null) name = a;
        else die(`unexpected argument: ${a}\n${EXAMPLES}`);
      }
    }
  }

  if (g.entertainment) {
    if (g.sources === 'sources.md') g.sources = 'sources_entertainment.md'; // --sources not given
    if (url) g.narrate = !(g.text || g.text_file); // entertainment-style story for a direct URL too
    else g.auto = true; // no URL: crawl the entertainment hubs and pick the story like --auto
  }
  if (g.auto) {
    g.mobile = !g.long;
    g.narrate = true;
    g.warmup = true;
  }
  if (g.long) {
    if (g.mobile) console.log('note: --long renders a 16:9 desktop video; ignoring --mobile');
    g.mobile = false;
    if (!(g.text || g.text_file)) g.narrate = true; // --long generates the long script itself
    if (g.script_seconds === 60) g.script_seconds = 360; // untouched default -> 6 min
    g.warmup = true; // long pages must be fully loaded before a 5-7 min render
  }

  let text = g.text ?? null;
  if (g.text_file) {
    try { text = fs.readFileSync(g.text_file, 'utf8').trim(); } catch (e) { die(`cannot read --text-file: ${e.message}`); }
  }
  const narrating = Boolean(g.narrate || text || g.auto);
  if (g.image && !narrating) die('--image needs a voice-over to time the slideshow: add --narrate, use --auto, or pass --text/--text-file');
  if (g.enhance && !text) die('--enhance needs --text or --text-file (use --narrate to let the AI write from the page)');
  if (text !== null && !text.trim()) die('the text to speak is empty');
  const maxSecs = g.long ? 600 : 60;
  if (!(5 <= g.script_seconds && g.script_seconds <= maxSecs)) {
    die(`--script-seconds must be between 5 and ${maxSecs}` + (g.long ? ' (--long allows up to 600s = 10 min)' : ''));
  }
  if (!/^#?[0-9a-fA-F]{6}$/.test(g.cap_color)) die('--cap-color must be a hex colour like #FFD700');
  if (g.avatar_file && !g.avatar) g.avatar = true; // passing --avatar-file implies --avatar
  if (g.avatar && !(5 <= g.avatar_size && g.avatar_size <= 60)) die('--avatar-size must be 5-60 (% of the video\'s short side)');
  if (g.image) {
    if (!g.long) die('--image works with --long only: it is the long-form YouTube slideshow mode\n  (short videos keep the scrolling style)');
    if (g.slide_seconds < 2 || g.slide_seconds > 30) die('--slide-seconds must be 2-30');
    if (g.image_limit && !(2 <= g.image_limit && g.image_limit <= 200)) die('--image-limit must be 2-200');
  }

  if (g.ratio && (g.width || g.height)) die('use either --ratio or --width/--height, not both');
  let w, h;
  if (g.ratio) [w, h] = RATIOS[g.ratio];
  else if (g.width || g.height) {
    if (!(g.width && g.height)) die('--width and --height must be given together');
    [w, h] = [g.width, g.height];
  } else {
    [w, h] = RATIOS[g.long ? '16:9' : ((narrating || g.mobile) ? '9:16' : '16:9')];
  }
  if (w % 2 || h % 2 || w < 16 || h < 16) die('width/height must be even numbers >= 16 (H.264 yuv420p)');
  if ((g.mode === 'virtual' || g.mode === 'tiles') && g.browser === 'obscura') {
    die(`--mode ${g.mode} needs Chromium (Obscura has no virtual clock); use --mode stepped or realtime`);
  }
  g.mode = g.mode || (g.browser === 'obscura' ? 'stepped' : 'virtual');
  if (narrating && g.mode === 'realtime') {
    die('--mode realtime can\'t be used with narration (needs frame-exact timing); use --mode virtual or stepped');
  }
  if (g.start_delay === null || g.start_delay === undefined) g.start_delay = narrating ? 0 : 1;
  if (g.pause === null || g.pause === undefined) g.pause = narrating ? 0 : 1.5;
  if (g.duration <= 0 || Math.min(g.wait, g.pause, g.start_delay, g.pause_for) < 0) {
    die('duration must be > 0 and delays must be >= 0');
  }
  if (!(0 <= g.quality && g.quality <= 51)) die('--quality must be 0-51');
  let speed = null;
  if (g.speed) {
    speed = SPEEDS[g.speed];
    if (speed === undefined) {
      speed = parseFloat(g.speed);
      if (!(speed > 0)) die('--speed must be slow, normal, fast or a positive number (px/s)');
    }
  }
  let vp = null;
  if (g.viewport) {
    const parts = g.viewport.toLowerCase().split('x').map((x) => parseInt(x, 10));
    if (parts.length !== 2 || parts.some((x) => !(x > 0))) die('--viewport must look like 1280x720');
    vp = parts;
  }

  if (url) {
    url = validateUrl(url);
    name = (name || (new URL(url).hostname.replace(/^www\./, '')) || 'video').replace(/[^\w.-]+/g, '_').replace(/^_+|_+$/g, '');
  } else {
    name = name || 'auto_short';
  }
  const out = g.output || path.join('output', name + '.mp4');
  fs.mkdirSync(path.dirname(path.resolve(out)) || '.', { recursive: true });

  const cfg = makeConfig({
    url: url ?? '', width: w, height: h, viewport: vp, mobile: g.mobile, long: g.long,
    duration: g.duration, fps: g.fps, speed, quality: g.quality, bitrate: g.bitrate ?? null,
    output: out, wait: g.wait, start_delay: g.start_delay, pause: g.pause,
    scroll_mode: g.scroll_mode, pause_every: g.pause_every, pause_for: g.pause_for,
    browser: g.browser, mode: g.mode, capture: g.capture, warmup: g.warmup,
    headed: g.headed && !g.headless, channel: g.channel ?? null, timeout: g.timeout,
    narrate: g.narrate, text, enhance: g.enhance, prompt: g.prompt ?? '',
    persona_file: g.persona, entertainment: g.entertainment,
    script_seconds: g.script_seconds, script_only: g.script_only, scraper: g.scraper,
    obscura_path: g.obscura_path ?? null, private_net: g.allow_private_network,
    llm_model: g.llm_model, voice: g.voice, tts_model: g.tts_model,
    tts_quant: g.tts_quant, tts_lang: g.tts_lang, whisper_model: g.whisper_model,
    captions: g.captions, cap_style: g.cap_style,
    cap_color: '#' + g.cap_color.replace(/^#/, '').toUpperCase(), cap_font: g.cap_font,
    cap_size: g.cap_size, cap_position: g.cap_position, cap_words: g.cap_words,
    cap_chars: g.cap_chars, cap_caps: !g.no_caps, separate: g.separate, keep_files: g.keep_files,
    avatar_file: g.avatar ? findAvatar(g.avatar_file) : null,
    avatar_corner: g.avatar_corner, avatar_size: g.avatar_size,
    image: g.image, image_dir: g.image_dir ?? null, image_limit: g.image_limit,
    slide_seconds: g.slide_seconds, image_sources: g.image_sources, kenburns: g.kenburns,
    encoder_channel: g.encoder_channel ?? null,
  });
  return { cfg, g };
}

function findAvatar(explicit) {
  // Locate the talking-character gif: an explicit --avatar-file path wins, otherwise
  // auto-detect avatar*.gif (then any .gif) in the factory root folder, then in the cwd.
  if (explicit) {
    const p = path.resolve(explicit);
    if (!fs.existsSync(p)) die(`--avatar-file not found: ${p}`);
    return p;
  }
  for (const root of [ROOT, process.cwd()]) {
    let gifs = [];
    try { gifs = fs.readdirSync(root).filter((n) => n.toLowerCase().endsWith('.gif')).sort(); } catch { continue; }
    const exact = gifs.find((n) => n.toLowerCase() === 'avatar.gif');
    if (exact) return path.join(root, exact);
    const anyAvatar = gifs.find((n) => n.toLowerCase().startsWith('avatar'));
    if (anyAvatar) return path.join(root, anyAvatar);
    if (gifs.length) return path.join(root, gifs[0]);
  }
  die('--avatar could not find a .gif in the factory root folder (' + ROOT + '). ' +
    'Drop your talking-character gif there (e.g. avatar.gif) or pass --avatar-file <path>');
}

function defaultViewport(cfg) {
  if (cfg.viewport) return cfg.viewport;
  if (cfg.mobile) {
    const vw = MOBILE_WIDTH;
    const vh = Math.round(vw * cfg.height / cfg.width);
    return [vw, vh];
  }
  return [cfg.width, cfg.height];
}

// ---------------------------------------------------------------------------
// --image slideshow pipeline
// ---------------------------------------------------------------------------
async function runImageSlideshow(cfg, work, wav, assData, secs, script) {
  const images = await import('./src/images.js');
  const slides = await import('./src/slides.js');
  const { EncoderHost } = await import('./src/encoder/host.js');
  const need = cfg.image_limit || Math.min(100, Math.ceil(secs / Math.max(cfg.slide_seconds, 2.0)) + 8);
  const idir = path.join(work, 'images');
  const imgs = cfg.image_dir
    ? await images.localImages(cfg.image_dir, idir, need)
    : await images.collect(cfg, script || cfg.text || '', idir, need, cfg.image_sources);
  if (imgs.length < 2) {
    die('--image found fewer than 2 usable images (network down or every source empty); ' +
      'try again, widen --image-sources, or pass --image-dir with your own pictures');
  }
  console.log(`[image] building the slideshow from ${imgs.length} image(s)`);
  const host = await new EncoderHost().launch({ channel: cfg.encoder_channel ?? null });
  try {
    await host.initEncoder({ width: cfg.width, height: cfg.height, fps: cfg.fps, quality: cfg.quality, bitrate: cfg.bitrate ?? null });
    await host.setFontFile?.(null);
    const fontPath = path.join(FONTS, 'Anton-Regular.ttf');
    if (fs.existsSync(fontPath)) await host.setFont(fs.readFileSync(fontPath));
    if (assData) await host.setCaptions(assData);
    if (cfg.avatar_file) {
      console.log(`[avatar] looping ${path.basename(cfg.avatar_file)} in the ${cfg.avatar_corner} corner over the slideshow`);
      await host.setAvatar(fs.readFileSync(cfg.avatar_file), cfg.avatar_corner, cfg.avatar_size);
    }
    if (wav) {
      const { readWav } = await import('./src/tts.js');
      const { pcm, sampleRate } = fs.existsSync(wav) ? readWav(wav) : { pcm: new Float32Array(1), sampleRate: 24000 };
      await host.deliverAudio(pcm, { sampleRate, audioDelay: cfg.start_delay, totalDuration: secs + cfg.start_delay + cfg.pause });
    }
    const info = await slides.renderSlideshow(host, cfg, imgs, { assData, totalS: secs + cfg.start_delay + cfg.pause, work });
    await host.rpc('finish'); // finalize output2 + attach audio
    await host.pullToFile(cfg.output);
    console.log(`done: ${info.frames} frames in ${(info.ms / 1000).toFixed(0)}s`);
  } finally {
    await host.close();
  }
  await probe(cfg.output);
}

// ---------------------------------------------------------------------------
// Scroll-video pipeline
// ---------------------------------------------------------------------------
async function run(cfg, work) {
  let wav = null;
  let assData = null;
  let assFile = null;
  let secs = 0.0;
  let script = null;
  if (cfg.narrate || cfg.text) {
    wav = path.join(work, 'speech.wav');
    const res = await narration(cfg, wav).catch((e) => die(String(e?.message ?? e)));
    ({ script, secs } = res);
    const words = res.words;
    const base = cfg.output.replace(/\.mp4$/i, '');
    if (cfg.script_only || cfg.keep_files) {
      fs.writeFileSync(base + '.txt', script + '\n');
      console.log(`[out] script saved to ${base}.txt`);
    }
    if (cfg.script_only) return;
    if (cfg.separate || cfg.keep_files) {
      fs.copyFileSync(wav, base + '.wav');
      console.log(`[out] voice-over saved to ${base}.wav`);
    }
    if (!cfg.speed) cfg.duration = Math.max(secs - 1 / cfg.fps, 0.1);
    if (!cfg.long) {
      // Hard-cap the final video at the Instagram API reel limit (90 s; the in-app 3-min
      // limit does NOT apply to API publishing). Nothing upstream bounded the voice-over:
      // a long AI script -> a >90 s video -> Meta's transcoder kills it after a full
      // 20+ MB upload with ProcessingFailedError. REEL_MAX_SECONDS leaves a 1 s margin.
      // (--long is exempt: it is the 5-7 min YouTube long-form mode.)
      const reelMax = Number(process.env.REEL_MAX_SECONDS ?? '89');
      if (secs + cfg.start_delay + cfg.pause > reelMax) {
        console.log(`note: voice-over is ${secs.toFixed(1)}s; cutting video+audio at the ${reelMax.toFixed(0)}s reel cap ` +
          '(Instagram API rejects reels longer than 90s)');
        secs = Math.max(reelMax - cfg.start_delay - cfg.pause, 1.0);
      }
    }
    if (cfg.captions) {
      assData = buildCaptionData(words, cfg, cfg.width, cfg.height, cfg.start_delay);
      if (cfg.keep_files) { // keep the classic .ass too, for users who burn with other tools
        assFile = path.join(work, 'captions.ass');
        fs.writeFileSync(assFile, buildAss(words, cfg, cfg.width, cfg.height, cfg.start_delay));
      }
    }
  }
  if (cfg.image) {
    // --image: no browser, no scrolling - the narration drives an AI-sourced image slideshow
    return runImageSlideshow(cfg, work, wav, assData, secs, script);
  }
  const [vw, vh] = defaultViewport(cfg);
  if (cfg.browser === 'obscura') console.log('[browser] Obscura: exact stepped scroll capture');
  if (cfg.mobile) {
    const dpr = Math.max(1, Math.ceil(Math.max(cfg.width / vw, cfg.height / vh)));
    console.log(`[mobile] phone viewport ${vw}x${vh} CSS px @ ${dpr}x DPR -> ${vw * dpr}x${vh * dpr} native capture`);
  }

  const encoder = await EncoderSession.create(cfg);
  let st = null;
  const Br = cfg.browser === 'obscura' ? ObscuraBrowser : Browser;
  let br = null;
  try {
    if (cfg.avatar_file) {
      console.log(`[avatar] looping ${path.basename(cfg.avatar_file)} in the ${cfg.avatar_corner} corner ` +
        `(${cfg.avatar_size.toFixed(0)}% of the short side) until the render ends`);
      await encoder.setAvatarFile(cfg.avatar_file, cfg.avatar_corner, cfg.avatar_size);
    }
    const fontPath = path.join(FONTS, 'Anton-Regular.ttf');
    if (assData && fs.existsSync(fontPath)) await encoder.setFontFile(fontPath);
    if (assData) await encoder.setCaptionsData(assData);
    if (cfg.separate) { /* voice-over stays a separate .wav: render with silent audio */ }
    await encoder.setAudio(cfg.separate ? null : wav, { audioDelay: cfg.start_delay, totalDuration: 0 });
    await encoder.begin();

    br = await new Br(cfg).open();
    const page = await br.newPage(vw, vh);
    await br.load(page);
    const [aw, ah] = await ev(page, '() => [window.innerWidth, window.innerHeight]');
    if (aw !== vw || ah !== vh) {
      console.log(`warning: viewport is ${aw}x${ah}, requested ${vw}x${vh}; the encoder will fit it without stretching`);
    }
    if (cfg.warmup) await warmUp(page, Math.max(ah - 100, 100));
    const maxY = await maxScroll(page);
    const nOut = wav ? Math.ceil((cfg.start_delay + secs + cfg.pause) * cfg.fps) : null;
    let ys;
    if (cfg.long) {
      // --long: the video length is locked to the narration; the scroll runs (or ping-pong
      // loops) until the audio ends. Auto-pick a pleasant slow speed when none was given.
      if (!cfg.speed) {
        const target = wav ? secs : cfg.duration;
        const ideal = maxY / Math.max(target, 0.1); // px/s that covers the page exactly once over the audio
        if (ideal > SPEEDS.slow) {
          cfg.speed = SPEEDS.slow;
          console.log(`[long] page is ${maxY}px: capping the scroll at the slow speed (${SPEEDS.slow} px/s), looping until the audio ends`);
        } else if (ideal < 12) {
          cfg.speed = 45.0;
          console.log(`[long] short page (${maxY}px): scrolling at a gentle 45 px/s, ping-pong looping until the audio ends`);
        } else {
          console.log(`[long] one slow pass: ~${ideal.toFixed(1)} px/s spread across the full ${target.toFixed(0)}s video`);
        }
      }
      ys = planLoop(cfg, maxY, nOut || Math.round((cfg.start_delay + cfg.duration + cfg.pause) * cfg.fps));
    } else {
      ys = plan(cfg, maxY);
      if (wav) {
        if (ys.length > nOut) console.log(`note: scroll is slower than the voice-over; video ends with the audio (${secs.toFixed(1)}s)`);
        else if (ys.length < nOut) console.log(`note: scroll ends ${((nOut - ys.length) / cfg.fps).toFixed(1)}s before the audio; playing in reverse to fill`);
      }
    }
    console.log(`page scroll range ${maxY}px, ${nOut ?? ys.length} frames (${((nOut ?? ys.length) / cfg.fps).toFixed(1)}s) at ${cfg.fps}fps`);
    const rec = { page, ys, encoder, cfg, cache: work };
    const wantTiles = cfg.mode === 'tiles' ||
      (cfg.mode === 'virtual' && String(process.env.RENDER_TILED ?? '1') !== '0');
    if (wantTiles) {
      // Snapshot-tile capture: one page pass + local composition, immune to slow/heavy pages
      // (the "--entertainment renders for 40 minutes" bug). Failures before the first encoded
      // frame fall back to the live virtual-time capture; mid-render failures are fatal.
      try {
        st = await recordTiled({ ...rec, nOut });
      } catch (e) {
        if (e?.tilePhase === 'render') throw e;
        console.log(`note: tile snapshot capture unavailable (${String(e?.message ?? e).split('\n')[0]}); using live virtual-time capture`);
        st = await recordVirtual({ ...rec, nOut });
      }
    } else if (cfg.mode === 'virtual') st = await recordVirtual({ ...rec, nOut });
    else if (cfg.mode === 'stepped') st = await recordStepped({ ...rec, nOut });
    else st = await recordRealtime(rec);
  } catch (e) {
    await encoder.abort();
    die(String(e?.message ?? e).split('\n')[0] || 'browser error (crash or timeout)');
  } finally {
    if (br) await br.close();
  }

  try {
    await encoder.finish(cfg.output);
  } catch (e) {
    die(`encode failed: ${String(e?.message ?? e).split('\n')[0]}`);
  }
  if (st.backend === 'tiled') {
    const rev = st.reversed ? ` (${st.reversed} played in reverse)` : '';
    console.log(`done: ${st.frames} frames from ${st.tiles} page tile(s) in ${st.renderS.toFixed(0)}s${rev}`);
  } else if (st.backend === 'virtual-time' || st.backend === 'stepped') {
    const rev = st.reversed ? ` (${st.reversed} played in reverse)` : '';
    console.log(`done: ${st.frames} frames${rev} in ${st.renderS.toFixed(0)}s`);
  } else {
    console.log(`done: ${st.backend} capture, ${(st.captured / st.seconds).toFixed(1)} distinct frames/s`);
  }
  await probe(cfg.output);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  const { cfg, g } = parse();
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'wvg_'));
  let pkg = null;
  let metaPath = null;
  try {
    if (g.auto || g.publish !== 'none') {
      const agent = await import('./src/agent.js');
      const { pageText } = await import('./src/scrape.js');
      let content;
      if (!cfg.url) {
        const found = await agent.discoverBestUrl(cfg, g.sources, g.random);
        cfg.url = found.url;
        content = found.content;
      } else {
        content = await pageText(cfg, cfg.url);
      }
      console.log(`[agent] generating persona script & social metadata for ${cfg.url}`);
      pkg = await agent.generateContentPackage(cfg, cfg.url, content);
      cfg.text = pkg.script;
      cfg.narrate = false;
      if (!g.name && !g.output && pkg.slug) {
        const slug = pkg.slug.replace(/[^\w.-]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 50);
        cfg.output = path.join('output', `${slug}.mp4`);
      }
      metaPath = cfg.output.replace(/\.mp4$/i, '') + '.meta.json';
      fs.writeFileSync(metaPath, JSON.stringify(pkg, null, 2));
      console.log(`[agent] metadata saved to ${metaPath}`);
      console.log(`  YT Title : ${pkg.youtube.title}`);
      if (pkg.instagram) console.log(`  IG Hook  : ${pkg.instagram.title}`);
    }

    await run(cfg, work);

    if (pkg && !cfg.script_only) {
      const agent = await import('./src/agent.js');
      const publish = await import('./src/publish.js');
      const errors = [];
      let publishedAny = false;

      if (g.publish === 'youtube' || g.publish === 'all') {
        try {
          const yt = pkg.youtube;
          await publish.uploadYoutubeShort(cfg.output, yt.title, yt.description, yt.hashtags ?? [], g.yt_privacy,
            yt.tags ?? null, yt.category ?? null);
          publishedAny = true;
        } catch (e) {
          console.log(`[youtube] ERROR: ${e.message}`);
          errors.push(`YouTube: ${e.message}`);
        }
      }

      if (g.publish === 'instagram' || g.publish === 'all') {
        if (g.long || !pkg.instagram) {
          // --long is YouTube long-form: the agent writes long-form YouTube metadata INSTEAD of an IG
          // caption, and the Instagram Graph API hard-caps reels at 90s anyway, so uploading there
          // would only ever post a truncated video. Long-form reach lives on YouTube.
          console.log('note: Instagram publishing skipped (--long is YouTube long-form; reels are capped at 90s)');
        } else {
          try {
            const ig = pkg.instagram;
            await publish.uploadInstagramReel(cfg.output, ig.title, ig.description, ig.hashtags, true, null);
            publishedAny = true;
          } catch (e) {
            console.log(`[instagram] ERROR: ${e.message}`);
            errors.push(`Instagram: ${e.message}`);
          }
        }
      }

      // Save URL to history as soon as it rendered (or published to at least 1 platform) so we don't duplicate YT Shorts
      if (publishedAny || g.publish === 'none') agent.saveHistory(cfg.url);

      // Clean up the MP4 and metadata JSON to save storage (only when something actually published)
      if (g.publish !== 'none' && publishedAny && !cfg.keep_files) {
        for (const fpath of [cfg.output, metaPath]) {
          if (fpath && fs.existsSync(fpath)) {
            fs.rmSync(fpath);
            console.log(`[cleanup] removed ${fpath}`);
          }
        }
      }

      if (errors.length) die(errors.join(' | '));
    }
  } catch (e) {
    die(`pipeline aborted due to error: ${String(e?.message ?? e).split('\n')[0]}`);
  } finally {
    if (cfg.keep_files) console.log(`[keep] intermediate files in ${work}`);
    else fs.rmSync(work, { recursive: true, force: true });
  }
}

main();