# Reddit Mode (`--reddit`) — the b.mjs integration

Your b.mjs screenshot machine is now part of the factory. One command runs the whole loop:
**random subreddit → hot section → AI picks the post it likes → clean b.mjs-style screenshot
→ persona caption → Instagram feed post**, and with `--video` the same post becomes a
TTS-voiced 9:16 video for YouTube Shorts + Instagram Reels. Same dependencies as b.mjs
(Obscura over CDP via Playwright), no new npm packages.

---

## Commands

```powershell
# Screenshot + persona caption -> Instagram FEED image post (not a reel)
node main.js --reddit --publish instagram

# Your original command still works: --long/--short are aliases inside --reddit mode
node main.js --reddit --long --publish instagram        # full-post screenshot
node main.js --reddit --short --publish instagram       # snippet screenshot (truncated + fade)

# New canonical names (recommended, they don't collide with the video --long)
node main.js --reddit --full --publish instagram
node main.js --reddit --snippet --publish instagram

# TTS story video (summary + hot-take voice-over, captions) -> YT Shorts + IG Reels
node main.js --reddit --video --publish all

# Same video, static-screenshot look instead of the slow scroll
node main.js --reddit --video --video-style card --publish all

# Pin a subreddit instead of a random pick
node main.js --reddit --subreddit developersIndia --publish instagram

# Skip discovery entirely - screenshot this exact post
node main.js --reddit --post-url "https://www.reddit.com/r/developersIndia/comments/abc123/some_post/" --publish instagram

# Capture + caption only, publish later (prints the caption, saves the PNG)
node main.js --reddit
node main.js --reddit --script-only --video             # just the video voice-over script
```

Without `--publish` nothing is posted: the screenshot and caption land in `output/reddit/`.

## Flags

| Flag | Meaning |
| --- | --- |
| `--reddit` | reddit mode |
| `--video` | with `--reddit`: build the TTS video in addition to (well, instead of) the image post |
| `--video-style scroll\|card` | video look: `scroll` (default) pans slowly down the restyled post page; `card` shows the composed screenshot card on a blurred background |
| `--full` / `--snippet` | screenshot style. `snippet` (default) truncates long text at ~360px with a white fade (b.mjs `--short`); `full` expands the read-more and captures everything (b.mjs `--long`) |
| `--long` / `--short` | aliases of `--full`/`--snippet` **inside `--reddit` mode only** (b.mjs compatibility). They never trigger long-form video there |
| `--subreddits <file>` | subreddit list (default `sources_reddit.md`) |
| `--subreddit <name>` | pin one subreddit (`developersIndia`, `r/developersIndia` or full URL) |
| `--post-url <url>` | skip discovery, use this exact post |
| `--image-url <url>` | public URL of the screenshot for the IG image post (skips auto-hosting) |
| `--yt-privacy`, `--publish`, `--persona`, `--prompt`, `--llm-model`, `--voice`, `--script-seconds`, caption flags, `--avatar*`, `--keep-files`, `--script-only` | all work exactly as in the rest of the factory |

## What gets published where

| Command | Instagram | YouTube |
| --- | --- | --- |
| `--reddit --publish instagram` | feed image post (photo, not reel) | — |
| `--reddit --publish all` | feed image post (a note says `all` = IG for images) | — |
| `--reddit --video --publish all` | Reel | Short |
| `--reddit --video --publish youtube` | — | Short |

## Files

```
output/reddit/<slug>-<style>.png            the screenshot (always kept)
output/reddit/<slug>-<style>.caption.txt    IG caption (image mode)
output/reddit/<slug>-<style>.meta.json      post + caption metadata (image mode)
output/reddit/<slug>-<style>-card.png       composed 9:16 frame (--video-style card)
output/reddit/<slug>.mp4 / .meta.json       video + metadata (video mode, deleted after
                                            successful publish like every factory render,
                                            unless --keep-files)
output/reddit/debug.png                     only when the post element wasn't found
output/history.json                         the post URL is recorded so it is never re-posted
```

## How publishing works

**Instagram image post** (`publish.uploadInstagramImage`): the Graph API only accepts photos
from a **public image URL**, so the PNG is auto-hosted on **catbox.moe** (permanent public
link, no API key) with **tmpfiles.org** as fallback, then
`POST /{ig-user-id}/media {image_url, caption}` → status poll → `media_publish`.
Host it yourself with `--image-url` / `IG_IMAGE_URL` if you don't want a public copy of the
screenshot. IG photo rules: 320–1440 px, ≤ 8 MB, aspect ratio 4:5 to 1.91:1 — the feed post
is composed at **1080×1350 (4:5)** when you use `--video-style card` assets; the raw
screenshot works as-is for standard posts.

**Video** (`--video`): Gemini writes the summary + hot-take script in your persona voice →
phonon TTS → phonon-2 captions → 9:16 render (capped at the 89 s API reel limit) → a full
metadata package (YT title/description/tags/hashtags + IG hook/caption/hashtags) →
`uploadYoutubeShort` + `uploadInstagramReel` exactly like `--auto`.

## Requirements

- Same as before: `GEMINI_API_KEY`, plus `IG_ACCESS_TOKEN`/`IG_USER_ID` (needs the
  `instagram_content_publish` permission) for image posts, plus `client_secrets.json` +
  `token.json` for YouTube.
- Obscura is optional but recommended for Reddit: drop `obscura.exe` into `obscura/`.
  Without it, bundled Chromium is used automatically.
- No new npm dependencies.

## Notes & troubleshooting

| Symptom | Fix |
| --- | --- |
| `hot.json` fails / HTML listing blocked | Obscura usually solves it; also try `--subreddit <name>` on a different sub |
| catbox.moe unreachable | tmpfiles.org is the automatic fallback; or host yourself via `--image-url` |
| IG rejects the photo | check dimensions/aspect (4:5–1.91:1) and that the hosted URL is publicly reachable |
| IG auth error on the image post | the token needs `instagram_content_publish` (same token as reels) |
| "every hot post is already in history" | pick a pinned subreddit, add subs to `sources_reddit.md`, or clear `output/history.json` |
| scroll video films a broken page | keep Obscura in `obscura/` (auto-selected); try `--video-style card` which renders from the screenshot instead of the live page |
| long post cut off in the video | pass `--full` so the whole text is captured |
