# Deploying the Node.js Factory to GitHub Actions

Complete runbook: clean the folder → create the repo → push → add secrets → verify.
The workflow (`.github/workflows/factory.yml`) runs the full pipeline
**scout → script → render → publish (YouTube Shorts + Instagram Reels) → cleanup**
4 times a day (09:07, 13:07, 17:07, 23:07 IST) and commits `output/history.json`
after every run so published topics are never picked again.

---

## 1. Why your old Python workflow misbehaved (and what's different now)

| Old problem | Cause | Fix in the new setup |
| --- | --- | --- |
| Never ran at the scheduled time | `queue: max` is **not a valid key** under `concurrency:` — it can make the whole workflow file fail validation | Removed; only `group` + `cancel-in-progress` are used |
| Runs start late | GitHub schedules are **best-effort**: 5–30 min delays are normal, worse near the top of the hour | Cron set to `:37` past the hour; times documented in the yml |
| Workflow silently stops running later | GitHub **disables schedules after 60 days with no repo activity** | The bot commits `history.json` every run — those commits keep the repo active |
| Schedule changes don't take effect | Schedules only fire from the **default branch** (`main`) | Workflow lives on `main`; verify after the first push |
| YouTube auth would die on a headless runner | Installed-app OAuth opens a browser | `token.json` (with `refresh_token`) is restored from a secret; `publish.js` now **fails fast with a clear message** if CI ever loses it |

---

## 2. Clean the project folder before pushing

Run this from the project folder (Windows) and make sure these exist / don't exist:

**Delete (local junk — never push):**
- `token.json`, `client_secrets.json`, `.env`, `env.txt`, `insta env.txt` — secrets. They go in as **GitHub Secrets**, not files.
- `node_modules/`, `.cache/` — huge; regenerated on the runner.
- `output/*.mp4`, `output/*.meta.json`, `output/*.wav` — run artifacts (keep `output/history.json` if you have it — it carries your published-URL history over).
- `obscura/` — not needed on CI; scraping falls back to Chromium automatically.
- Any leftover old Python port files (`main.py`, `publish.py`, `requirements.txt`, `__pycache__/`, etc.) — avoids confusion about which pipeline runs.

**Must be present (the workflow depends on them):**
- `vendor/phonon-2-web/web/pkg-st/` (4 files, ~468 KB) — the prebuilt ASR wasm engine. **Without it, CI needs a Rust toolchain to rebuild.**
- `fonts/Anton-Regular.ttf`, `persona.md`, `sources.md`, `sources_entertainment.md`
- `package.json` **and** `package-lock.json` (npm ci needs the lockfile)
- `main.js`, `src/`, `scripts/`, `.github/workflows/factory.yml`

All of these are already handled by the updated `.gitignore`.

---

## 3. Create the repo

1. github.com → **New repository**.
2. Name it (e.g. `ai-content-factory`), **Public recommended** — see the cost note, then **Create** (no README/gitignore init; keep it empty).
3. **Cost reality check** — each run can use up to 90 runner-minutes (typically ~15–40 min once the model cache is warm):
   - **Public repo: $0, unlimited minutes.** Recommended.
   - Private repo: 2,000 free min/month, then paid. 4 runs/day ≈ 1,800–3,600+ min/month → will likely exceed the free quota.

Secrets are encrypted either way — a public repo does **not** expose them.

---

## 4. Push the code (PowerShell, from the project folder)

```powershell
cd "D:\latest"

# safety check: this MUST print nothing
git ls-files | Select-String -Pattern "token\.json|client_secrets|env\.txt|node_modules"

git init -b main
git add .
git commit -m "AI Content Factory - Node.js edition with GitHub Actions pipeline"
git remote add origin https://github.com/<YOUR_USER>/<YOUR_REPO>.git
git push -u origin main
```

Confirm on GitHub that `.github/workflows/factory.yml` exists on `main` (the Actions tab should already list the workflow).

---

## 5. Add the 5 secrets

Repo → **Settings → Secrets and variables → Actions → New repository secret**:

| Secret name | Value |
| --- | --- |
| `GEMINI_API_KEY` | Your Gemini API key |
| `IG_ACCESS_TOKEN` | Instagram Graph long-lived access token |
| `IG_USER_ID` | Your Instagram business user ID |
| `YT_CLIENT_SECRETS_B64` | Base64 of `client_secrets.json` |
| `YT_TOKEN_JSON_B64` | Base64 of `token.json` (must contain `refresh_token`) |

Make the base64 strings in PowerShell (copy → paste into the secret box):

```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("D:\latest\client_secrets.json")) | Set-Clipboard
[Convert]::ToBase64String([IO.File]::ReadAllBytes("D:\latest\token.json")) | Set-Clipboard
```

---

## 6. Repo settings

Nothing is mandatory — the workflow declares its own permissions. Two optional checks:

- **Settings → Actions → General → Workflow permissions**: leave on default. If the final history push ever fails with `403`, switch it to **Read and write permissions**.
- **Settings → Actions → General → Allow all actions**: default is fine.

---

## 7. First run (do this before trusting the schedule)

**Actions tab → "AI Content Factory" → Run workflow → Run.** Watch the job:

1. First run is the slow one: it downloads the TTS model (~65 MB), the ASR model (~164 MB) and Chromium. Later runs restore them from cache in seconds.
2. The **doctor** step prints a ✓/✗ environment summary — read it once.
3. Success = a Short + Reel published, and a bot commit `chore: update history.json [skip ci]` on `main`.

After that, the schedule takes over on its own.

---

## 8. Living with the schedule

- Runs start **around** 09:07 / 13:07 / 17:07 / 23:07 IST — ±30 min drift is GitHub's scheduler, not a bug.
- If a run is still going when the next slot hits, the new one **queues** (doesn't kill the running upload).
- If email says *"scheduled workflows were disabled due to inactivity"*: open the Actions tab and re-enable, then trigger one manual run (any push/commit also resets the 60-day clock).
- To change times: edit the cron — remember to subtract 5:30 from IST (e.g. 20:41 IST = `11 15 * * *`).

---

## 9. Maintenance calendar (things that DO expire)

| What | Lifetime | What to do |
| --- | --- | --- |
| Instagram access token | long-lived tokens still expire (~60 days) | Refresh it, update the `IG_ACCESS_TOKEN` secret |
| YouTube `refresh_token` | indefinite **only if the OAuth consent screen is in Production** | Google Cloud console → OAuth consent screen → publish to Production. In *Testing* mode Google kills refresh tokens after **7 days** — the #1 cause of "YouTube worked for a week then stopped" |
| Model cache | forever, auto-refreshed via `restore-keys` | Bump the `models-v1` cache key in the yml only if a cached model ever goes stale |

---

## 10. Troubleshooting

| Symptom | Fix |
| --- | --- |
| Schedule never fires | Workflow file must be on `main`; open the yml on GitHub and check for a red banner (validation error) |
| `npm ci` fails | `package-lock.json` missing or out of sync with `package.json` — commit both from the folder where `npm install` last ran |
| `Chromium is not installed` | The `npx playwright install --with-deps chromium` step failed — open its log |
| YouTube: "OAuth needs a one-time interactive browser sign-in" | Your `token.json` secret has no `refresh_token` — re-authorize locally, re-base64, update `YT_TOKEN_JSON_B64` |
| YouTube worked ~1 week then stopped | OAuth consent screen in *Testing* mode — publish it to Production (see §9) |
| Instagram `ProcessingFailedError` / FILE_NOT_FOUND | Already hardened (chunked resumable upload + fresh container per attempt + 89 s cap). If it recurs, check `IG_ACCESS_TOKEN` freshness first |
| Job turns red but video still published | Check the log — a failed upload on one platform records history and still commits it; the job color reflects the error. IG token expiry is the usual culprit |
