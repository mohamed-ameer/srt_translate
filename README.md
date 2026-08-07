# Subtitle Toolkit: Extract → Translate

Two small command-line tools that work together to take a movie with subtitles **burned into the video picture** (no separate subtitle file available) and turn them into a clean, translated `.srt` file you can load into any video player.

```
 [video file]  →  extract_hardcoded_subtitles.py  →  [subtitles.srt in the original language]
                                                              │
                                                              ▼
                                            srt_translate.py (AI translation)
                                                              │
                                                              ▼
                                          [subtitles.srt in your language]
```

- **`extract_hardcoded_subtitles.py`** — reads a video, takes a screenshot every fraction of a second, and uses OCR (Optical Character Recognition — software that reads text out of images) to turn the subtitle text burned into the picture into a normal `.srt` file.
- **`srt_translate.py`** — takes any `.srt` file and translates it into a different language using an AI model (Google Gemini or OpenAI GPT), producing natural, colloquial subtitles rather than a stiff word-for-word translation.

You only need `extract_hardcoded_subtitles.py` if you don't already have a subtitle file for your video — if you can download one (e.g. from [downsub.com](https://downsub.com/)), skip straight to `srt_translate.py`.

---

## Contents

1. [One-time setup](#1-one-time-setup)
2. [Translating subtitles (`srt_translate.py`)](#2-translating-subtitles-srt_translatepy)
3. [Extracting burned-in subtitles from a video (`extract_hardcoded_subtitles.py`)](#3-extracting-burned-in-subtitles-from-a-video-extract_hardcoded_subtitlespy)
4. [Downloading a video from YouTube (optional)](#4-downloading-a-video-from-youtube-optional)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. One-time setup

### 1.1 Python packages

```bash
pip install google-generativeai openai opencv-python pytesseract tqdm numpy
```

You don't need both `google-generativeai` and `openai` unless you plan to use both AI providers — but installing both is harmless and saves you from a mid-project surprise.

> **Using a virtual environment (venv)?** Make sure it's *activated* every time before running these scripts, and that you installed the packages *into that same environment*. The single most common error people hit here is "No module named 'google'" — nine times out of ten it means the terminal running the script and the terminal that ran `pip install` were using two different Pythons. Run `python -c "import sys; print(sys.executable)"` to check which Python is actually active.

### 1.2 An AI API key (for translation)

You need a free or paid API key from whichever provider you want to translate with:

- **Google Gemini** (recommended, has a free tier): [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys)
- **OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

Once you have one, make it available to the script as an environment variable:

```bash
export GEMINI_API_KEY=your_key_here
# or
export OPENAI_API_KEY=your_key_here
```

(If you skip this, the interactive wizard — see below — will just ask you to paste it in when needed.)

### 1.3 Tesseract OCR + ffmpeg (only needed for extracting burned-in subtitles)

```bash
brew install tesseract ffmpeg        # macOS
# or: sudo apt install tesseract-ocr ffmpeg   # Ubuntu/Debian
```

Homebrew's Tesseract only ships **English** by default. If your video's subtitles are in another language (e.g. Arabic), you need to download that language's data file too. For Arabic:

```bash
curl -fL -o "$(brew --prefix tesseract)/share/tessdata/ara.traineddata" \
  https://github.com/tesseract-ocr/tessdata_best/raw/main/ara.traineddata
```

(Swap `ara` for the [3-letter Tesseract code](https://github.com/tesseract-ocr/tessdata_best) of whatever language you need — e.g. `fra` for French, `spa` for Spanish.)

Confirm everything is in place:

```bash
tesseract --list-langs      # should list your language, e.g. "ara"
ffmpeg -version              # should print a version, not "command not found"
```

---

## 2. Translating subtitles (`srt_translate.py`)

This is the main tool. Give it a `.srt` file in one language, and it hands the text to an AI model (in batches, never touching the timestamps/numbering) with instructions to produce a **natural, everyday-spoken** translation — adapting jokes and idioms, keeping swearing at the same intensity, and translating religious expressions instead of leaving them untouched — rather than a stiff literal translation.

### 2.1 The easy way: interactive mode

If you're not sure what to type, just run it with no arguments and answer the questions it asks:

```bash
python srt_translate.py
```

It will walk you through, one question at a time, with a sensible default shown in `[brackets]` — press **Enter** to accept the default, or type something else to override it:

```
=== Interactive setup — press Enter to accept the default shown in [brackets] ===

Path to source .srt file: input_ar.srt

-- Source (what the movie/dialogue actually is) --
Source language (e.g. Arabic, Spanish, French, Hindi) [Arabic]:
Country/region the movie is from — sets the dialect (e.g. Egypt, Lebanon, Mexico, Argentina).
Leave blank for a generic/standard dialect [Egypt]:

-- Target (what you want the subtitles translated into) --
Target language to translate into [Indonesian]:
Country/region for the target dialect/accent (e.g. Indonesia, Spain, Argentina).
Leave blank for a generic/standard register [blank]: Indonesia

-- Translation engine --
Provider (gemini/openai) [gemini]:
Model [gemini-flash-latest]:

-- Output & batching --
Output .srt path [blank]:
Subtitle lines per API call (batch size) [100]:
```

You can also force this wizard to run even after typing some flags, in case you want to double-check or tweak something before it starts:

```bash
python srt_translate.py input_ar.srt --interactive
```

### 2.2 The fast way: command-line flags

Once you know what you want, flags are quicker for repeat use or scripting:

```bash
python srt_translate.py input_ar.srt -o output_id.srt --lang Indonesian
```

That's really all you *need* — everything else has a sensible default. A more fully-specified example:

```bash
python srt_translate.py input_ar.srt \
  --source-language Arabic --source-country Egypt \
  --lang Indonesian --target-country Indonesia \
  --provider gemini --model gemini-flash-latest \
  -o output_id.srt
```

### 2.3 Why "country" matters, not just "language"

A language spoken casually varies enormously by country — "Arabic" alone is ambiguous between Egyptian, Lebanese, Gulf, or Moroccan slang, which can be as different from each other as separate languages in everyday speech. Telling the AI **which country the movie is from** (for the source) and **which country's dialect you want** (for the target) lets it use the actual slang, expressions, and tone people really use there, instead of a generic textbook version nobody talks like.

| Flag | Meaning | Default |
|---|---|---|
| `--source-language` | The language the subtitles are currently in | `Arabic` |
| `--source-country` | Which country's dialect that language is in | `Egypt` |
| `--lang` | The language you want translated into | `Indonesian` |
| `--target-country` | Which country's dialect to use for the translation (optional) | *(none — neutral/standard)* |

If you need full manual control over how the source language is described to the AI (e.g. a dialect that isn't well captured by "language + country"), use `--source-lang "your own description"` instead — it overrides `--source-language`/`--source-country` entirely.

### 2.4 Choosing a provider and model

```bash
python srt_translate.py --provider gemini --list-models
```

This prints every model your API key is actually allowed to use — handy since providers periodically retire model names, and a model that worked last month can start returning "not found" errors. Pass whichever name it lists via `--model`.

| Flag | Meaning | Default |
|---|---|---|
| `--provider` | `gemini` or `openai` | `gemini` |
| `--model` | Exact model name | `gemini-flash-latest` (Gemini) / `gpt-4o` (OpenAI) |
| `--api-key` | API key, if you'd rather not use an environment variable | *(reads `GEMINI_API_KEY`/`OPENAI_API_KEY`)* |

### 2.5 If it gets interrupted

Every batch of translated lines is saved to a `<output>.progress.json` cache file as it goes. If the script crashes, your internet drops, or you hit a rate limit and give up for the day — just run the **exact same command again**. It picks up where it left off instead of re-translating (and re-paying for) lines it already finished. Use `--fresh` to ignore the cache and start over from scratch.

### 2.6 If you're hitting rate-limit / quota errors (HTTP 429)

Free API tiers usually cap how many requests you can make per minute. If you see a `429` / "quota exceeded" error:

```bash
python srt_translate.py input_ar.srt --rpm 10
```

`--rpm 10` throttles the script to at most 10 requests per minute so it never trips the limit in the first place. If it still fails immediately, that's a **daily** cap rather than a per-minute one — check your plan at [ai.dev/rate-limit](https://ai.dev/rate-limit); no amount of client-side throttling fixes a daily quota, only time or a paid plan does.

### 2.7 Full option reference

| Flag | Meaning | Default |
|---|---|---|
| `input_srt` | Path to the source `.srt` file (omit this to launch the interactive wizard) | — |
| `-i`, `--interactive` | Force the interactive wizard even if flags are set | off |
| `-o`, `--output` | Output `.srt` path | `<input>_<lang>.srt` |
| `--lang` | Target language | `Indonesian` |
| `--target-country` | Target dialect's country/region | *(none)* |
| `--source-language` | Source language | `Arabic` |
| `--source-country` | Source dialect's country/region | `Egypt` |
| `--source-lang` | Raw override for the source description (bypasses the two above) | *(none)* |
| `--provider` | `gemini` or `openai` | `gemini` |
| `--model` | Model name | see [2.4](#24-choosing-a-provider-and-model) |
| `--api-key` | API key | reads from environment |
| `--batch-size` | Subtitle lines sent per API call | `100` |
| `--max-retries` | Retry attempts per API call before giving up | `3` |
| `--rpm` | Max API requests per minute (throttle) | `0` (unlimited) |
| `--fresh` | Ignore cached progress, start over | off |
| `--list-models` | List models your key can use, then exit | — |

---

## 3. Extracting burned-in subtitles from a video (`extract_hardcoded_subtitles.py`)

Use this **only** if you don't have a subtitle file at all and the subtitles are permanently baked into the video picture. It works by cropping a strip near the bottom of each sampled frame and running OCR on it — it is not perfect and works best when the subtitles have good contrast against the background (e.g. white text with a dark outline).

### 3.1 Basic usage

```bash
python extract_hardcoded_subtitles.py movie.mp4
```

This creates `movie.srt` next to it. To pick where the subtitle text sits on screen or how often to sample:

```bash
python extract_hardcoded_subtitles.py movie.mp4 \
  -o subtitles.srt \
  --lang ara \
  --top 0.80 --bottom 0.95 \
  --sample 0.5
```

### 3.2 Tuning tips

- **`--top` / `--bottom`** — the vertical crop window, as a fraction of the video's height (`0.0` = top of frame, `1.0` = bottom). Default `0.72`–`0.98` covers the bottom ~28% of the frame, which fits most subtitle placements. If subtitles are being missed or garbage text is being picked up, watch a few seconds of the video and estimate roughly where in that vertical range the subtitle text actually sits.
- **`--sample`** — how often (in seconds) to grab a frame and OCR it. Smaller = more accurate timing but much slower to run. `0.5` is a reasonable default; try `1.0` for a faster first pass.
- **`--min-confidence`** — how sure Tesseract needs to be about a word before keeping it (0–100). Raise it if you're getting noisy/garbled text; lower it if real subtitle words are being dropped.
- **`--lang`** — the Tesseract language code for OCR (not the movie's spoken language necessarily — the *written script* of the subtitles). `ara` for Arabic, `eng` for English, etc.

### 3.3 If the video won't open

Some `opencv-python` builds are compiled **without** any video-decoding support at all (you can check with `python -c "import cv2; print(cv2.getBuildInformation())" | grep FFMPEG` — if it says `NO`, that's this). If that's the case, this script automatically falls back to decoding frames via the `ffmpeg` command-line tool instead, as long as it's installed (see [1.3](#13-tesseract-ocr--ffmpeg-only-needed-for-extracting-burned-in-subtitles)). You can force one backend or the other with `--backend cv2` / `--backend ffmpeg` if you need to debug which one is actually working.

### 3.4 Full option reference

| Flag | Meaning | Default |
|---|---|---|
| `video` | Input video file | — |
| `-o`, `--output` | Output `.srt` path | `<video_name>.srt` |
| `--lang` | Tesseract OCR language code | `ara` |
| `--sample` | Seconds between sampled frames | `0.50` |
| `--top` | Crop top, as a fraction of frame height | `0.72` |
| `--bottom` | Crop bottom, as a fraction of frame height | `0.98` |
| `--min-confidence` | Minimum OCR confidence to keep a word (0–100) | `35` |
| `--min-change` | How different two frames must look to count as a subtitle change | `0.18` |
| `--backend` | Video decoding backend: `auto`, `cv2`, or `ffmpeg` | `auto` |

---

## 4. Downloading a video from YouTube (optional)

If you're starting from a YouTube link rather than a local file:

```bash
python3 -m pip install -U yt-dlp
python3 -m yt_dlp \
  -f "bv*+ba/b" \
  --merge-output-format mp4 \
  -o "%(title)s.%(ext)s" \
  "youtube_video_link"
```

---

## 5. Troubleshooting

| Symptom | What it means | Fix |
|---|---|---|
| `No module named 'google'` (or `'openai'`) | The Python actually running the script doesn't have the package installed — usually because `pip install` and `python script.py` were run in two different Python environments | Run `python -c "import sys; print(sys.executable)"` to see which Python is active, then `<that python> -m pip install google-generativeai` (or `openai`). If using a venv, activate it first. |
| `404 ... model ... no longer available` | The AI provider retired that model name | Run `python srt_translate.py --list-models` and pass one of the listed names via `--model` |
| `429 ... quota ... billing` | You've hit the API's rate limit or usage cap | Add `--rpm 10` to slow requests down; if it fails immediately with no retry, it's a daily cap — check your plan/billing at the provider's dashboard |
| `OpenCV: Couldn't read movie file` / `ERROR: Could not open video` | Your `opencv-python` build has no video decoder at all (check with the command in [3.3](#33-if-the-video-wont-open)) | Install `ffmpeg` (`brew install ffmpeg`); the script auto-falls-back to it |
| `Error opening data file .../ara.traineddata` | Tesseract doesn't have the language pack for the subtitle's script | Download the `.traineddata` file for that language into Tesseract's `tessdata` folder — see [1.3](#13-tesseract-ocr--ffmpeg-only-needed-for-extracting-burned-in-subtitles) |
| Script stopped partway through a long translation | Network hiccup, rate limit, or you cancelled it | Just rerun the exact same command — cached progress means it resumes instead of starting over (see [2.5](#25-if-it-gets-interrupted)) |
