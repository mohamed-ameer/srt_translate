#!/usr/bin/env python3
"""
Extract hardcoded/burned-in subtitles from a video using OCR and create an SRT.

Requirements:
  pip install opencv-python pytesseract tqdm numpy
  Install Tesseract OCR separately and make sure Arabic language data is installed.

  Some opencv-python wheels are built WITHOUT FFmpeg support (check with
  `python -c "import cv2; print(cv2.getBuildInformation())" | grep FFMPEG`).
  If so, cv2 cannot open ANY video file, regardless of which file you pass it.
  Install system ffmpeg as a fallback decoder:
    macOS:   brew install ffmpeg
    Ubuntu:  sudo apt install ffmpeg
  This script auto-detects that case and falls back to shelling out to
  `ffmpeg`/`ffprobe` for frame extraction.

Examples:
  python extract_hardcoded_subtitles.py movie.mp4
  python extract_hardcoded_subtitles.py movie.mp4 -o subtitles.srt --lang ara
  python extract_hardcoded_subtitles.py movie.mp4 --bottom 0.72 --top 0.92 --sample 0.25
  python extract_hardcoded_subtitles.py movie.mp4 --backend ffmpeg   # force the fallback

Notes:
- This is OCR, not subtitle-stream extraction.
- It works best when subtitles are in a fixed area and have good contrast.
- For Arabic, Tesseract's Arabic trained data ("ara") is required.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from tqdm import tqdm


def check_tesseract():
    if shutil.which("tesseract") is None:
        print("ERROR: Tesseract OCR was not found in PATH.")
        print("Install Tesseract and then run this program again.")
        sys.exit(1)


def cv2_has_ffmpeg():
    build_info = cv2.getBuildInformation()
    line = next((l for l in build_info.splitlines() if "FFMPEG" in l), "")
    return "YES" in line


def diagnose_video_issue(video_path):
    print("Diagnosing why the video won't open...")

    try:
        size = video_path.stat().st_size
        print(f"  File size: {size / 1024 / 1024:.1f} MB"
              + ("  <-- empty file!" if size == 0 else ""))
    except OSError as e:
        print(f"  Could not stat file: {e}")

    print(f"  OpenCV built with FFmpeg support: {'YES' if cv2_has_ffmpeg() else 'NO'}")

    if shutil.which("ffprobe"):
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", str(video_path)],
                capture_output=True, text=True, timeout=20,
            )
            if r.returncode != 0:
                print(f"  ffprobe ALSO failed to read this file:\n{r.stderr.strip()[:800]}")
                print("  -> The file itself is likely corrupted, incomplete, or unsupported.")
            else:
                print("  ffprobe CAN read this file fine — this is an OpenCV/FFmpeg-build "
                      "gap, not a broken file. The --backend ffmpeg fallback should work.")
        except Exception as e:
            print(f"  ffprobe check itself failed to run: {e}")
    else:
        print("  ffprobe/ffmpeg not found on PATH.")
        print("  Install it: `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Linux).")
    print()


class Cv2FrameSource:
    """Reads frames via OpenCV's own VideoCapture (fast, seeks natively)."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError("cv2.VideoCapture could not open the file")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps else 0

        # Some containers report a valid open but immediately fail every read
        # (exactly the "Couldn't read movie file" case) — catch that here too,
        # rather than only discovering it mid-loop.
        ok, _ = self.cap.read()
        if not ok:
            self.cap.release()
            raise RuntimeError("cv2.VideoCapture opened the file but cannot decode any frames")
        self.cap.set(cv2.CAP_PROP_POS_MSEC, 0)

    def get_frame_at(self, t):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


class FfmpegFrameSource:
    """Reads frames by shelling out to the ffmpeg/ffprobe CLI per sampled
    timestamp. Slower than cv2 (spawns a process per frame) but works
    regardless of what codecs cv2's own build was compiled with."""

    def __init__(self, path):
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise RuntimeError("ffmpeg/ffprobe not found on PATH")

        self.path = str(path)

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-show_entries", "format=duration",
             "-of", "json", self.path],
            capture_output=True, text=True, timeout=30,
        )
        if probe.returncode != 0 or not probe.stdout:
            raise RuntimeError(f"ffprobe failed to read the file: {probe.stderr.strip()}")

        data = json.loads(probe.stdout)
        stream = data["streams"][0]
        self.width = int(stream["width"])
        self.height = int(stream["height"])

        num, _, den = stream["r_frame_rate"].partition("/")
        den = den or "1"
        self.fps = (float(num) / float(den)) if float(den) else 25.0

        self.duration = float(data["format"]["duration"])
        self.frame_count = int(self.duration * self.fps)

    def get_frame_at(self, t):
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", str(max(t, 0)), "-i", self.path,
             "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0 or not result.stdout:
            return None
        arr = np.frombuffer(result.stdout, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def release(self):
        pass


def open_video(video_path, backend):
    """Returns an opened frame source, trying cv2 first (fast) and falling
    back to the ffmpeg CLI (slower, but works when cv2's build lacks a
    usable video decoder) — unless a specific backend was forced via --backend."""
    errors = []

    if backend in ("auto", "cv2"):
        try:
            return Cv2FrameSource(video_path)
        except Exception as e:
            errors.append(f"cv2 backend: {e}")
            if backend == "cv2":
                diagnose_video_issue(video_path)
                print("\n".join(errors))
                sys.exit(1)
            print(f"  [info] cv2 backend failed ({e}); trying ffmpeg fallback...")

    if backend in ("auto", "ffmpeg"):
        try:
            return FfmpegFrameSource(video_path)
        except Exception as e:
            errors.append(f"ffmpeg backend: {e}")

    diagnose_video_issue(video_path)
    print("ERROR: Could not open video with any available backend:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="OCR hardcoded subtitles from a video and create an SRT."
    )
    p.add_argument("video", help="Input video file")
    p.add_argument(
        "-o", "--output",
        help="Output SRT path. Default: <video_name>.srt"
    )
    p.add_argument(
        "--lang",
        default="ara",
        help="Tesseract language, default: ara"
    )
    p.add_argument(
        "--sample",
        type=float,
        default=0.50,
        help="Analyze one frame every N seconds. Default: 0.50"
    )
    p.add_argument(
        "--top",
        type=float,
        default=0.72,
        help="Subtitle crop top as fraction of video height. Default: 0.72"
    )
    p.add_argument(
        "--bottom",
        type=float,
        default=0.98,
        help="Subtitle crop bottom as fraction of video height. Default: 0.98"
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=35,
        help="Minimum OCR confidence. Default: 35"
    )
    p.add_argument(
        "--min-change",
        type=float,
        default=0.18,
        help="Image difference threshold used to detect subtitle changes. Default: 0.18"
    )
    p.add_argument(
        "--backend",
        choices=["auto", "cv2", "ffmpeg"],
        default="auto",
        help="Video decoding backend. 'auto' tries cv2 then falls back to the "
             "ffmpeg CLI. Default: auto"
    )
    return p.parse_args()


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Remove common OCR artifacts.
    text = text.strip("-–—|~_ ")

    return text


def timestamp(seconds):
    if seconds < 0:
        seconds = 0

    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    ms = total_ms % 1000

    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Upscale improves OCR for small subtitles.
    gray = cv2.resize(
        gray,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    # Increase contrast.
    gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    # Binary image. Otsu is usually good for white subtitles.
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Small cleanup.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


def ocr_subtitle(frame, lang, min_confidence):
    processed = preprocess(frame)

    config = "--oem 1 --psm 6"

    data = pytesseract.image_to_data(
        processed,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    words = []

    for i, txt in enumerate(data["text"]):
        txt = clean_text(txt)
        if not txt:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1

        if confidence >= min_confidence:
            words.append(txt)

    return clean_text(" ".join(words))


def image_difference(a, b):
    if a is None or b is None:
        return 1.0

    diff = cv2.absdiff(a, b)
    return float(diff.mean()) / 255.0


def main():
    args = parse_args()
    check_tesseract()

    video_path = Path(args.video)

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    output = Path(args.output) if args.output else video_path.with_suffix(".srt")

    source = open_video(video_path, args.backend)

    fps = source.fps
    duration = source.duration
    width = source.width
    height = source.height

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"OCR language: {args.lang}")
    print(
        f"Subtitle crop: {args.top:.2f} - {args.bottom:.2f} "
        f"of video height"
    )
    print()

    # Previous grayscale subtitle crop.
    previous_crop = None

    # Current subtitle segment.
    current_text = ""
    current_start = None

    subtitles = []

    t = 0.0

    with tqdm(total=int(duration / args.sample) + 1, desc="OCR") as progress:
        while t <= duration:
            frame = source.get_frame_at(t)

            if frame is None:
                # A single unreadable timestamp isn't necessarily end-of-video
                # (especially with the ffmpeg-per-frame backend), so skip it
                # rather than aborting the whole run.
                t += args.sample
                progress.update(1)
                continue

            y1 = max(0, int(height * args.top))
            y2 = min(height, int(height * args.bottom))

            crop = frame[y1:y2, :]

            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            changed = True

            if previous_crop is not None:
                changed = (
                    image_difference(gray_crop, previous_crop)
                    >= args.min_change
                )

            # OCR periodically. For subtitle extraction, OCR every sample
            # is more reliable than OCR only after a large image change.
            text = ocr_subtitle(
                crop,
                args.lang,
                args.min_confidence
            )

            if text:
                if not current_text:
                    current_text = text
                    current_start = t

                elif text != current_text:
                    # Close previous subtitle.
                    if current_start is not None:
                        end = max(current_start + 0.1, t)

                        subtitles.append(
                            (current_start, end, current_text)
                        )

                    current_text = text
                    current_start = t

            else:
                # No OCR text. Close the current subtitle only after
                # a subtitle-sized gap.
                if current_text and current_start is not None:
                    end = max(current_start + 0.1, t)

                    subtitles.append(
                        (current_start, end, current_text)
                    )

                    current_text = ""
                    current_start = None

            previous_crop = gray_crop
            t += args.sample
            progress.update(1)

    source.release()

    # Close final subtitle.
    if current_text and current_start is not None:
        subtitles.append(
            (current_start, max(current_start + 0.1, duration), current_text)
        )

    # Merge duplicate/near-identical consecutive OCR results.
    merged = []

    for start, end, text in subtitles:
        if not text:
            continue

        if merged:
            ps, pe, pt = merged[-1]

            if text == pt and start - pe <= 1.0:
                merged[-1] = (ps, end, pt)
                continue

        merged.append((start, end, text))

    # Write SRT.
    with open(output, "w", encoding="utf-8-sig") as f:
        for index, (start, end, text) in enumerate(merged, start=1):
            f.write(f"{index}\n")
            f.write(f"{timestamp(start)} --> {timestamp(end)}\n")
            f.write(f"{text}\n\n")

    print()
    print(f"Done.")
    print(f"Detected subtitles: {len(merged)}")
    print(f"SRT: {output}")


if __name__ == "__main__":
    main()
