"""Video in, the same typed decision out: the sound track if it carries speech, frames if not.

No open model under a billion parameters watches a clip and reads a question sheet, so the
video path does not pretend to be one. It splits the clip into the two things the engine can
already read, and chooses between them by what the clip contains:

* **Speech.** The sound track is pulled out at 16 kHz mono and transcribed. If the transcript
  holds real words, the decision is made on it through the text path, so fitted heads, the
  refit calibration, the conformal set and the route all apply, exactly as for audio. The
  transcript rides along, because a decision made on words the caller never saw cannot be
  audited.
* **Frames.** A silent clip (or one whose sound track is music, noise, or nothing) has a few
  evenly spaced frames sampled and tiled into one contact sheet, which goes through the image
  path. That is the vision tower's zero-shot readout: heads and calibration do not apply to
  pixels, and the answer says so with ``calibrated: False``.

Decoding goes through the ``ffmpeg`` binary when it is on the path. laya takes on no Python
dependency for video, and a machine without ffmpeg is told exactly that in one sentence rather
than failing somewhere inside a codec. Everything runs locally: the clip never leaves the
process.

    from laya.video import probe, extract_audio, sample_frames, contact_sheet
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import wave as _wave
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["ffmpeg_available", "probe", "extract_audio", "sample_frames", "contact_sheet",
           "has_speech", "as_path", "TARGET_SR"]

#: What whisper-family transcribers expect.
TARGET_SR = 16_000

#: Below this root-mean-square level the sound track is treated as silent and never sent to the
#: transcriber, which would otherwise invent words for it.
SILENCE_RMS = 1e-3

#: A transcript with fewer words than this is noise the transcriber dressed up, not speech.
MIN_WORDS = 3

_NO_FFMPEG = ("video needs the ffmpeg binary on the PATH (brew install ffmpeg, apt install "
              "ffmpeg, or https://ffmpeg.org/download.html); laya takes no Python dependency "
              "for it. Images and audio work without it.")


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _need_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError(_NO_FFMPEG)
    return exe


def _run(args: Sequence[str], what: str) -> bytes:
    proc = subprocess.run(list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        raise RuntimeError("ffmpeg could not %s: %s" % (what, " | ".join(tail) or "no output"))
    return proc.stdout


def as_path(video) -> Tuple[str, Optional[str]]:
    """A readable path for the clip, and the temporary file to delete afterwards, if any.

    ffmpeg needs a seekable input for most containers, so raw bytes are written to a temporary
    file rather than piped. A path or PathLike is used as given.
    """
    if isinstance(video, (bytes, bytearray)):
        fd, path = tempfile.mkstemp(suffix=".video")
        with os.fdopen(fd, "wb") as f:
            f.write(bytes(video))
        return path, path
    if isinstance(video, (str, os.PathLike)):
        path = os.fspath(video)
        if not os.path.isfile(path):
            raise FileNotFoundError("no video file at %r" % path)
        return path, None
    raise TypeError("video must be a path or raw video-file bytes, got %r" % type(video).__name__)


def probe(path: str) -> dict:
    """Duration in seconds (0.0 when the tool reports none) and whether the clip has an audio
    stream, from ffprobe when it is there and from ffmpeg's own banner when it is not."""
    _need_ffmpeg()
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        out = _run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                    "-show_entries", "stream=codec_type", "-of", "json", path], "probe the clip")
        info = json.loads(out.decode("utf-8", "replace") or "{}")
        dur = float((info.get("format") or {}).get("duration") or 0.0)
        kinds = {s.get("codec_type") for s in info.get("streams", [])}
        return {"duration_s": dur, "has_audio": "audio" in kinds, "has_video": "video" in kinds}
    proc = subprocess.run([shutil.which("ffmpeg"), "-i", path], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    text = proc.stderr.decode("utf-8", "replace")
    dur = 0.0
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            # A stream with no container duration -- an elementary .h264, a live capture --
            # banners "Duration: N/A". Leave 0.0, the same answer an absent `format.duration`
            # gets on the ffprobe path, instead of crashing the unpack or the int() reads.
            fields = line.split(",")[0].split("Duration:")[1].strip().split(":")
            if len(fields) == 3:
                try:
                    dur = int(fields[0]) * 3600 + int(fields[1]) * 60 + float(fields[2])
                except ValueError:
                    pass
    return {"duration_s": dur, "has_audio": "Audio:" in text, "has_video": "Video:" in text}


def extract_audio(path: str) -> Optional[np.ndarray]:
    """The sound track as a 1-D float32 waveform at 16 kHz mono, or None when the clip has
    no audio stream."""
    exe = _need_ffmpeg()
    if not probe(path)["has_audio"]:
        return None
    out = _run([exe, "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(TARGET_SR),
                "-f", "wav", "-acodec", "pcm_s16le", "-"], "extract the sound track")
    with _wave.open(io.BytesIO(out)) as w:
        frames = w.readframes(w.getnframes())
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return pcm


def has_speech(wave: Optional[np.ndarray], transcript: Optional[str] = None) -> bool:
    """The gate that chooses the path: a sound track quiet enough to be silence never reaches
    the transcriber, and a transcript shorter than a few words is not treated as speech."""
    if wave is None or wave.size == 0:
        return False
    if float(np.sqrt(np.mean(wave.astype(np.float64) ** 2))) < SILENCE_RMS:
        return False
    if transcript is None:
        return True            # loud enough to be worth transcribing; the caller decides after
    # Words with letters or digits in them. A tone or music comes back from whisper as rows of
    # dots or dashes, which split() counts as words and this does not.
    return len(re.findall(r"[^\W_]+", transcript)) >= MIN_WORDS


def sample_frames(path: str, n: int = 4, duration_s: Optional[float] = None) -> Tuple[List, List[float]]:
    """``n`` frames at evenly spaced times, as PIL RGB images, with the times they came from.

    Times sit at the centres of ``n`` equal slices, so a two-frame sample of a ten-second clip
    reads 2.5 s and 7.5 s rather than the first and last frame, which are often black. A time
    whose frame ffmpeg cannot produce is skipped, and the times returned are the ones the
    frames actually came from.
    """
    exe = _need_ffmpeg()
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("video frames need Pillow: pip install pillow") from e
    if duration_s is None:
        duration_s = probe(path)["duration_s"]
    n = max(1, int(n))
    if duration_s <= 0:
        times = [0.0]
    else:
        times = [duration_s * (i + 0.5) / n for i in range(n)]
    frames, got = [], []
    for t in times:
        png = _run([exe, "-v", "error", "-ss", "%.3f" % t, "-i", path, "-frames:v", "1",
                    "-f", "image2pipe", "-vcodec", "png", "-"], "read a frame at %.1fs" % t)
        if not png:
            continue
        with Image.open(io.BytesIO(png)) as im:
            frames.append(im.convert("RGB"))
        got.append(t)
    if not frames:
        raise RuntimeError("no frame could be read from %r" % path)
    return frames, got


def contact_sheet(frames: Sequence, cell: int = 512):
    """The frames tiled left to right, top to bottom, into one image the vision tower reads
    at once. Each frame is fitted into a ``cell`` square, keeping its aspect ratio."""
    from PIL import Image
    if not frames:
        raise ValueError("no frames to tile")
    n = len(frames)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    sheet = Image.new("RGB", (cols * cell, rows * cell), (0, 0, 0))
    for i, fr in enumerate(frames):
        fr = fr.copy()
        fr.thumbnail((cell, cell))
        x = (i % cols) * cell + (cell - fr.width) // 2
        y = (i // cols) * cell + (cell - fr.height) // 2
        sheet.paste(fr, (x, y))
    return sheet
