"""Extract distinct drone-active segments from the YouTube short Nl_DgGbxCbw.

The clip contains six audibly distinct drones across ~16 seconds. This
script:

  1. Downloads the audio via yt-dlp if not already cached locally,
     normalized to 16 kHz mono PCM_16 WAV under
     ``data/extra_raw/youtube_Nl_DgGbxCbw/source.wav``.
  2. Slides the trained binary head over the clip at 0.1 s hop to find
     drone-active runs (drone_prob >= 0.5).
  3. Greedy-merges adjacent runs across the smallest inter-run gaps
     until exactly N segments remain (N defaults to 6, matching the
     external ground truth that the clip contains six distinct drones).
  4. Writes each segment as
     ``segments/drone_<idx>_t<start_ms>-<end_ms>.wav`` plus a summary
     JSON with timing + detector top-1 per segment.

The output WAVs are gitignored under ``data/extra_raw/`` and serve as
inputs to ``evaluate_real_samples.py``.

Run:
    python extract_youtube_segments.py
    python extract_youtube_segments.py --target-segments 7  # if you want a different count
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
OUT_DIR = ROOT / "data" / "extra_raw" / "youtube_Nl_DgGbxCbw"
SEG_DIR = OUT_DIR / "segments"
SOURCE_WAV = OUT_DIR / "source.wav"
YOUTUBE_URL = "https://www.youtube.com/shorts/Nl_DgGbxCbw"
SR = 16_000

# Maker + model display names per trained subtype token. The classifier
# emits the short token (matrice / mavic3 / mavicmini / mambo / bebop /
# no_drone); this mapping is the source of truth for the human-readable
# "<maker> <model>" string used in console output and summary.json.
# Kept in sync with the same table in evaluate_real_samples.py.
DISPLAY = {
    "bebop":     ("Parrot", "Bebop"),
    "mambo":     ("Parrot", "Mambo"),
    "matrice":   ("DJI",    "Matrice M100"),
    "mavic3":    ("DJI",    "Mavic 3"),
    "mavicmini": ("DJI",    "Mavic Mini 2"),
    "no_drone":  ("",       "no_drone"),
}


def fmt(label: str) -> str:
    maker, model = DISPLAY.get(label, ("?", label))
    return f"{maker} {model}".strip()


def _ensure_source() -> None:
    """Download (or copy from /tmp cache) the YouTube clip if absent."""
    if SOURCE_WAV.exists():
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cached = Path("/tmp/yt_drone/Nl_DgGbxCbw.wav")
    if cached.exists():
        shutil.copy(cached, SOURCE_WAV)
        print(f"Reused cached download {cached} -> {SOURCE_WAV}")
        return
    print(f"Downloading {YOUTUBE_URL} via yt-dlp ...")
    cmd = [
        sys.executable.replace("python", "yt-dlp"),  # venv-local yt-dlp
        "-f", "bestaudio",
        "--extract-audio", "--audio-format", "wav", "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", str(OUT_DIR / "%(id)s.%(ext)s"),
        YOUTUBE_URL,
    ]
    subprocess.run(cmd, check=True)
    downloaded = OUT_DIR / "Nl_DgGbxCbw.wav"
    if downloaded.exists() and downloaded != SOURCE_WAV:
        downloaded.rename(SOURCE_WAV)


def _resample_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != SR:
        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        audio = resample_poly(audio, SR // g, int(sr) // g).astype(np.float32)
    return audio


def _energy_segments(
    audio: np.ndarray,
    frame_ms: float = 20.0,
    smooth_ms: float = 80.0,
    silence_db_below_peak: float = 18.0,
    min_silence_ms: float = 200.0,
    min_segment_ms: float = 300.0,
) -> list[tuple[float, float]]:
    """Split audio into sound-active segments using short-time RMS energy.

    Algorithm:
      1. Compute RMS energy in non-overlapping ``frame_ms`` frames.
      2. Smooth with a moving average of width ``smooth_ms``.
      3. Mark a frame as "silence" if its level is below
         ``peak - silence_db_below_peak`` dB.
      4. Collapse silence runs shorter than ``min_silence_ms`` (a brief
         dip in the middle of a drone burst shouldn't split it).
      5. Segments = contiguous non-silence runs >= ``min_segment_ms``.

    Returns [(start_s, end_s)] in clip time.
    """
    frame_n = max(1, int(frame_ms * SR / 1000))
    n_frames = audio.size // frame_n
    if n_frames == 0:
        return []
    framed = audio[: n_frames * frame_n].reshape(n_frames, frame_n)
    rms = np.sqrt((framed.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    # dB relative to peak; smaller = quieter
    db = 20.0 * np.log10(rms / rms.max())

    # Smooth
    k = max(1, int(smooth_ms / frame_ms))
    pad = np.pad(db, k // 2, mode="edge")
    db_smooth = np.convolve(pad, np.ones(k) / k, mode="valid")[:n_frames]

    silence = db_smooth < (-silence_db_below_peak)

    # Collapse silence runs shorter than min_silence_ms (treat as still-active).
    min_silence_frames = max(1, int(min_silence_ms / frame_ms))
    i = 0
    while i < len(silence):
        if silence[i]:
            j = i
            while j < len(silence) and silence[j]:
                j += 1
            if (j - i) < min_silence_frames:
                silence[i:j] = False
            i = j
        else:
            i += 1

    # Active runs between silences.
    segments: list[tuple[float, float]] = []
    i = 0
    while i < len(silence):
        if not silence[i]:
            j = i
            while j < len(silence) and not silence[j]:
                j += 1
            s = i * frame_n / SR
            e = j * frame_n / SR
            if (e - s) * 1000 >= min_segment_ms:
                segments.append((s, e))
            i = j
        else:
            i += 1
    return segments


def _merge_to_target(
    runs: list[tuple[float, float]], target: int
) -> list[tuple[float, float]]:
    """Greedy-merge across the smallest inter-run gaps until len == target."""
    work = list(runs)
    if len(work) <= target:
        return work
    while len(work) > target:
        gaps = [(work[i + 1][0] - work[i][1], i) for i in range(len(work) - 1)]
        gaps.sort()
        i = gaps[0][1]
        work[i] = (work[i][0], work[i + 1][1])
        del work[i + 1]
    return work


def _characterize(
    yam, binary, subtype, labels, audio: np.ndarray, hop_s: float = 0.1
) -> dict:
    """Mean-pool over the whole segment + sliding-window peak.

    Mean-pool reflects "how drone-y is this segment averaged" — sensitive
    to silence padding around a short drone burst. Peak reflects "what
    would the production K-of-N detector see at its best moment" — closer
    to the operational metric.
    """
    wav = tf.constant(audio, dtype=tf.float32)
    _scores, emb, _spec = yam(wav)
    emb_np = emb.numpy()
    pooled = emb_np.mean(axis=0, keepdims=True).astype(np.float32)
    mean_drone_p = float(binary(pooled, training=False).numpy()[0, 0])
    mean_sub = subtype(pooled, training=False).numpy()[0]
    mean_idx = int(np.argmax(mean_sub))

    # Sliding-window peak within this segment.
    win_n = int(1.0 * SR)
    hop_n = max(1, int(hop_s * SR))
    peak_drone_p = 0.0
    peak_top = labels[mean_idx]
    peak_top_p = float(mean_sub[mean_idx])
    if audio.size >= win_n:
        for start in range(0, audio.size - win_n + 1, hop_n):
            w = tf.constant(audio[start : start + win_n], dtype=tf.float32)
            _s, e, _sp = yam(w)
            p_pooled = e.numpy().mean(axis=0, keepdims=True).astype(np.float32)
            p = float(binary(p_pooled, training=False).numpy()[0, 0])
            if p > peak_drone_p:
                peak_drone_p = p
                ssub = subtype(p_pooled, training=False).numpy()[0]
                pi = int(np.argmax(ssub))
                peak_top = labels[pi]
                peak_top_p = float(ssub[pi])
    return {
        "mean_drone_p": mean_drone_p,
        "mean_top": labels[mean_idx],
        "mean_top_p": float(mean_sub[mean_idx]),
        "peak_drone_p": peak_drone_p,
        "peak_top": peak_top,
        "peak_top_p": peak_top_p,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-segments", type=int, default=6)
    parser.add_argument(
        "--silence-db",
        type=float,
        default=18.0,
        help="Frames quieter than (peak - this dB) are treated as silence.",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=float,
        default=200.0,
        help="A silence gap shorter than this is ignored (treated as still-active).",
    )
    parser.add_argument(
        "--min-segment-ms",
        type=float,
        default=300.0,
        help="Segments shorter than this are discarded.",
    )
    parser.add_argument(
        "--pad-s",
        type=float,
        default=0.2,
        help="Audio context added to each side of every segment when writing the WAV.",
    )
    args = parser.parse_args()

    _ensure_source()
    audio, sr = sf.read(str(SOURCE_WAV), dtype="float32")
    audio = _resample_mono_16k(audio, sr)
    total_s = audio.size / SR
    print(f"Source: {SOURCE_WAV} ({total_s:.2f}s @ {SR} Hz mono)")

    print("Loading YAMNet + dense heads ...", flush=True)
    yam = hub.load("https://tfhub.dev/google/yamnet/1")
    binary = tf.keras.models.load_model(MODELS / "drone_classifier_binary.keras")
    subtype = tf.keras.models.load_model(MODELS / "drone_classifier_subtype.keras")
    labels = json.loads((MODELS / "drone_classifier_subtype.labels.json").read_text())

    print("Energy-based segmentation ...")
    raw_segments = _energy_segments(
        audio,
        frame_ms=20.0,
        smooth_ms=80.0,
        silence_db_below_peak=args.silence_db,
        min_silence_ms=args.min_silence_ms,
        min_segment_ms=args.min_segment_ms,
    )
    print(f"  found {len(raw_segments)} energy-active segments")

    # If the first energy segment starts > 0.5s into the clip and the
    # 0-1.0s window scores drone_prob >= 0.5, prepend it. Brief drones at
    # the start (the Mavic Mini 2 burst at t=0.0 in this clip) can sit
    # below the global energy peak threshold and get missed.
    if raw_segments and raw_segments[0][0] > 0.5:
        head = audio[: int(1.0 * SR)]
        w = tf.constant(head, dtype=tf.float32)
        _s, em, _sp = yam(w)
        p_pooled = em.numpy().mean(axis=0, keepdims=True).astype(np.float32)
        head_p = float(binary(p_pooled, training=False).numpy()[0, 0])
        if head_p >= 0.5:
            print(f"  prepending 0.0-1.0s drone hit (drone_p={head_p:.3f})")
            raw_segments.insert(0, (0.0, 1.0))

    # Score every candidate by peak sliding-window drone_p for ranking.
    def _peak_drone_p(s: float, e: float) -> float:
        si, ei = int(s * SR), int(e * SR)
        chunk = audio[si:ei]
        win_n = int(1.0 * SR)
        hop_n = max(1, int(0.1 * SR))
        if chunk.size < win_n:
            chunk = np.pad(chunk, (0, win_n - chunk.size))
        peak = 0.0
        for st in range(0, chunk.size - win_n + 1, hop_n):
            w = tf.constant(chunk[st : st + win_n], dtype=tf.float32)
            _s, em, _sp = yam(w)
            p_pooled = em.numpy().mean(axis=0, keepdims=True).astype(np.float32)
            peak = max(peak, float(binary(p_pooled, training=False).numpy()[0, 0]))
        return peak

    scored = [(s, e, _peak_drone_p(s, e)) for s, e in raw_segments]
    # If we have more than target, drop the lowest-drone-prob extras
    # (most likely voiceover / music / silence). Otherwise greedy-merge
    # smallest gaps as before.
    if len(scored) > args.target_segments:
        scored.sort(key=lambda r: -r[2])  # highest peak_drone_p first
        kept = sorted(scored[: args.target_segments], key=lambda r: r[0])
        dropped = scored[args.target_segments :]
        for s, e, p in dropped:
            print(f"  dropping lowest-drone-prob segment t={s:.2f}-{e:.2f}s (peak={p:.3f})")
        segments = [(s, e) for s, e, _ in kept]
    else:
        segments = _merge_to_target([(s, e) for s, e, _ in scored], args.target_segments)
    print(f"  final {len(segments)} segments (target={args.target_segments})")

    if SEG_DIR.exists():
        shutil.rmtree(SEG_DIR)
    SEG_DIR.mkdir(parents=True)

    summary: list[dict] = []
    for i, (s_raw, e_raw) in enumerate(segments, start=1):
        # Pad the segment but clip to clip boundaries.
        s = max(0.0, s_raw - args.pad_s)
        e = min(total_s, e_raw + args.pad_s)
        si = int(round(s * SR))
        ei = int(round(e * SR))
        chunk = audio[si:ei]
        # Run the full characterizer on the chunk (mean-pool + sliding peak).
        c = _characterize(yam, binary, subtype, labels, chunk)
        path = SEG_DIR / f"drone_{i:02d}_t{int(s*1000):05d}-{int(e*1000):05d}.wav"
        sf.write(str(path), chunk, SR, subtype="PCM_16")
        row = {
            "idx": i,
            "path": str(path.relative_to(ROOT)),
            "start_s": round(s, 3),
            "end_s": round(e, 3),
            "duration_s": round(e - s, 3),
            "active_window_start_s": round(s_raw, 3),
            "active_window_end_s": round(e_raw, 3),
            "mean_drone_p": round(c["mean_drone_p"], 4),
            "mean_top": c["mean_top"],
            "mean_top_display": fmt(c["mean_top"]),
            "mean_top_p": round(c["mean_top_p"], 4),
            "peak_drone_p": round(c["peak_drone_p"], 4),
            "peak_top": c["peak_top"],
            "peak_top_display": fmt(c["peak_top"]),
            "peak_top_p": round(c["peak_top_p"], 4),
        }
        summary.append(row)
        print(
            f"  segment {i}  t={s:6.2f}-{e:6.2f}s  "
            f"mean drone_p={c['mean_drone_p']:.3f} top={fmt(c['mean_top']):<22}({c['mean_top_p']:.2f})  "
            f"peak drone_p={c['peak_drone_p']:.3f} top={fmt(c['peak_top']):<22}({c['peak_top_p']:.2f})"
        )

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {len(summary)} WAVs under {SEG_DIR.relative_to(ROOT)}/")
    print(f"Wrote summary to {(OUT_DIR / 'summary.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
