"""
Embed the field/contributed WAV sets (QST sensor detections + USAFA DFEC)
into the same ``.tfdata`` YAMNet-embedding shape ``train.py`` consumes.

Unlike ``embed_extra_audio.py`` (which mean-pools each short source clip into
one vector), these sources contain *long* recordings — QST detection clips run
up to ~145 s, USAFA clips are 10 s. Mean-pooling a 145 s clip into a single
vector would smear everything together, so this script slices each recording
into fixed ~1 s windows (matching YAMNet's frame granularity and the ERAU
1-second convention) and writes one mean-pooled (1, 1024) embedding per window.

Label mapping (binary head; all new drone buckets are subtype-excluded in
train.py because they have no usable model-level subtype / too few clips):

  detection-audio/**/*.wav                 -> data/extra/drone/qst_detections/
  detection-audio-negatives/**/*.wav       -> data/extra/no_drone/qst_night_negatives/
  detection-audio-falsepos/**/*.wav        -> data/extra/no_drone/qst_verified_negatives/
  usafa-dfec-dataset-16k/{450,teal2,hunter}* -> data/extra/drone/usafa_dfec/
  usafa-dfec-dataset-16k/{box_fan,tow_planes}* -> data/extra/no_drone/usafa_dfec_negatives/
"""

from __future__ import annotations

import argparse
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import resample_poly
from tqdm import tqdm

ROOT = Path(__file__).parent
OUT_ROOT = ROOT / "data" / "extra"

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
TARGET_SR = 16_000
EMBEDDING_DIM = 1024
WINDOW_S = 1.0           # slice length
MIN_DURATION_S = 0.96    # YAMNet needs ~0.96 s to emit a frame


def _to_yamnet_input(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sample_rate != TARGET_SR:
        g = gcd(int(sample_rate), TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, sample_rate // g).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio


def _windows(audio: np.ndarray) -> list[np.ndarray]:
    """Non-overlapping ~1 s windows; drop a trailing remainder < 0.96 s.
    Clips shorter than one window are padded to the YAMNet minimum."""
    win = int(WINDOW_S * TARGET_SR)
    floor = int(MIN_DURATION_S * TARGET_SR)
    if audio.size < floor:
        return [np.pad(audio, (0, floor - audio.size))]
    out = []
    for start in range(0, audio.size, win):
        chunk = audio[start:start + win]
        if chunk.size < floor:
            break
        out.append(chunk)
    return out


class Embedder:
    def __init__(self) -> None:
        print(f"Loading YAMNet from {YAMNET_URL} ...", flush=True)
        self._yamnet = hub.load(YAMNET_URL)

    def embed_window(self, chunk: np.ndarray) -> np.ndarray | None:
        _scores, embeddings, _spec = self._yamnet(tf.constant(chunk, dtype=tf.float32))
        emb = embeddings.numpy()
        if emb.size == 0:
            return None
        return emb.mean(axis=0, keepdims=True).astype(np.float32)  # (1, 1024)


def _write_tfdata(emb: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = tf.io.serialize_tensor(tf.constant(emb, dtype=tf.float32)).numpy()
    out_path.write_bytes(payload)


def _usafa_is_drone(name: str) -> bool:
    n = name.lower()
    return not (n.startswith("box_fan") or n.startswith("tow_planes"))


def _process(embedder, wavs, dst_dir, stem_fn, force) -> tuple[int, int, int]:
    """Returns (files, windows_written, windows_skipped)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_files = n_win = n_skip = 0
    for wav in tqdm(wavs, desc=dst_dir.name):
        try:
            audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        except Exception as e:  # noqa: BLE001
            print(f"  skip (read error): {wav.name}: {e}")
            continue
        if audio.size == 0:
            continue
        n_files += 1
        audio = _to_yamnet_input(audio, sr)
        for i, chunk in enumerate(_windows(audio)):
            out_path = dst_dir / f"{stem_fn(wav)}_w{i:04d}.tfdata"
            if out_path.exists() and not force:
                n_win += 1
                continue
            emb = embedder.embed_window(chunk)
            if emb is None:
                n_skip += 1
                continue
            _write_tfdata(emb, out_path)
            n_win += 1
    return n_files, n_win, n_skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Re-embed even if output exists.")
    args = ap.parse_args()

    qst_dir = ROOT / "detection-audio"
    qst_neg_dir = ROOT / "detection-audio-negatives"
    qst_fp_dir = ROOT / "detection-audio-falsepos"
    usafa_dir = ROOT / "usafa-dfec-dataset-16k"
    if not qst_dir.is_dir() or not usafa_dir.is_dir():
        print(f"Missing input dirs: {qst_dir} and/or {usafa_dir}")
        return 1

    qst_wavs = sorted(qst_dir.rglob("*.wav"))
    qst_neg_wavs = sorted(qst_neg_dir.rglob("*.wav")) if qst_neg_dir.is_dir() else []
    qst_fp_wavs = sorted(qst_fp_dir.rglob("*.wav")) if qst_fp_dir.is_dir() else []
    usafa_wavs = sorted(usafa_dir.glob("*.wav"))
    usafa_drone = [w for w in usafa_wavs if _usafa_is_drone(w.name)]
    usafa_neg = [w for w in usafa_wavs if not _usafa_is_drone(w.name)]

    embedder = Embedder()
    summary = {}
    # QST: unique stem from date dir + filename (sensors/times don't collide, but be safe).
    summary["qst_detections"] = _process(
        embedder, qst_wavs, OUT_ROOT / "drone" / "qst_detections",
        lambda w: f"{w.parent.name}_{w.stem}", args.force,
    )
    summary["usafa_dfec (drone)"] = _process(
        embedder, usafa_drone, OUT_ROOT / "drone" / "usafa_dfec",
        lambda w: w.stem, args.force,
    )
    summary["usafa_dfec_negatives"] = _process(
        embedder, usafa_neg, OUT_ROOT / "no_drone" / "usafa_dfec_negatives",
        lambda w: w.stem, args.force,
    )
    # QST dead-of-night windows -> no_drone negatives (rebalances the
    # positive-heavy field set + gives held-out field negatives for eval).
    summary["qst_night_negatives"] = _process(
        embedder, qst_neg_wavs, OUT_ROOT / "no_drone" / "qst_night_negatives",
        lambda w: f"{w.parent.name}_{w.stem}", args.force,
    )
    # Kilo-verified false positives from the live Shaw deployment -> no_drone
    # hard negatives (daytime vehicle/water/wind confounders the night
    # negatives never covered). Pulled by pull_falsepos_audio.py.
    summary["qst_verified_negatives"] = _process(
        embedder, qst_fp_wavs, OUT_ROOT / "no_drone" / "qst_verified_negatives",
        lambda w: f"{w.parent.name}_{w.stem}", args.force,
    )

    print("\n=== Summary (files, windows_written, windows_skipped) ===")
    for k, (f, w, s) in summary.items():
        print(f"  {k:<24} files={f:>4}  windows={w:>6}  skipped={s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
