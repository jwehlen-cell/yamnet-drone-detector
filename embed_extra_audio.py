"""
Convert raw drone-audio WAV files (saraalemadi/DroneAudioDataset and
mackenzie-jane/drone-visualization) into the same on-disk shape the ERAU
training pipeline expects: per-clip mean-pooled YAMNet 1024-d embeddings.

Output is written under ``data/extra/`` as ``.tfdata`` files using
``tf.io.serialize_tensor`` of a (1, 1024) float32 tensor. ``train.py``
already accepts that format, so the merged dataset can be loaded with no
further changes by pointing ``load_split`` at extra directory roots.

Mapping decisions
-----------------
* DroneAudioDataset/Binary_Drone_Audio/yes_drone
    -> data/extra/drone/bebop_or_mambo/   (filename prefix carries subtype)
* DroneAudioDataset/Binary_Drone_Audio/unknown
    -> data/extra/no_drone/dataset_esc50/
* DroneAudioDataset/Multiclass_Drone_Audio/bebop_1
    -> data/extra/drone/bebop/
* DroneAudioDataset/Multiclass_Drone_Audio/membo_1
    -> data/extra/drone/mambo/
* DroneAudioDataset/Multiclass_Drone_Audio/unknown
    -> skipped (duplicate of binary unknown)
* drone-visualization/public/droneAudio/*.wav
    -> data/extra/drone/visualization_samples/  (positives for binary; not
       used as subtype classes because there's only one clip per model)

This keeps the existing label semantics intact and adds two new drone
subtypes (``bebop``, ``mambo``) plus a large pool of diverse environmental
negatives for the binary head.
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
RAW_ROOT = ROOT / "data" / "extra_raw"
OUT_ROOT = ROOT / "data" / "extra"

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
TARGET_SR = 16_000
EMBEDDING_DIM = 1024
MIN_DURATION_S = 0.96  # YAMNet needs ~0.96 s to emit one frame


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


class Embedder:
    def __init__(self) -> None:
        print(f"Loading YAMNet from {YAMNET_URL} ...", flush=True)
        self._yamnet = hub.load(YAMNET_URL)

    def embed_file(self, wav_path: Path) -> np.ndarray | None:
        try:
            audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        except Exception as e:  # noqa: BLE001
            print(f"  skip (read error): {wav_path.name}: {e}")
            return None
        if audio.size == 0:
            return None
        audio = _to_yamnet_input(audio, sr)
        if audio.size < int(MIN_DURATION_S * TARGET_SR):
            # Pad with silence so YAMNet still emits one frame.
            pad = int(MIN_DURATION_S * TARGET_SR) - audio.size
            audio = np.pad(audio, (0, pad))
        wav = tf.constant(audio, dtype=tf.float32)
        _scores, embeddings, _spec = self._yamnet(wav)
        emb = embeddings.numpy()  # (T, 1024)
        if emb.size == 0:
            return None
        return emb.mean(axis=0, keepdims=True).astype(np.float32)  # (1, 1024)


def _write_tfdata(emb: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = tf.io.serialize_tensor(tf.constant(emb, dtype=tf.float32)).numpy()
    out_path.write_bytes(payload)


def _process_dir(
    embedder: Embedder,
    src_dir: Path,
    dst_dir: Path,
    limit: int | None = None,
) -> tuple[int, int]:
    files = sorted(src_dir.glob("*.wav"))
    if limit is not None:
        files = files[:limit]
    if not files:
        return 0, 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_skip = 0
    for wav in tqdm(files, desc=f"{src_dir.name} -> {dst_dir.name}"):
        out_path = dst_dir / (wav.stem + ".tfdata")
        if out_path.exists():
            n_ok += 1
            continue
        emb = embedder.embed_file(wav)
        if emb is None:
            n_skip += 1
            continue
        _write_tfdata(emb, out_path)
        n_ok += 1
    return n_ok, n_skip


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit-per-dir",
        type=int,
        default=None,
        help="Cap files per source dir (useful for a smoke test).",
    )
    args = parser.parse_args()

    saraalemadi = RAW_ROOT / "DroneAudioDataset"
    vis_samples = RAW_ROOT / "drone-visualization" / "public" / "droneAudio"
    esc50_conf  = RAW_ROOT / "esc50_confounders"
    droneaudset = RAW_ROOT / "droneaudioset" / "drone-only-wavs"

    missing = [p for p in (saraalemadi, vis_samples) if not p.is_dir()]
    if missing:
        print("Missing raw data; clone the datasets first:")
        for p in missing:
            print(f"  {p}")
        return 1

    embedder = Embedder()

    summary: dict[str, tuple[int, int]] = {}

    summary["bebop"] = _process_dir(
        embedder,
        saraalemadi / "Multiclass_Drone_Audio" / "bebop_1",
        OUT_ROOT / "drone" / "bebop",
        args.limit_per_dir,
    )
    summary["mambo"] = _process_dir(
        embedder,
        saraalemadi / "Multiclass_Drone_Audio" / "membo_1",
        OUT_ROOT / "drone" / "mambo",
        args.limit_per_dir,
    )

    # Binary positives that aren't already covered by the multiclass folders.
    # The yes_drone folder is a superset of bebop+mambo so we drop it here to
    # avoid double-counting; the multiclass folders already give us 666+666
    # subtype-labelled positives.

    # 32 single-clip-per-model demo WAVs from the 2025 multiclass arXiv paper.
    # We bucket these under "drone/visualization_samples/" — they're useful as
    # diverse positive examples for the binary head but should NOT form
    # individual subtype classes (one clip per model isn't enough to learn).
    summary["vis_samples"] = _process_dir(
        embedder,
        vis_samples,
        OUT_ROOT / "drone" / "visualization_samples",
        args.limit_per_dir,
    )

    # ESC-50 + speech-command noise pool that ships with DroneAudioDataset.
    # 10k+ clips would over-dominate the negatives; cap at 2000 (still 2x the
    # ERAU no_drone count) so the binary head sees diverse non-drone audio
    # without smothering the existing negatives.
    summary["esc50_noise"] = _process_dir(
        embedder,
        saraalemadi / "Binary_Drone_Audio" / "unknown",
        OUT_ROOT / "no_drone" / "esc50_noise",
        args.limit_per_dir if args.limit_per_dir else 2000,
    )

    # Targeted confounders: full ESC-50 exterior-sound classes 40-49
    # (helicopter, chainsaw, siren, car_horn, engine, train, church_bells,
    # airplane, fireworks, hand_saw). Each class has 40 clips; saraalemadi's
    # pool only includes ~21 each from classes 40-44, so adding these 400
    # roughly triples confounder coverage for the binary head and adds
    # 200 clips for classes 45-49 it had never seen.
    if esc50_conf.is_dir():
        summary["esc50_confounders"] = _process_dir(
            embedder,
            esc50_conf,
            OUT_ROOT / "no_drone" / "esc50_confounders",
            args.limit_per_dir,
        )

    # DroneAudioset (Bosch/AHL 2025) — pure rotor recordings from many
    # drone+throttle+environment combinations, downmixed to mono and
    # sliced into 1-sec chunks by /tmp/dl_droneaudioset.py. No subtype
    # labels available in the source dataset, so these go into the
    # binary head only via a dedicated "droneaudioset" subfolder which
    # SUBTYPE_EXCLUDED_LABELS in train.py drops before subtype training.
    if droneaudset.is_dir():
        summary["droneaudioset"] = _process_dir(
            embedder,
            droneaudset,
            OUT_ROOT / "drone" / "droneaudioset",
            args.limit_per_dir,
        )

    print("\n=== Summary (ok, skipped) ===")
    for k, (ok, sk) in summary.items():
        print(f"  {k:<22} ok={ok:>6}  skipped={sk}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
