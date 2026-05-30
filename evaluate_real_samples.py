"""Run the trained detector against a curated set of real labeled drone
samples and report per-sample verdicts.

The samples come from the same source repos used in training, so each row
has a known ground-truth subtype. Three buckets:

  - **trained**: the *drone model* itself was a subtype class during
    training. We expect the binary head to fire AND the subtype top-1
    to match.
  - **same_family / untrained**: the drone is in the wider family of a
    trained class (e.g. DJI Mavic Mini 1 vs trained Mini 2) or has no
    trained equivalent at all. Binary should still fire; the subtype
    label reveals which trained class is acoustically closest, which is
    itself informative.
  - **negative**: not a drone (ESC-50 background noise). Binary should
    *not* fire.

The script requires the raw extra-data clones produced by
``embed_extra_audio.py``'s setup (``data/extra_raw/DroneAudioDataset`` and
``data/extra_raw/drone-visualization``). If they're absent, the script
skips affected rows and prints which paths it expected, so you can wire
in your own samples.

Run:
    python evaluate_real_samples.py
"""

from __future__ import annotations

import json
from collections import Counter
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
SARA = ROOT / "data" / "extra_raw" / "DroneAudioDataset"
VIS = ROOT / "data" / "extra_raw" / "drone-visualization" / "public" / "droneAudio"

# Mapping from subtype token -> ("Maker", "Model") display name. The token
# is what the classifier emits and what labels.json contains; the display
# pair is purely for human-readable output.
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


# Curated test cases. ``in_training=True`` means the *drone model* was a
# subtype class during training; we expect the top-1 to match ``gt``.
# Same-family / untrained rows still expect drone_prob >= 0.5 but the
# subtype label is informational only.
CASES: list[tuple[Path, str, bool, str]] = [
    # --- Trained subtype classes ---
    (SARA / "Multiclass_Drone_Audio/bebop_1/B_S2_D1_067-bebop_000_.wav",
     "bebop", True, "Parrot Bebop — in training (DroneAudioDataset)"),
    (SARA / "Multiclass_Drone_Audio/membo_1/Membo_0_000-membo_000_.wav",
     "mambo", True, "Parrot Mambo — in training (DroneAudioDataset)"),
    (VIS / "DJI_Mavic_Mini2_35.wav",
     "mavicmini", True, "DJI Mavic Mini 2 — same model as ERAU 'mavicmini'"),

    # --- Same-family but not the exact trained model ---
    (VIS / "DJI_Matrice_200_38.wav",
     "matrice", False, "DJI Matrice 200 — trained class was Matrice M100"),
    (VIS / "DJI_Matrice600p_12.wav",
     "matrice", False, "DJI Matrice 600 Pro — same maker, larger frame"),
    (VIS / "DJI_Mavic_Mini1_10.wav",
     "mavicmini", False, "DJI Mavic Mini 1 — trained class was Mini 2"),

    # --- Untrained drones (no equivalent class). Binary should fire;
    #     subtype label will collapse to whatever's acoustically nearest.
    (VIS / "DJI_FPV_33.wav",
     "?", False, "DJI FPV (racing drone) — no trained class"),
    (VIS / "DJI_Mavic2pro_81.wav",
     "?", False, "DJI Mavic 2 Pro — no trained class"),
    (VIS / "DJI_Phantom4_41.wav",
     "?", False, "DJI Phantom 4 — no trained class"),
    (VIS / "Autel_Evo_II_20.wav",
     "?", False, "Autel Evo II — non-DJI / non-Parrot"),
    (VIS / "Syma_X8SW_87.wav",
     "?", False, "Syma X8SW — consumer toy drone"),
    (VIS / "DJI_Tello_75.wav",
     "?", False, "DJI Tello (Ryze) — small palm-size drone"),

    # --- Non-drone negative ---
    (SARA / "Binary_Drone_Audio/unknown/1-100032-A-00.wav",
     "no_drone", True, "ESC-50 background noise — should NOT trigger"),
]


def _load_audio(path: Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        from scipy.signal import resample_poly
        g = gcd(int(sr), target_sr)
        audio = resample_poly(audio, target_sr // g, int(sr) // g).astype(np.float32)
        sr = target_sr
    if audio.size < int(0.96 * target_sr):
        audio = np.pad(audio, (0, int(0.96 * target_sr) - audio.size))
    return audio, sr


def main() -> int:
    print("Loading YAMNet + dense heads ...", flush=True)
    yam = hub.load("https://tfhub.dev/google/yamnet/1")
    binary = tf.keras.models.load_model(MODELS / "drone_classifier_binary.keras")
    subtype = tf.keras.models.load_model(MODELS / "drone_classifier_subtype.keras")
    labels = json.loads((MODELS / "drone_classifier_subtype.labels.json").read_text())

    def infer(audio: np.ndarray) -> tuple[float, dict[str, float]]:
        _scores, emb, _spec = yam(tf.constant(audio, dtype=tf.float32))
        pooled = emb.numpy().mean(axis=0, keepdims=True).astype(np.float32)
        drone_p = float(binary(pooled, training=False).numpy()[0, 0])
        sub = subtype(pooled, training=False).numpy()[0]
        return drone_p, {lab: float(p) for lab, p in zip(labels, sub.tolist())}

    print("\n=== Real labeled samples ===\n")
    header = (
        f"{'GT':<18}  {'drone_p':>7}  {'characterizer top':<22}  "
        f"{'top_p':>5}  {'dur':>5}  description"
    )
    print(header)
    print("-" * len(header))

    trained_correct = 0
    trained_total = 0
    untrained_hits = 0
    untrained_total = 0
    neg_correct = 0
    neg_total = 0
    collapse_targets: Counter[str] = Counter()
    missing: list[Path] = []

    for path, gt, in_training, desc in CASES:
        if not path.exists():
            missing.append(path)
            continue
        audio, sr = _load_audio(path)
        drone_p, sub = infer(audio)
        top_lab, top_p = max(sub.items(), key=lambda x: x[1])
        gt_display = fmt(gt) if gt in DISPLAY else gt
        pred_display = fmt(top_lab)
        dur = audio.size / sr

        if gt == "no_drone":
            neg_total += 1
            status = "OK" if drone_p < 0.5 else "FP"
            if status == "OK":
                neg_correct += 1
        elif in_training:
            trained_total += 1
            status = "OK" if top_lab == gt else "X "
            if status == "OK":
                trained_correct += 1
        else:
            untrained_total += 1
            status = "OK" if drone_p >= 0.5 else "FN"
            if status == "OK":
                untrained_hits += 1
                collapse_targets[top_lab] += 1

        print(
            f"{gt_display:<18}  {drone_p:>7.3f}  {pred_display:<22}  "
            f"{top_p:>5.3f}  {dur:>5.2f}  {status} {desc}"
        )

    print("\n=== Summary ===")
    if trained_total:
        print(f"Trained-class samples:   {trained_correct}/{trained_total} top-1 subtype correct")
    if untrained_total:
        print(f"Same-family / untrained: {untrained_hits}/{untrained_total} flagged as drone (binary)")
        if collapse_targets:
            print(f"  Untrained drones collapsed to these trained subtypes:")
            for lab, n in collapse_targets.most_common():
                print(f"    {fmt(lab):<22} {n}")
    if neg_total:
        print(f"Non-drone negative:      {neg_correct}/{neg_total} correctly rejected")

    if missing:
        print(
            f"\n{len(missing)} sample path(s) missing — clone the extra datasets first "
            f"(see README.md 'Re-training from scratch'). Missing:"
        )
        for p in missing:
            print(f"  {p}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
