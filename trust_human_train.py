"""
Human-trust override: the 04:11 both-fire event is a TRAIN (operator-confirmed,
corroborated by AST -> Train/Railroad/Rail transport), NOT a drone -- even though
all 3 detectors (Angels Envy + Kilo + GANON) fired on it.

Pulls the confirmed-train native clips, saves them, embeds them as a labeled
no_drone HARD NEGATIVE (data/extra/no_drone/train_negatives/), and records the
verdict in false-detects/human_labels.json. Does NOT retrain (the broader
re-labeling will be redone later with new rules).
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import resample_poly

ROOT = Path(__file__).parent
GCS = "gs://aftac-argos-dataflow-unzipped/ensco/SH"
WAV_OUT = ROOT / "detection-audio-confounders" / "train"
EMB_OUT = ROOT / "data" / "extra" / "no_drone" / "train_negatives"
LABELS = ROOT / "false-detects" / "human_labels.json"

# The operator-confirmed 04:11 train pass (the both-fire clips, multi-sensor).
TRAIN_CLIPS = [
    ("SH009", "2026-06-04T04:11:34Z"),
    ("SH002", "2026-06-04T04:11:36Z"),
    ("SH002", "2026-06-04T04:11:40Z"),
    ("SH009", "2026-06-04T04:11:42Z"),
    ("SH010", "2026-06-04T04:11:44Z"),
]


def native_uri(sensor: str, t: datetime) -> str:
    stem = f"SH.{sensor}.Scell.{t:%Y%m%d_%H%M%S}"
    return f"{GCS}/{sensor}/{t:%Y/%m/%d/%H}/{stem}/{stem}.wav"


def main() -> int:
    WAV_OUT.mkdir(parents=True, exist_ok=True)
    EMB_OUT.mkdir(parents=True, exist_ok=True)
    print("loading YAMNet ...", flush=True)
    yam = hub.load("https://tfhub.dev/google/yamnet/1")

    saved = 0
    with tempfile.TemporaryDirectory() as td:
        for sensor, ts in TRAIN_CLIPS:
            t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            dst = Path(td) / f"{sensor}_{t:%H%M%S}.wav"
            if subprocess.run(["gsutil", "cp", native_uri(sensor, t), str(dst)],
                              capture_output=True).returncode != 0 or not dst.exists():
                print(f"  miss: {sensor} {ts}")
                continue
            sr, i16 = wavfile.read(str(dst))
            if sr != 8000 or not i16.size:
                continue
            a = i16.astype(np.float32) / 32768.0
            # save 16 kHz wav for the record
            a16 = resample_poly(a, 2, 1).astype(np.float32)
            sf.write(str(WAV_OUT / f"{sensor}_{t:%Y%m%d_%H%M%S}_train.wav"), a16, 16000)
            # embed as ~1 s YAMNet windows -> no_drone/train_negatives
            win = 16000
            for i in range(0, len(a16), win):
                chunk = a16[i:i + win]
                if chunk.size < int(0.96 * 16000):
                    break
                _s, emb, _sp = yam(tf.constant(chunk, tf.float32))
                e = emb.numpy().mean(axis=0, keepdims=True).astype(np.float32)
                payload = tf.io.serialize_tensor(tf.constant(e, tf.float32)).numpy()
                (EMB_OUT / f"{sensor}_{t:%H%M%S}_w{i//win:04d}.tfdata").write_bytes(payload)
            saved += 1
            print(f"  train negative: {sensor} {t:%H:%M:%S}")

    n_emb = len(list(EMB_OUT.glob("*.tfdata")))
    label_rec = {
        "event": "04:11 multi-sensor pass",
        "human_label": "train",
        "is_drone": False,
        "basis": "operator ear (high confidence) + AST AudioSet (Train / Railroad car / "
                 "Rail transport in top-8); overrides the 3-detector (AE+Kilo+GANON) drone vote",
        "clips": [{"sensor": s, "ts": ts} for s, ts in TRAIN_CLIPS],
        "action": "embedded as no_drone/train_negatives hard negatives; not yet retrained",
    }
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(LABELS.read_text()) if LABELS.exists() else []
    existing = [r for r in existing if r.get("event") != label_rec["event"]] + [label_rec]
    LABELS.write_text(json.dumps(existing, indent=2))
    print(f"\n{saved} train clips saved | {n_emb} train-negative windows embedded -> {EMB_OUT}")
    print(f"recorded human verdict -> {LABELS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
