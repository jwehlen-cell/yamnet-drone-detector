"""
Verify #1 (new model @0.70) + #2 (AudioSet confounder veto) on the same 41
last-hour Shaw detections, using YAMNet's own AudioSet scores exactly as the
inference worker does. Also runs the real USAFA drones as a control to prove the
veto never suppresses a genuine drone.

final_fire = drone_score >= 0.70  AND  max(confounder AudioSet) < 0.30
"""
from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly
import tensorflow as tf
import tensorflow_hub as hub

ROOT = Path(__file__).parent
GCS = "gs://aftac-argos-dataflow-unzipped/ensco/SH"
EVENTS = json.loads((ROOT / "false-detects" / "lasthour_events.json").read_text())
AE_THR, VETO_THR = 0.70, 0.30
# Safe set: confounders a DRONE does not trigger (engine/vehicle family excluded
# because YAMNet scores real drones as "Vehicle" 0.38-0.53 at field SNR).
CONFOUNDERS = {"Frog", "Croak", "Insect", "Cricket",
               "Train", "Railroad car, train wagon", "Rail transport", "Train horn"}

yam = hub.load("https://tfhub.dev/google/yamnet/1")
clf = tf.keras.models.load_model(ROOT / "models/drone_classifier_binary.keras", compile=False)
names = [r["display_name"] for r in csv.DictReader(open(yam.class_map_path().numpy().decode()))]
conf_idx = [i for i, n in enumerate(names) if n in CONFOUNDERS]
print(f"confounder classes matched in YAMNet map: {len(conf_idx)}/{len(CONFOUNDERS)}")


def score(wav16: np.ndarray):
    _s, emb, _ = yam(tf.constant(wav16, tf.float32))
    ae = float(clf(emb.numpy().mean(0, keepdims=True), training=False).numpy()[0, 0])
    # confounder read on a peak-normalized pass (matches model.py infer_pcm16)
    peak = float(np.abs(wav16).max())
    norm = (wav16 / peak * 0.95).astype(np.float32) if peak > 1e-6 else wav16
    ns, _, _ = yam(tf.constant(norm, tf.float32))
    npc = ns.numpy().mean(0)
    ci = max(conf_idx, key=lambda i: npc[i])
    return ae, float(npc[ci]), names[ci]


def list_hour(sensor, dt):
    out = subprocess.run(["gsutil", "ls", f"{GCS}/{sensor}/{dt:%Y/%m/%d/%H}/"],
                         capture_output=True, text=True).stdout
    clips = []
    for line in out.splitlines():
        line = line.strip().rstrip("/")
        try:
            ts = datetime.strptime(line.split(".")[-1], "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        clips.append((ts, f"{line}/{line.split('/')[-1]}.wav"))
    return sorted(clips)


def native16(sensor, t, td):
    clips = list_hour(sensor, t)
    if not clips:
        return None
    s, u = min(clips, key=lambda c: abs((c[0] - t).total_seconds()))
    dst = td / Path(u).name
    if subprocess.run(["gsutil", "cp", u, str(dst)], capture_output=True).returncode != 0:
        return None
    sr, i16 = wavfile.read(str(dst))
    if sr != 8000 or not i16.size:
        return None
    return resample_poly(i16.astype(np.float32) / 32768.0, 2, 1).astype(np.float32)


fire_new = fire_veto = vetoed = 0
rows = []
for e in EVENTS:
    t = datetime.strptime(e["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as td:
        w = native16(e["sensor"], t, Path(td))
    if w is None:
        continue
    ae, cf, cn = score(w)
    fn = ae >= AE_THR
    fv = fn and cf < VETO_THR
    fire_new += fn
    fire_veto += fv
    if fn and not fv:
        vetoed += 1
    rows.append({**e, "ae_new": round(ae, 3), "conf": round(cf, 3), "conf_cls": cn,
                 "fire_new": fn, "fire_final": fv})

n = len(rows)
print(f"\n=== {n} last-hour detections (old model fired on ALL) ===")
print(f"  #1 new model @0.70:            fires {fire_new}/{n}")
print(f"  #1+#2 new model + veto:        fires {fire_veto}/{n}   (veto removed {vetoed} more)")
print(f"\nsuppressed-by-veto clips (new fired, veto killed):")
for r in rows:
    if r["fire_new"] and not r["fire_final"]:
        print(f"  {r['sensor']} {r['ts'][11:19]}  AE={r['ae_new']}  conf={r['conf']} ({r['conf_cls']})")
(ROOT / "false-detects" / "lasthour_veto.json").write_text(json.dumps(rows, indent=2))

# --- control: real USAFA drones must NOT be vetoed ---
print("\n=== control: real USAFA drones (veto must NOT suppress) ===")
import soundfile as sf
for f in sorted((ROOT / "usafa-dfec-dataset-16k").glob("*.wav")):
    if f.name.startswith(("box_fan", "tow_planes")):
        continue
    try:
        a, sr = sf.read(str(f), dtype="float32", always_2d=False)
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != 16000:
            from math import gcd
            g = gcd(int(sr), 16000)
            a = resample_poly(a, 16000 // g, int(sr) // g).astype(np.float32)
        ae, cf, cn = score(a.astype(np.float32))
        verdict = "OK fires" if (ae >= AE_THR and cf < VETO_THR) else ("VETOED!" if ae >= AE_THR else "low AE (n/a)")
        print(f"  {f.name:40} AE={ae:.2f} conf={cf:.2f} ({cn})  -> {verdict}")
    except Exception as ex:  # noqa: BLE001
        print(f"  {f.name:40} ERROR {type(ex).__name__}: {ex}")
