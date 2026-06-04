"""
False-detect verification loop for the deployed Angels Envy (YAMNet binary) head.

Pulls the live-pull Shaw detections captured from the admin dashboard
(false-detects/snapshot.json), then for each detection:

  1. KILO GATE (native audio): downloads the native 8 kHz raw WAV(s) for the
     sensor/time from gs://aftac-argos-dataflow-unzipped (Kilo's calibrated
     domain) and scores them with the exact Kilo Lani v9 pipeline. A detection
     is a *verified negative* iff Kilo's max clip score <= KILO_THR (0.35) --
     i.e. the independent deployed detector agrees there is no drone.

  2. WHY ANGELS ENVY FIRES: pulls the QST-API clip (16 kHz, what the phone
     pipeline scored), re-scores it with the binary head, and reads YAMNet's
     top AudioSet classes for the window -- naming the confounder.

Output: a per-detection verdict table + false-detects/verdicts.json.

This does not modify training data; relabeling/retrain is a separate step that
consumes verdicts.json (Kilo-verified negatives only).
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import tensorflow as tf

# Reuse the QST auth + fetch plumbing from the existing puller.
import pull_detection_audio as qst
from drone_detector import DroneDetector

ROOT = Path(__file__).parent
SNAPSHOT = ROOT / "false-detects" / "snapshot.json"
OUT = ROOT / "false-detects" / "verdicts.json"

# --- Kilo Lani v9 (from /tmp/kilo .../config.toml + detector.py) -----------
KILO_DIR = Path("/tmp/kilo/home/anthony_christe_ctr/kilo_lani_detector")
KILO_MODEL = KILO_DIR / "Kilo_Lani_v9_APR202026.keras"
KILO_MU, KILO_SIGMA, KILO_THR = -14.0, 8.5, 0.35
GCS = "gs://aftac-argos-dataflow-unzipped/ensco/SH"

AE_THR = 0.7  # Angels Envy deployed threshold
WIN_BEFORE, WIN_AFTER = 6.0, 2.0  # seconds around the detection time to consider


# --------------------------------------------------------------------------
# Kilo scoring -- faithful reimplementation of detector.run_inference (mean).
# --------------------------------------------------------------------------
def kilo_score_clip(model, samples_f32: np.ndarray) -> float:
    wf = tf.reshape(tf.convert_to_tensor(samples_f32, dtype=tf.float32), [-1])
    spec = tf.square(tf.abs(tf.transpose(
        tf.signal.stft(signals=wf, fft_length=200, frame_length=200, frame_step=80))))
    spec = tf.math.log(spec + 1e-12)[1:65, :]
    spec = (spec - KILO_MU) / KILO_SIGMA
    framed = tf.signal.frame(signal=spec, frame_length=96, frame_step=48)
    if framed.shape[1] is None or framed.shape[1] == 0:
        return 0.0  # clip too short to form a 96-frame window
    x = tf.transpose(tf.reshape(framed, (64, framed.shape[1], 96, 1)), [1, 2, 0, 3])
    win = model(x, training=False)
    return float(tf.reduce_mean(win[:, 1]))


def gcs_list_hour(station: str, dt: datetime) -> list[tuple[datetime, str]]:
    """All native clip (start_dt, gs://...wav) for a station's UTC hour. Cached."""
    key = (station, dt.strftime("%Y%m%d%H"))
    cache = gcs_list_hour._cache
    if key in cache:
        return cache[key]
    prefix = f"{GCS}/{station}/{dt:%Y/%m/%d/%H}/"
    try:
        out = subprocess.run(["gsutil", "ls", prefix], capture_output=True,
                             text=True, timeout=60).stdout
    except Exception:
        out = ""
    clips = []
    for line in out.splitlines():
        line = line.strip().rstrip("/")
        tok = line.split(".")[-1]  # ...Scell.20260604_013433
        try:
            start = datetime.strptime(tok, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        clips.append((start, f"{line}/{line.split('/')[-1]}.wav"))
    clips.sort()
    cache[key] = clips
    return clips
gcs_list_hour._cache = {}


def kilo_gate(model, station: str, t: datetime) -> dict:
    """Score native clips overlapping [t-WIN_BEFORE, t+WIN_AFTER]; return max."""
    lo, hi = t - timedelta(seconds=WIN_BEFORE), t + timedelta(seconds=WIN_AFTER)
    # A clip starting at s covers ~[s, s+4]; include it if it overlaps the window.
    hour_clips = gcs_list_hour(station, t)
    # also pull the previous hour's tail if the window crosses the boundary
    if lo.hour != t.hour:
        hour_clips = gcs_list_hour(station, lo) + hour_clips
    picked = [(s, u) for (s, u) in hour_clips if s + timedelta(seconds=4) >= lo and s <= hi]
    scores = []
    with tempfile.TemporaryDirectory() as td:
        for s, uri in picked:
            dst = Path(td) / Path(uri).name
            r = subprocess.run(["gsutil", "cp", uri, str(dst)],
                               capture_output=True, text=True, timeout=60)
            if r.returncode != 0 or not dst.exists():
                continue
            sr, i16 = wavfile.read(str(dst))
            if sr != 8000:
                continue
            scores.append((s.strftime("%H%M%S"),
                           kilo_score_clip(model, i16.astype(np.float32) / 32768.0)))
    if not scores:
        return {"n_clips": 0, "max": None, "fires": None, "per_clip": []}
    mx = max(v for _, v in scores)
    return {"n_clips": len(scores), "max": mx, "fires": bool(mx > KILO_THR),
            "per_clip": [{"t": k, "score": round(v, 4)} for k, v in scores]}


# --------------------------------------------------------------------------
# Angels Envy "why": top YAMNet AudioSet classes for the QST clip.
# --------------------------------------------------------------------------
def load_yamnet_class_names(detector: DroneDetector) -> list[str]:
    path = detector.yamnet.class_map_path().numpy().decode()
    names = []
    with open(path) as f:
        for row in csv.DictReader(f):
            names.append(row["display_name"])
    return names


def yamnet_top_classes(detector, names, audio16k: np.ndarray, k: int = 5):
    wav = tf.constant(audio16k.astype(np.float32), dtype=tf.float32)
    scores, _emb, _spec = detector.yamnet(wav)
    mean = scores.numpy().mean(axis=0)
    top = mean.argsort()[::-1][:k]
    return [{"class": names[i], "score": round(float(mean[i]), 3)} for i in top]


def main() -> int:
    dets = json.loads(SNAPSHOT.read_text())
    print(f"{len(dets)} live-pull Shaw detections in snapshot.\n")

    print("Loading Angels Envy (YAMNet binary head)...", flush=True)
    detector = DroneDetector(model_path=str(ROOT / "models/drone_classifier_binary.keras"),
                             threshold=AE_THR)
    class_names = load_yamnet_class_names(detector)

    print(f"Loading Kilo Lani v9 from {KILO_MODEL}...", flush=True)
    kilo = tf.keras.models.load_model(KILO_MODEL, compile=False)

    print("Authenticating to QST...", flush=True)
    env = qst.load_env()
    token = qst.get_token(env)
    source_id = qst.resolve_source_id(env, token)

    rows = []
    for i, d in enumerate(sorted(dets, key=lambda x: x["last_frame_timestamp_ms"]), 1):
        station = d["device_id"].split("-")[-1]  # ARGOS-SHAW-SH006 -> SH006
        t = datetime.fromtimestamp(d["last_frame_timestamp_ms"] / 1000, tz=timezone.utc)

        # --- Kilo gate on native audio ---
        kg = kilo_gate(kilo, station, t)

        # --- pull QST clip, re-score with AE, read YAMNet "why" ---
        det_obj = qst.Detection(station, t - timedelta(seconds=WIN_BEFORE),
                                t + timedelta(seconds=WIN_AFTER), "", "snapshot", i)
        ae_conf, why = None, []
        try:
            body, sr = qst.fetch_clip(env, token, source_id, det_obj, pad=0.0)
            if body:
                if len(body) % 2:
                    body = body[:-1]
                i16 = np.frombuffer(body, dtype=np.int16)
                f32 = i16.astype(np.float32) / 32768.0
                res = detector.detect(f32, sr)
                ae_conf = round(res["confidence"], 4)
                a16 = detector._to_yamnet_input(f32, sr).numpy()
                why = yamnet_top_classes(detector, class_names, a16)
        except Exception as e:  # noqa: BLE001
            why = [{"class": f"<pull error: {type(e).__name__}>", "score": 0.0}]

        verified_neg = kg["fires"] is False  # Kilo present AND max <= thr
        rows.append({
            "detection_id": d["detection_id"], "station": station,
            "utc": t.strftime("%Y-%m-%d %H:%M:%SZ"),
            "ae_peak_deployed": round(d["peak_score"], 3),
            "ae_avg_deployed": round(d["average_score"], 3),
            "ae_rescore_window": ae_conf,
            "kilo_max": None if kg["max"] is None else round(kg["max"], 4),
            "kilo_n_clips": kg["n_clips"], "kilo_fires": kg["fires"],
            "verified_negative": verified_neg,
            "why_top_classes": why,
        })
        flag = "VERIFIED-NEG" if verified_neg else ("kilo:DRONE" if kg["fires"] else "no-native")
        topcls = ", ".join(f"{w['class']}({w['score']})" for w in why[:3])
        print(f"[{i:2}/{len(dets)}] {station} {t:%H:%M:%S}  "
              f"AE peak={d['peak_score']:.2f} re={ae_conf}  "
              f"Kilo max={kg['max'] if kg['max'] is None else round(kg['max'],3)} "
              f"({kg['n_clips']}clip)  -> {flag}\n            why: {topcls}", flush=True)

    OUT.write_text(json.dumps(rows, indent=2))
    vn = sum(r["verified_negative"] for r in rows)
    kd = sum(r["kilo_fires"] is True for r in rows)
    nn = sum(r["kilo_fires"] is None for r in rows)
    print(f"\n=== {vn} verified negatives | {kd} Kilo-agrees-drone | "
          f"{nn} no native audio ===\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
