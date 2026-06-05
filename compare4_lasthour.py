"""
Run all 4 detectors on the last-hour live-pull Shaw detections and merge.

  1. Angels Envy (NEW hard-negative model) -- per-clip mean-pool drone_score
     (== production semantics); "fires" at the retuned threshold 0.70.
  2. Kilo Lani v9 -- native 8 kHz, mean over windows, fires >= 0.35.
  3. GANON template matcher (via /tmp/ganon-venv, EVENTS_JSON).
  4. AST AudioSet classifier (via /tmp/ast-venv, EVENTS_JSON) -- names the sound.

Writes false-detects/lasthour_4det.json and prints a table + summary.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import tensorflow as tf

from drone_detector import DroneDetector

ROOT = Path(__file__).parent
GCS = "gs://aftac-argos-dataflow-unzipped/ensco/SH"
EVENTS = ROOT / "false-detects" / "lasthour_events.json"
KILO = Path("/tmp/kilo/home/anthony_christe_ctr/kilo_lani_detector/Kilo_Lani_v9_APR202026.keras")
AE_THR, KILO_THR = 0.70, 0.35


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


def nearest_clip(sensor, t, td):
    clips = list_hour(sensor, t)
    cand = [c for c in clips if c[0] <= t + tf_seconds(2)] or clips
    if not cand:
        return None
    s, u = min(cand, key=lambda c: abs((c[0] - t).total_seconds()))
    dst = td / Path(u).name
    if subprocess.run(["gsutil", "cp", u, str(dst)], capture_output=True).returncode != 0:
        return None
    sr, i16 = wavfile.read(str(dst))
    return (sr, i16) if sr == 8000 and i16.size else None


def tf_seconds(s):
    from datetime import timedelta
    return timedelta(seconds=s)


def kilo_score(model, x):
    wf = tf.reshape(tf.convert_to_tensor(x, tf.float32), [-1])
    spec = tf.square(tf.abs(tf.transpose(tf.signal.stft(wf, fft_length=200, frame_length=200, frame_step=80))))
    spec = tf.math.log(spec + 1e-12)[1:65, :]
    spec = (spec - (-14.0)) / 8.5
    fr = tf.signal.frame(spec, frame_length=96, frame_step=48)
    if fr.shape[1] is None or fr.shape[1] == 0:
        return 0.0
    xx = tf.transpose(tf.reshape(fr, (64, fr.shape[1], 96, 1)), [1, 2, 0, 3])
    return float(tf.reduce_mean(model(xx, training=False)[:, 1]))


def main():
    events = json.loads(EVENTS.read_text())
    print(f"=== 4-detector comparison on {len(events)} last-hour Shaw detections ===", flush=True)

    # --- detectors 3 & 4 in the background (separate venvs) ---
    env = {**os.environ, "EVENTS_JSON": str(EVENTS), "TFHUB_CACHE_DIR": str(Path.home() / ".cache/tfhub")}
    g_out = ROOT / "false-detects" / "lasthour_ganon.json"
    a_out = ROOT / "false-detects" / "lasthour_ast.json"
    print("launching GANON + AST ...", flush=True)
    g_proc = subprocess.Popen(["/tmp/ganon-venv/bin/python", "run_ganon_batch.py"],
                              cwd="/Users/josephwehlen/dev/argos-ganon",
                              env={**env, "OUT_JSON": str(g_out)},
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    a_proc = subprocess.Popen(["/tmp/ast-venv/bin/python", "run_ast.py"],
                              cwd=str(ROOT), env={**env, "OUT_JSON": str(a_out)},
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # --- detectors 1 & 2 inline (new model + Kilo) ---
    det = DroneDetector(model_path=str(ROOT / "models/drone_classifier_binary.keras"), threshold=AE_THR)
    kilo = tf.keras.models.load_model(KILO, compile=False)
    rows = []
    for e in events:
        t = datetime.strptime(e["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            got = nearest_clip(e["sensor"], t, Path(td))
        if got is None:
            rows.append({**e, "ae_new": None, "kilo": None}); continue
        sr, i16 = got
        x = i16.astype(np.float32) / 32768.0
        ae = float(det.detect(x, sr)["confidence"])
        k = kilo_score(kilo, x)
        rows.append({**e, "ae_new": round(ae, 3), "kilo": round(k, 3)})
        print(f"  {e['sensor']} {e['ts'][11:19]}  old_peak={e['old_peak']}  "
              f"AEnew={ae:.2f}{'*' if ae>=AE_THR else ''}  Kilo={k:.2f}{'*' if k>=KILO_THR else ''}", flush=True)

    print("\nwaiting for GANON + AST ...", flush=True)
    g_proc.wait(); a_proc.wait()
    ganon = {(r["sensor"], r["ts"]): r for r in (json.loads(g_out.read_text()) if g_out.exists() else [])}
    ast = {(r["sensor"], r["ts"]): r for r in (json.loads(a_out.read_text()) if a_out.exists() else [])}

    for r in rows:
        key = (r["sensor"], r["ts"])
        gr = ganon.get(key, {})
        r["ganon_fire"] = gr.get("ganon_fire")
        ar = ast.get(key, {})
        r["ast_top"] = ar.get("top", [])[:3]
    (ROOT / "false-detects" / "lasthour_4det.json").write_text(json.dumps(rows, indent=2))

    # --- summary ---
    n = len(rows)
    scored = [r for r in rows if r["ae_new"] is not None]
    ae_fire = sum(r["ae_new"] >= AE_THR for r in scored)
    kilo_fire = sum(r["kilo"] >= KILO_THR for r in scored)
    ganon_fire = sum(bool(r.get("ganon_fire")) for r in rows)
    all4 = sum(r["ae_new"] is not None and r["ae_new"] >= AE_THR and r["kilo"] >= KILO_THR and r.get("ganon_fire")
               for r in scored)
    print(f"\n=== SUMMARY ({n} detections; old model fired on all) ===")
    print(f"  Angels Envy NEW @0.70 would fire:  {ae_fire}/{n}  (suppresses {n-ae_fire})")
    print(f"  Kilo @0.35 fires:                  {kilo_fire}/{n}")
    print(f"  GANON fires:                       {ganon_fire}/{n}")
    print(f"  ALL of AE-new+Kilo+GANON fire:     {all4}/{n}")
    from collections import Counter
    asttop = Counter(r["ast_top"][0]["cls"] for r in rows if r.get("ast_top"))
    print(f"  AST top-class tally: {dict(asttop.most_common(8))}")


if __name__ == "__main__":
    raise SystemExit(main())
