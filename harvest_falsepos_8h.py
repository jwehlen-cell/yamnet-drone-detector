"""
Harvest Kilo-verified false positives from the last N hours of native Shaw audio.

For every native 4 s clip (gs://aftac-argos-dataflow-unzipped/ensco/SH/<sensor>/
.../*.wav) in the window, score it with BOTH detectors in their production form:

  * Angels Envy (YAMNet binary head): per-frame score over the clip -> peak.
    "Would the deployment fire?" == peak >= AE_THR (0.7).
  * Kilo Lani v9: exact run_inference (mean over windows). "Kilo sees a drone?"
    == score >= KILO_THR (0.35).

Verdict per clip:
  peak>=0.7 AND kilo<0.35  -> FALSE POSITIVE  (save 16 kHz WAV as hard negative)
  peak>=0.7 AND kilo>=0.35 -> BOTH-FIRE       (candidate real drone; flag, DON'T train)
  peak<0.7                 -> not a detection  (logged, skipped)

Only ADDS negatives. Output WAVs -> detection-audio-falsepos/<date>/ (embedded by
embed_detection_sets.py). Full per-clip log -> false-detects/harvest_8h.jsonl.
Checkpoint per (sensor,hour) in false-detects/harvest_8h.done so it is resumable.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf
import tensorflow as tf

from drone_detector import DroneDetector

ROOT = Path(__file__).parent
FP_DIR = ROOT / "detection-audio-falsepos"
LOG = ROOT / "false-detects" / "harvest_8h.jsonl"
DONE = ROOT / "false-detects" / "harvest_8h.done"

GCS = "gs://aftac-argos-dataflow-unzipped/ensco/SH"
SENSORS = ["SH002", "SH006", "SH009", "SH010", "SH013", "SH016"]
HOURS_BACK = 8
AE_THR, KILO_THR = 0.7, 0.35
KILO_MODEL = Path("/tmp/kilo/home/anthony_christe_ctr/kilo_lani_detector/Kilo_Lani_v9_APR202026.keras")
KILO_MU, KILO_SIGMA = -14.0, 8.5


def kilo_score(model, x_f32: np.ndarray) -> float:
    wf = tf.reshape(tf.convert_to_tensor(x_f32, tf.float32), [-1])
    spec = tf.square(tf.abs(tf.transpose(
        tf.signal.stft(wf, fft_length=200, frame_length=200, frame_step=80))))
    spec = tf.math.log(spec + 1e-12)[1:65, :]
    spec = (spec - KILO_MU) / KILO_SIGMA
    fr = tf.signal.frame(spec, frame_length=96, frame_step=48)
    if fr.shape[1] is None or fr.shape[1] == 0:
        return 0.0
    xx = tf.transpose(tf.reshape(fr, (64, fr.shape[1], 96, 1)), [1, 2, 0, 3])
    return float(tf.reduce_mean(model(xx, training=False)[:, 1]))


def ae_peak_avg(det: DroneDetector, x16: np.ndarray) -> tuple[float, float]:
    """Per-frame YAMNet embedding -> dense head -> peak & mean over frames."""
    _scores, emb, _spec = det.yamnet(tf.constant(x16, tf.float32))
    emb = emb.numpy()
    if emb.shape[0] == 0:
        return 0.0, 0.0
    p = det.classifier.predict(emb.astype(np.float32), verbose=0).ravel()
    return float(p.max()), float(p.mean())


def to16k(x_i16: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly
    x = x_i16.astype(np.float32) / 32768.0
    return resample_poly(x, 2, 1).astype(np.float32)  # 8k -> 16k


def list_clip_wavs(sensor: str, dt_hour: datetime) -> list[tuple[datetime, str]]:
    prefix = f"{GCS}/{sensor}/{dt_hour:%Y/%m/%d/%H}/"
    out = subprocess.run(["gsutil", "ls", prefix], capture_output=True, text=True).stdout
    clips = []
    for line in out.splitlines():
        line = line.strip().rstrip("/")
        tok = line.split(".")[-1]
        try:
            ts = datetime.strptime(tok, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        clips.append((ts, f"{line}/{line.split('/')[-1]}.wav"))
    return sorted(clips)


def main() -> int:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=HOURS_BACK)
    print(f"window: {cutoff:%Y-%m-%d %H:%MZ} .. {now:%H:%MZ}  sensors={SENSORS}", flush=True)

    det = DroneDetector(model_path=str(ROOT / "models/drone_classifier_binary.keras"), threshold=AE_THR)
    kilo = tf.keras.models.load_model(KILO_MODEL, compile=False)
    FP_DIR.mkdir(exist_ok=True); LOG.parent.mkdir(exist_ok=True)
    done = set(DONE.read_text().splitlines()) if DONE.exists() else set()
    logf = LOG.open("a")

    hours = sorted({(cutoff + timedelta(hours=i)).replace(minute=0, second=0, microsecond=0)
                    for i in range(HOURS_BACK + 1)})
    n_fp = n_both = n_clip = 0
    for sensor in SENSORS:
        for h in hours:
            key = f"{sensor}/{h:%Y%m%d%H}"
            if key in done:
                continue
            with tempfile.TemporaryDirectory() as td:
                # cp -r the whole hour prefix (reliable; `cp -I` from piped stdin
                # silently drops all but ~2 URLs under -m). Then glob the wavs.
                subprocess.run(["gsutil", "-m", "cp", "-r", f"{GCS}/{sensor}/{h:%Y/%m/%d/%H}/", td],
                               capture_output=True, text=True)
                wavs = sorted(Path(td).rglob("*.wav"))
                for f in wavs:
                    tok = f.stem.split(".")[-1]  # 20260604_032215
                    try:
                        ts = datetime.strptime(tok, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
                    if not (cutoff <= ts <= now):
                        continue
                    try:
                        sr, i16 = wavfile.read(str(f))
                    except Exception:
                        continue
                    if sr != 8000 or i16.size == 0:
                        continue
                    n_clip += 1
                    k = kilo_score(kilo, i16.astype(np.float32) / 32768.0)
                    peak, avg = ae_peak_avg(det, to16k(i16))
                    fired = peak >= AE_THR
                    verdict = ("false_positive" if fired and k < KILO_THR else
                               "both_fire" if fired else "no_detection")
                    rec = {"sensor": sensor, "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                           "ae_peak": round(peak, 4), "ae_avg": round(avg, 4),
                           "kilo": round(k, 4), "verdict": verdict}
                    logf.write(json.dumps(rec) + "\n")
                    if verdict == "false_positive":
                        n_fp += 1
                        day = FP_DIR / ts.strftime("%Y-%m-%d"); day.mkdir(exist_ok=True)
                        sf.write(str(day / f"{sensor}_{ts:%H%M%S}.wav"), to16k(i16), 16000)
                    elif verdict == "both_fire":
                        n_both += 1
            logf.flush()
            done.add(key); DONE.write_text("\n".join(sorted(done)))
            print(f"  {key}: clips so far={n_clip}  FP={n_fp}  both_fire={n_both}", flush=True)

    logf.close()
    print(f"\n=== harvest done: {n_clip} clips scored | {n_fp} false positives saved | "
          f"{n_both} both-fire (candidate real drones, flagged) ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
