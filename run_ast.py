"""
AST (Audio Spectrogram Transformer) as an on-request 4th contender.

Unlike the 3 drone-vs-not detectors, AST is an AudioSet *classifier* (527
classes) -- it NAMES the sound. Run it on the ambiguous events and the confirmed
quadcopters and print the top classes. If it says 'Train'/'Railroad' on the
04:11 clip, that adjudicates the 3-detector "drone" consensus.

Model: MIT/ast-finetuned-audioset-10-10-0.4593 (public).
Run with: /tmp/ast-venv/bin/python run_ast.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly
import torch
from transformers import ASTForAudioClassification, AutoFeatureExtractor

GCS = "gs://aftac-argos-dataflow-unzipped/ensco/SH"
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
BEFORE, AFTER = 6, 4  # ~10 s window (AST max ~10.24 s)

import json as _json
import os as _os

if _os.environ.get("EVENTS_JSON"):
    # Generalized mode: classify an arbitrary event list, write top-class JSON.
    EVENTS = [(f"{e['sensor']} {e['ts'][11:19]}", e["sensor"], e["ts"])
              for e in _json.loads(Path(_os.environ["EVENTS_JSON"]).read_text())]
    OUT_JSON = _os.environ.get("OUT_JSON")
else:
    EVENTS = [
        ("04:11 TRAIN (your call)",  "SH002", "2026-06-04T04:11:38Z"),
        ("04:57 PLANE (your call)",  "SH006", "2026-06-04T04:57:36Z"),
        ("CONFIRMED QUAD #1",        "SH008", "2026-04-09T13:02:05Z"),
        ("CONFIRMED QUAD #2",        "SH003", "2026-04-14T17:51:15Z"),
        ("09:46 (2-of-3, GANON no)", "SH009", "2026-06-04T09:46:45Z"),
    ]
    OUT_JSON = None


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


def pull16k(sensor, t, td):
    lo, hi = t - timedelta(seconds=BEFORE), t + timedelta(seconds=AFTER)
    hours = sorted({lo.replace(minute=0, second=0, microsecond=0), t.replace(minute=0, second=0, microsecond=0)})
    clips = []
    for h in hours:
        clips += [c for c in list_hour(sensor, h) if lo <= c[0] <= hi]
    audio = []
    for s, u in sorted(set(clips)):
        dst = td / Path(u).name
        if subprocess.run(["gsutil", "cp", u, str(dst)], capture_output=True).returncode == 0:
            try:
                sr, i16 = wavfile.read(str(dst))
                if sr == 8000 and i16.size:
                    audio.append(i16.astype(np.float32) / 32768.0)
            except Exception:
                pass
    if not audio:
        return None
    a = np.concatenate(audio)
    return resample_poly(a, 2, 1).astype(np.float32)  # 8k -> 16k


def main():
    print(f"loading {MODEL_ID} ...", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = ASTForAudioClassification.from_pretrained(MODEL_ID).eval()
    id2label = model.config.id2label

    results = []
    for tag, sensor, ts in EVENTS:
        t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            wav = pull16k(sensor, t, Path(td))
        if wav is None:
            print(f"\n### {tag} [{sensor} {ts[11:19]}] -> NO NATIVE AUDIO")
            results.append({"sensor": sensor, "ts": ts, "top": []})
            continue
        inputs = fe(wav, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        probs = torch.sigmoid(logits)
        top = torch.topk(probs, 8)
        pairs = [(round(p, 3), id2label[idx]) for p, idx in zip(top.values.tolist(), top.indices.tolist())]
        print(f"\n### {tag} [{sensor} {ts[11:19]}, {len(wav)/16000:.0f}s]")
        for p, name in pairs:
            print(f"    {p:5.2f}  {name}")
        results.append({"sensor": sensor, "ts": ts, "top": [{"cls": n, "p": p} for p, n in pairs]})
    if OUT_JSON:
        Path(OUT_JSON).write_text(_json.dumps(results, indent=2))
        print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    sys.exit(main())
