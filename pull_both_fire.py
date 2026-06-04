"""
Stage the 35 both-fire candidate-real-drone clips for human listening.

For each clip in false-detects/both_fire_candidates.json (Angels Envy peak>=0.7
AND Kilo>=0.35 on the same native audio), pull the native 8 kHz WAV plus a few
seconds of context on each side, concatenate, and write a single listenable file
named with sensor/time/scores. Also writes an index.txt.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf

ROOT = Path(__file__).parent
OUT = ROOT / "both-fire-review"
GCS = "gs://aftac-argos-dataflow-unzipped/ensco/SH"
CONTEXT_S = 8.0   # seconds of context each side of the event


def list_hour(sensor: str, dt: datetime):
    out = subprocess.run(["gsutil", "ls", f"{GCS}/{sensor}/{dt:%Y/%m/%d/%H}/"],
                         capture_output=True, text=True).stdout
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
    rows = json.loads((ROOT / "false-detects" / "both_fire_candidates.json").read_text())
    OUT.mkdir(exist_ok=True)
    index = []
    hour_cache: dict = {}
    for r in sorted(rows, key=lambda r: (r["sensor"], r["ts"])):
        sensor = r["sensor"]
        t = datetime.strptime(r["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        key = (sensor, t.strftime("%Y%m%d%H"))
        if key not in hour_cache:
            hour_cache[key] = list_hour(sensor, t)
        lo, hi = t - timedelta(seconds=CONTEXT_S), t + timedelta(seconds=CONTEXT_S)
        picked = [(s, u) for s, u in hour_cache[key] if lo <= s <= hi]
        audio = []
        with tempfile.TemporaryDirectory() as td:
            for s, u in picked:
                dst = Path(td) / Path(u).name
                if subprocess.run(["gsutil", "cp", u, str(dst)],
                                  capture_output=True).returncode != 0:
                    continue
                try:
                    sr, i16 = wavfile.read(str(dst))
                except Exception:
                    continue
                if sr == 8000 and i16.size:
                    audio.append(i16.astype(np.int16))
        if not audio:
            print(f"  no audio: {sensor} {r['ts']}"); continue
        wav = np.concatenate(audio)
        name = f"{sensor}_{t:%H%M%S}_ae{r['ae_peak']:.2f}_kilo{r['kilo']:.2f}.wav"
        sf.write(str(OUT / name), wav, 8000)
        index.append(f"{name}\t{len(wav)/8000:.0f}s\tae_peak={r['ae_peak']}\tkilo={r['kilo']}")
        print(f"  staged {name}  ({len(audio)} clips, {len(wav)/8000:.0f}s)")
    (OUT / "index.txt").write_text("\n".join(index) + "\n")
    print(f"\n{len(index)} events staged in {OUT}/  (see index.txt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
