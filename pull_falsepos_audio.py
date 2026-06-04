"""
Persist the Kilo-verified false-positive clips (from false-detects/verdicts.json)
as 16 kHz WAVs under detection-audio-falsepos/<date>/, so embed_detection_sets.py
can fold them into the no_drone class as hard negatives.

Re-pulls the same QST-API window the false_detect_pipeline scored, for every row
with verified_negative == true.
"""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pull_detection_audio as qst

ROOT = Path(__file__).parent
VERDICTS = ROOT / "false-detects" / "verdicts.json"
OUT_DIR = ROOT / "detection-audio-falsepos"
WIN_BEFORE, WIN_AFTER = 6.0, 2.0  # match false_detect_pipeline window


def main() -> int:
    rows = [r for r in json.loads(VERDICTS.read_text()) if r.get("verified_negative")]
    print(f"{len(rows)} verified-negative clips to pull.")
    env = qst.load_env()
    token = qst.get_token(env)
    source_id = qst.resolve_source_id(env, token)

    ok = fail = 0
    for r in rows:
        station = r["station"]
        t = datetime.strptime(r["utc"], "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=timezone.utc)
        det = qst.Detection(station, t - timedelta(seconds=WIN_BEFORE),
                            t + timedelta(seconds=WIN_AFTER), "", "verdicts", 0)
        day_dir = OUT_DIR / t.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        out = day_dir / det.out_name
        try:
            body, sr = qst.fetch_clip(env, token, source_id, det, pad=0.0)
            if not body:
                print(f"  empty: {out.name}"); fail += 1; continue
            qst.write_wav(out, body, sr)
            if sr != 16000:
                qst.resample_inplace(out, 16000)
            ok += 1
            print(f"  ok   {out.relative_to(ROOT)}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {out.name}: {type(e).__name__}: {e}"); fail += 1

    print(f"\n{ok} saved, {fail} failed -> {OUT_DIR}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
