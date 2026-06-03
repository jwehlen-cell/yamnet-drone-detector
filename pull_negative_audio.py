"""
Pull *negative* (no-drone) audio from QST: dead-of-night windows for the same
sensors/days as the drone detections. The SH sensors are in Sumter, SC (US
Eastern); 1 am local in April = 05:00 UTC, when a drone is extremely unlikely.

Output mirrors the positive pull: 16 kHz mono WAVs under
``detection-audio-negatives/<date>/<sensor>_<HHMMSS-HHMMSS>.wav``. These are
intended as no_drone training negatives to rebalance the QST-positive-heavy
field set. Reuses the QST client + resample helpers from pull_detection_audio.

Usage:
    python pull_negative_audio.py                  # 05:00Z, 180 s, all sensors/days
    python pull_negative_audio.py --start 05:00:00 --seconds 180
    python pull_negative_audio.py --force
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pull_detection_audio import (
    CHANNEL,
    REPO,
    get_token,
    iso_z,
    load_env,
    resample_inplace,
    resolve_source_id,
    write_wav,
)

OUT_DIR = REPO / "detection-audio-negatives"

# Union of sensors across the three detection CSVs, and the same days.
SENSORS = ["SH003", "SH004", "SH007", "SH008", "SH011", "SH013", "SH015", "SH016"]
DAYS = ["2026-04-09", "2026-04-14", "2026-04-15"]


def _availability(env, token, source_id, sensor, start, end) -> bool:
    import json

    q = urllib.parse.urlencode({
        "source_id": source_id, "station": sensor, "channel": CHANNEL,
        "start": iso_z(start), "end": iso_z(end),
    })
    req = urllib.request.Request(f"{env['QST_API_BASE']}/api/analyze/availability?{q}",
                                 headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return bool(json.load(r).get("has_data"))
    except urllib.error.HTTPError:
        return False


def _fetch(env, token, source_id, sensor, start, end) -> tuple[bytes, int]:
    q = urllib.parse.urlencode({
        "station": sensor, "channel": CHANNEL, "source_id": source_id,
        "start_time": iso_z(start), "end_time": iso_z(end),
    })
    req = urllib.request.Request(f"{env['QST_API_BASE']}/api/stream/audio?{q}",
                                 headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read(), int(r.headers.get("x-sample-rate", "8000"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Pull QST night-time negatives.")
    ap.add_argument("--start", default="05:00:00", help="Start time of day, UTC (HH:MM:SS).")
    ap.add_argument("--seconds", type=int, default=180, help="Window length per sensor/day.")
    ap.add_argument("--target-sr", type=int, default=16000, help="Resample rate; 0 = native.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    env = load_env()
    token = get_token(env)
    source_id = resolve_source_id(env, token)
    print(f"Authenticated. source_id={source_id}  start={args.start}Z  window={args.seconds}s")
    OUT_DIR.mkdir(exist_ok=True)

    ok = no_data = fail = skip = 0
    for day in DAYS:
        (OUT_DIR / day).mkdir(exist_ok=True)
        for sensor in SENSORS:
            start = datetime.strptime(f"{day} {args.start}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            end = start + timedelta(seconds=args.seconds)
            out = OUT_DIR / day / f"{sensor}_{start.strftime('%H%M%S')}-{end.strftime('%H%M%S')}.wav"
            if out.exists() and not args.force:
                skip += 1
                continue
            if not _availability(env, token, source_id, sensor, start, end):
                print(f"  --   {day} {sensor}: no data at {args.start}Z")
                no_data += 1
                continue
            try:
                body, sr = _fetch(env, token, source_id, sensor, start, end)
                if not body:
                    no_data += 1
                    continue
                write_wav(out, body, sr)
                if args.target_sr and args.target_sr != sr:
                    resample_inplace(out, args.target_sr)
                ok += 1
                print(f"  ok   {out.relative_to(REPO)}  ({len(body)/2/sr:.1f}s @ {sr}Hz -> {args.target_sr or sr})")
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {day} {sensor}: {type(e).__name__}: {e}")
                fail += 1

    print(f"\nDone: {ok} pulled, {no_data} no-data, {fail} failed, {skip} skipped.")
    print(f"Output: {OUT_DIR.relative_to(REPO)}/<date>/")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
