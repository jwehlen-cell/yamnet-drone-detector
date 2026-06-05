"""
Tune the temporal-persistence gate (option #4) on 8 h of alert stream
reconstructed from harvest_8h.jsonl (old-model per-clip scores, 6 sensors,
2026-06-04 03:02-11:02 UTC).

Steps:
  1. Reconstruct the realistic ALERT stream: a clip "fires" if ae_peak >= 0.7,
     then apply the production per-sensor 60 s suppression window -> alerts.
  2. Chronic-sensor gate with cooldown: a sensor is "chronic" once it has
     >= K alerts in the last W seconds; while chronic its alerts are muted;
     it leaves chronic after QUIET seconds with no fire. Sweep K / QUIET.
  3. Transit scan: windows with >= 3 distinct sensors within 90 s (the only
     thing that should override a chronic mute).
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
ROWS = [json.loads(l) for l in open(ROOT / "false-detects" / "harvest_8h.jsonl")]
FIRE_THR = 0.70
SUPPRESS = 60        # production per-device suppression window (s)
W = 600              # chronic look-back window (s)


def epoch(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


# --- 1. reconstruct alert stream (fire + 60s per-sensor suppression) ---
fires = sorted(([r["sensor"], epoch(r["ts"]), r["ts"]] for r in ROWS if r["ae_peak"] >= FIRE_THR),
               key=lambda x: x[1])
last_alert: dict[str, float] = {}
alerts = []
for sensor, t, ts in fires:
    if sensor not in last_alert or t - last_alert[sensor] >= SUPPRESS:
        alerts.append((sensor, t, ts))
        last_alert[sensor] = t
print(f"8h: {len(fires)} clip-fires -> {len(alerts)} alerts after 60s suppression "
      f"(~{len(alerts)/8:.0f}/hr)")
from collections import Counter
print("alerts by sensor:", dict(Counter(a[0] for a in alerts)))


# --- 2. chronic-sensor gate with cooldown ---
def run_gate(alerts, K, QUIET):
    recent: dict[str, deque] = defaultdict(deque)
    chronic: dict[str, bool] = defaultdict(bool)
    last_fire: dict[str, float] = {}
    kept = []
    for sensor, t, ts in alerts:
        # exit chronic if quiet long enough
        if chronic[sensor] and sensor in last_fire and t - last_fire[sensor] >= QUIET:
            chronic[sensor] = False
            recent[sensor].clear()
        dq = recent[sensor]
        dq.append(t)
        while dq and t - dq[0] > W:
            dq.popleft()
        if len(dq) >= K:
            chronic[sensor] = True
        last_fire[sensor] = t
        if not chronic[sensor]:
            kept.append((sensor, t, ts))
    return kept


print("\n=== chronic-gate sweep (alerts kept of %d) ===" % len(alerts))
print(f"{'K':>3} {'QUIET':>6} {'kept':>6} {'suppressed':>11} {'%cut':>6}")
for K in (3, 4, 5, 6):
    for QUIET in (300, 600):
        kept = run_gate(alerts, K, QUIET)
        print(f"{K:>3} {QUIET:>6} {len(kept):>6} {len(alerts)-len(kept):>11} "
              f"{100*(1-len(kept)/len(alerts)):>5.0f}%")

# detail at a recommended setting
kept = run_gate(alerts, 4, 600)
print(f"\nrecommended K=4 QUIET=600: {len(alerts)} -> {len(kept)} alerts")
print("  kept by sensor:", dict(Counter(a[0] for a in kept)))


# --- 3. transit scan over the alert stream ---
print("\n=== transit scan: >=3 distinct sensors within 90s (candidate real array transit) ===")
hits = 0
for i, (s0, t0, ts0) in enumerate(alerts):
    near = {a[0] for a in alerts if 0 <= a[1] - t0 <= 90}
    if len(near) >= 3:
        hits += 1
        if hits <= 8:
            print(f"  {ts0[11:19]}: {sorted(near)}")
print(f"  {hits} such windows in 8h" + (" (these would override the chronic mute)" if hits else ""))
