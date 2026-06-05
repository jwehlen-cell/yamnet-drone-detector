"""
Prototype option #4: temporal-persistence + multi-sensor gate on the alert stream.

A real drone is BOUNDED in time and TRANSITS the array (several sensors in a short
window). A fixed confounder (frog/vehicle/HVAC) fires near-continuously at ONE
sensor for a long time with no corroboration. This gate suppresses an alert when
its sensor is in a persistent-firing state AND no other sensor corroborates.

Per alert d (stream sorted by time):
  persistence  = # alerts from d.sensor in (d.ts - PERSIST_WINDOW, d.ts]
  corroboration= # DISTINCT other sensors with an alert within +/- CORROB_DT of d.ts
  SUPPRESS if persistence >= PERSIST_K and corroboration == 0   (stationary, alone)

Runs on (a) the raw old-model stream (41) and (b) the post-#1 stream (new model
@0.70). Pure stream logic — no audio/model needed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
ROWS = json.loads((ROOT / "false-detects" / "lasthour_veto.json").read_text())

PERSIST_WINDOW = 600   # seconds (10 min look-back for same-sensor persistence)
PERSIST_K = 4          # >= this many same-sensor alerts in the window = persistent
CORROB_DT = 60         # seconds: another sensor within +/- this = corroborated


def epoch(ts: str) -> float:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def gate(stream: list[dict], mode: str) -> list[dict]:
    """mode 'persist_only': suppress chronic sensors outright.
       mode 'persist_isolated': suppress chronic AND uncorroborated (lenient)."""
    s = sorted(stream, key=lambda r: r["_t"])
    out = []
    for d in s:
        persistence = sum(1 for o in s if o["sensor"] == d["sensor"]
                          and d["_t"] - PERSIST_WINDOW < o["_t"] <= d["_t"])
        corrob = len({o["sensor"] for o in s if o["sensor"] != d["sensor"]
                      and abs(o["_t"] - d["_t"]) <= CORROB_DT})
        chronic = persistence >= PERSIST_K
        suppress = chronic if mode == "persist_only" else (chronic and corrob == 0)
        out.append({**d, "persistence": persistence, "corrob": corrob, "alert": not suppress})
    return out


def summarize(name: str, stream: list[dict], mode: str) -> None:
    g = gate(stream, mode)
    kept = [r for r in g if r["alert"]]
    from collections import Counter
    by = Counter(r["sensor"] for r in g)
    kept_by = Counter(r["sensor"] for r in kept)
    print(f"\n=== {name} [{mode}]: {len(stream)} -> {len(kept)} "
          f"(suppressed {len(stream)-len(kept)}) ===")
    print("    " + " ".join(f"{s}:{by[s]}->{kept_by[s]}" for s in sorted(by)))
    print("  kept: " + ", ".join(f"{r['sensor']}@{r['ts'][11:16]}(p{r['persistence']})" for r in kept))


for r in ROWS:
    r["_t"] = epoch(r["ts"])
post1 = [r for r in ROWS if (r.get("ae_new") or 0) >= 0.70]

for mode in ("persist_isolated", "persist_only"):
    summarize("RAW (41)", ROWS, mode)
    summarize("POST-#1 (new@0.70)", post1, mode)

# Is there ANY genuine bounded multi-sensor transit this hour? (>=3 distinct
# sensors firing within a 90 s window = candidate moving source worth keeping.)
print("\n=== transit scan: windows with >=3 distinct sensors in 90s ===")
s = sorted(ROWS, key=lambda r: r["_t"])
found = False
for i, d in enumerate(s):
    near = {o["sensor"] for o in s if 0 <= o["_t"] - d["_t"] <= 90}
    if len(near) >= 3:
        print(f"  {d['ts'][11:19]}: sensors {sorted(near)}")
        found = True
if not found:
    print("  NONE — every cluster is 1-2 sensors; no drone-like array transit this hour.")

print("\nparams: PERSIST_WINDOW=%ds PERSIST_K=%d CORROB_DT=%ds" % (PERSIST_WINDOW, PERSIST_K, CORROB_DT))
print("chronic sensors (fired >= %d times in a 10-min window at some point): " % PERSIST_K
      + ", ".join(sorted({r["sensor"] for r in gate(ROWS, "persist_only") if not r["alert"]})))
