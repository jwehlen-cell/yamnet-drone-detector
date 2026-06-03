"""
Pull drone-detection audio clips from the QST API.

Reads the detection-metadata CSVs (Sensor, StartTime(Z), EndTime(Z), Detection;
the date lives in the filename), authenticates to QST via Keycloak
client-credentials, and downloads the audio for each detection window as a WAV.

QST audio is served as raw PCM s16le, mono, at the rate reported in the
`x-sample-rate` response header (8 kHz for these sensors). The audio for a
window may run marginally longer than the exact range because QST returns whole
stored clips that overlap the window.

Clips are resampled to 16 kHz (YAMNet's expected input) by default via ffmpeg;
pass --target-sr 0 to keep the native 8 kHz instead.

Credentials and host come from `.env` (see .env.example). Nothing is hardcoded.

Usage:
    python pull_detection_audio.py                 # pull everything, skip existing
    python pull_detection_audio.py --force         # re-download
    python pull_detection_audio.py --only 04152026 # only files matching a token
    python pull_detection_audio.py --workers 6
    python pull_detection_audio.py --pad 1.0       # add N seconds either side
    python pull_detection_audio.py --target-sr 0   # keep native 8 kHz (no resample)
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).parent
META_DIR = REPO / "detection-metadata"
OUT_DIR = REPO / "detection-audio"

CHANNEL = "Scell"  # all bq_sensors stations expose a single "Scell" channel


def load_env(path: Path = REPO / ".env") -> dict:
    if not path.exists():
        sys.exit(f"Missing {path}. Copy .env.example to .env and fill it in.")
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    required = ["QST_KC_URL", "QST_REALM", "QST_CLIENT_ID", "QST_CLIENT_SECRET", "QST_API_BASE"]
    missing = [k for k in required if not env.get(k)]
    if missing:
        sys.exit(f"Missing {missing} in {path}.")
    return env


def get_token(env: dict) -> str:
    url = f"{env['QST_KC_URL']}/realms/{env['QST_REALM']}/protocol/openid-connect/token"
    data = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": env["QST_CLIENT_ID"],
            "client_secret": env["QST_CLIENT_SECRET"],
        }
    ).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=30) as r:
        import json

        return json.load(r)["access_token"]


def resolve_source_id(env: dict, token: str) -> str:
    """Pick the BigQuery source (the one carrying the SH sensors)."""
    import json

    url = f"{env['QST_API_BASE']}/api/inventory/sources"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        sources = json.load(r)
    for s in sources:
        if s.get("source_type") == "bigquery":
            return s["id"]
    sys.exit(f"No BigQuery source found among: {[s.get('key') for s in sources]}")


# Filenames look like "04152026 Detections - Sheet1.csv" or
# "Detections 04092026 - Sheet1.csv": an 8-digit MMDDYYYY token.
DATE_RE = re.compile(r"(\d{8})")


def date_from_filename(name: str) -> str:
    m = DATE_RE.search(name)
    if not m:
        raise ValueError(f"No MMDDYYYY date token in filename: {name}")
    return datetime.strptime(m.group(1), "%m%d%Y").strftime("%Y-%m-%d")


def parse_time(date_iso: str, hms: str) -> datetime:
    """Combine 'YYYY-MM-DD' + 'HH:MM:SS(.fff)' (UTC) into an aware datetime."""
    hms = hms.strip()
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in hms else "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(f"{date_iso} {hms}", fmt).replace(tzinfo=timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class Detection:
    __slots__ = ("sensor", "start", "end", "label", "src_file", "row")

    def __init__(self, sensor, start, end, label, src_file, row):
        self.sensor = sensor
        self.start = start
        self.end = end
        self.label = label
        self.src_file = src_file
        self.row = row

    @property
    def out_name(self) -> str:
        s = self.start.strftime("%H%M%S")
        e = self.end.strftime("%H%M%S")
        return f"{self.sensor}_{s}-{e}.wav"


def read_detections(only: str | None) -> list[Detection]:
    dets: list[Detection] = []
    files = sorted(META_DIR.glob("*.csv"))
    if only:
        files = [f for f in files if only in f.name]
    if not files:
        sys.exit(f"No matching CSVs in {META_DIR} (only={only!r}).")
    for f in files:
        date_iso = date_from_filename(f.name)
        with f.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader, start=2):  # row 1 = header
                sensor = (row.get("Sensor") or "").strip()
                st = (row.get("StartTime(Z)") or "").strip()
                et = (row.get("EndTime(Z)") or "").strip()
                if not (sensor and st and et):
                    continue
                start = parse_time(date_iso, st)
                end = parse_time(date_iso, et)
                if end <= start:
                    print(f"  WARN {f.name} row {i}: end <= start ({st} -> {et}); "
                          f"swapping to recover a valid window.")
                    start, end = (end, start) if end < start else (start, start + timedelta(seconds=1))
                dets.append(
                    Detection(sensor, start, end, (row.get("Detection") or "").strip(), f.name, i)
                )
    return dets


def fetch_clip(env, token, source_id, det: Detection, pad: float) -> tuple[bytes, int]:
    start = det.start - timedelta(seconds=pad)
    end = det.end + timedelta(seconds=pad)
    q = urllib.parse.urlencode(
        {
            "station": det.sensor,
            "channel": CHANNEL,
            "source_id": source_id,
            "start_time": iso_z(start),
            "end_time": iso_z(end),
        }
    )
    url = f"{env['QST_API_BASE']}/api/stream/audio?{q}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
        sr = int(r.headers.get("x-sample-rate", "8000"))
    return body, sr


def write_wav(path: Path, pcm: bytes, sr: int) -> None:
    if len(pcm) % 2:
        pcm = pcm[:-1]  # keep 16-bit alignment
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sr)
        w.writeframes(pcm)


def resample_inplace(path: Path, target_sr: int) -> None:
    """Resample a mono 16-bit WAV to target_sr in place using ffmpeg."""
    tmp = path.with_suffix(path.suffix + ".rs.tmp")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
         "-ar", str(target_sr), "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav", str(tmp)],
        check=True,
    )
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Pull QST detection audio clips.")
    ap.add_argument("--force", action="store_true", help="Re-download existing WAVs.")
    ap.add_argument("--only", help="Only process CSVs whose filename contains this token.")
    ap.add_argument("--workers", type=int, default=4, help="Concurrent downloads.")
    ap.add_argument("--pad", type=float, default=0.0, help="Seconds of context to add each side.")
    ap.add_argument("--target-sr", type=int, default=16000,
                    help="Resample to this rate (YAMNet wants 16000); 0 keeps native 8 kHz.")
    args = ap.parse_args()

    if args.target_sr and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found but --target-sr set. Install ffmpeg or pass --target-sr 0.")

    env = load_env()
    token = get_token(env)
    source_id = resolve_source_id(env, token)
    print(f"Authenticated. BigQuery source_id = {source_id}")

    dets = read_detections(args.only)
    print(f"{len(dets)} detections across {len({d.src_file for d in dets})} file(s).")

    OUT_DIR.mkdir(exist_ok=True)
    jobs = []
    for d in dets:
        day_dir = OUT_DIR / date_from_filename(d.src_file)
        day_dir.mkdir(exist_ok=True)
        out = day_dir / d.out_name
        if out.exists() and not args.force:
            continue
        jobs.append((d, out))

    skipped = len(dets) - len(jobs)
    if skipped:
        print(f"Skipping {skipped} already-downloaded (use --force to redo).")
    if not jobs:
        print("Nothing to do.")
        return 0

    ok = 0
    fail = 0
    empty = 0

    def work(job):
        d, out = job
        body, sr = fetch_clip(env, token, source_id, d, args.pad)
        if not body:
            return d, out, "empty", 0.0
        write_wav(out, body, sr)
        dur = len(body) / 2 / sr
        if args.target_sr and args.target_sr != sr:
            resample_inplace(out, args.target_sr)
        return d, out, "ok", dur

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, j): j for j in jobs}
        for fut in as_completed(futs):
            d, out = futs[fut]
            try:
                _, _, status, dur = fut.result()
            except urllib.error.HTTPError as e:
                status, dur = f"HTTP {e.code}", 0.0
                if e.code == 404:
                    status = "no-data"
            except Exception as e:  # noqa: BLE001
                status = f"ERR {type(e).__name__}: {e}"
                dur = 0.0
            if status == "ok":
                ok += 1
                print(f"  ok   {out.relative_to(REPO)}  ({dur:.1f}s)")
            elif status in ("empty", "no-data"):
                empty += 1
                print(f"  --   {d.sensor} {iso_z(d.start)}  ({status})")
            else:
                fail += 1
                print(f"  FAIL {d.sensor} {iso_z(d.start)}  {status}")

    print(f"\nDone: {ok} downloaded, {empty} no-data, {fail} failed, {skipped} skipped.")
    print(f"Output: {OUT_DIR.relative_to(REPO)}/<date>/")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
