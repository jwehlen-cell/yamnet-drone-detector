"""
Download the ERAU YAMNet drone-embedding dataset.

Dataset:
  Embry-Riddle Aeronautical University
  "YAMNet Embeddings for Drone Detection"
  DOI: 10.17632/5dmcszvym4.3
  License: CC BY 4.0
  Mirror: https://datacommons.erau.edu/datasets/5dmcszvym4/3

This script attempts a few known direct-download URLs (Mendeley's S3 zip cache
typically works). If they all fail, it prints clear manual instructions and
exits non-zero so the caller can fall back.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"

# Known direct-download candidates for Mendeley dataset 5dmcszvym4, version 3.
CANDIDATE_URLS = [
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/5dmcszvym4-3.zip",
    "https://data.mendeley.com/public-files/datasets/5dmcszvym4/files-archive/3",
]


def _stream_download(url: str, dest: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
            if r.status_code != 200:
                print(f"  HTTP {r.status_code} from {url}")
                return False
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=dest.name
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        return dest.exists() and dest.stat().st_size > 0
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def _print_manual_instructions() -> None:
    print()
    print("=" * 72)
    print("Automatic download failed.")
    print("Please download the dataset manually:")
    print()
    print("  1. Visit:  https://data.mendeley.com/datasets/5dmcszvym4/3")
    print("     (or mirror:  https://datacommons.erau.edu/datasets/5dmcszvym4/3)")
    print("  2. Download the dataset zip.")
    print("  3. Extract it so that ./data/ contains:")
    print("       data/drone/DJI_Matrice_M100/...")
    print("       data/drone/Mavic_3/...")
    print("       data/drone/Mavic_Mini_2/...")
    print("       data/no_drone/...")
    print()
    print("  Cite: Embry-Riddle Aeronautical University,")
    print("        DOI 10.17632/5dmcszvym4.3, CC BY 4.0")
    print("=" * 72)


def _flatten_into_data_dir(extract_root: Path) -> None:
    """
    Some Mendeley archives wrap files in an extra top-level directory.
    Detect that case and move drone/ + no_drone/ up into DATA_DIR.
    """
    # If drone/ and no_drone/ are already at DATA_DIR, nothing to do.
    if (DATA_DIR / "drone").is_dir() and (DATA_DIR / "no_drone").is_dir():
        return
    # Look one level down.
    for child in extract_root.iterdir():
        if child.is_dir() and (child / "drone").is_dir() and (child / "no_drone").is_dir():
            for sub in ("drone", "no_drone"):
                shutil.move(str(child / sub), str(DATA_DIR / sub))
            return


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the ERAU YAMNet drone dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if ./data/drone and ./data/no_drone already exist.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    if (
        (DATA_DIR / "drone").is_dir()
        and (DATA_DIR / "no_drone").is_dir()
        and not args.force
    ):
        print(f"Dataset already present at {DATA_DIR}/. Use --force to re-download.")
        return 0

    zip_path = DATA_DIR / "_erau_drone_yamnet.zip"

    for url in CANDIDATE_URLS:
        print(f"Trying {url}")
        if _stream_download(url, zip_path):
            print(f"Downloaded {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
            break
    else:
        _print_manual_instructions()
        return 1

    print(f"Extracting into {DATA_DIR}/ ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
    except zipfile.BadZipFile:
        print("Downloaded file is not a valid zip archive.")
        _print_manual_instructions()
        return 1

    _flatten_into_data_dir(DATA_DIR)

    if not ((DATA_DIR / "drone").is_dir() and (DATA_DIR / "no_drone").is_dir()):
        print("Extracted, but expected drone/ and no_drone/ folders are missing.")
        _print_manual_instructions()
        return 1

    try:
        zip_path.unlink()
    except OSError:
        pass

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
