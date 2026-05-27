"""
CLI wrapper around DroneDetector for quick local testing.

    python detect_drone_cli.py --input audio.wav --threshold 0.7
    python detect_drone_cli.py --input audio.wav --model models/drone_classifier_subtype.keras
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

from drone_detector import DroneDetector


def main() -> int:
    p = argparse.ArgumentParser(description="Detect drone audio in a WAV file.")
    p.add_argument("--input", required=True, help="Path to a WAV/FLAC/OGG audio file.")
    p.add_argument(
        "--model",
        default="models/drone_classifier_binary.keras",
        help="Path to a trained .keras classifier (default: binary).",
    )
    p.add_argument("--threshold", type=float, default=0.7, help="Detection threshold in [0,1].")
    p.add_argument(
        "--labels",
        default=None,
        help="Optional sidecar labels JSON for multiclass models. "
             "Default: <model>.labels.json next to the model file.",
    )
    args = p.parse_args()

    audio_path = Path(args.input)
    if not audio_path.exists():
        print(f"Input file not found: {audio_path}", file=sys.stderr)
        return 2

    audio, sr = sf.read(str(audio_path), always_2d=False)
    detector = DroneDetector(model_path=args.model, threshold=args.threshold,
                             labels_path=args.labels)
    result = detector.detect(audio, sr)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
