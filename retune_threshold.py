"""
Retune the live inference detection_threshold for the shipped hard-negative model.

The worker mean-pools each clip into ONE drone_score (model.py infer_pcm16), and a
4 s Shaw clip over detection_threshold already exceeds min_seconds_over_threshold
(3 s) -> one clip over threshold fires. So detection_threshold operates on the
per-clip mean-pool score -- the same distribution as DroneDetector.detect.

Computes, for the SHIPPED model, the per-clip score on:
  - held-out CONFOUNDERS (data/_vn_holdout, NOT used in training) -> false-positive rate
  - real USAFA drones (data/extra/drone/usafa_dfec, confirmed close-range) -> detect rate
  - qst_detections (UNVERIFIED field 'positives', caveated) -> detect rate
and prints the FP/detect tradeoff vs threshold.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).parent
WIN = re.compile(r"_w\d+$")
model = tf.keras.models.load_model(ROOT / "models/drone_classifier_binary.keras", compile=False)


def per_clip_scores(d: Path) -> np.ndarray:
    """Mean-pool each source clip's windows -> one embedding -> one score."""
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for p in d.rglob("*.tfdata"):
        key = WIN.sub("", p.stem)
        e = tf.io.parse_tensor(tf.io.read_file(str(p)).numpy(), tf.float32).numpy().reshape(-1, 1024)
        groups[key].append(e.mean(0))
    if not groups:
        return np.array([])
    X = np.stack([np.mean(v, axis=0) for v in groups.values()]).astype(np.float32)
    return model.predict(X, verbose=0).ravel()


conf = per_clip_scores(ROOT / "data/_vn_holdout")                      # held-out confounders
usafa = per_clip_scores(ROOT / "data/extra/drone/usafa_dfec")          # real drones
qst = per_clip_scores(ROOT / "data/extra/drone/qst_detections")        # unverified field

print(f"held-out confounder clips: {len(conf)} | USAFA real-drone clips: {len(usafa)} | "
      f"qst_detections clips: {len(qst)}\n")
print(f"{'thr':>5} {'confounder_FP':>14} {'USAFA_detect':>13} {'qst_detect(unver)':>18}")
for t in [0.45, 0.50, 0.60, 0.70, 0.80, 0.90]:
    fp = float((conf >= t).mean()) if len(conf) else float("nan")
    ud = float((usafa >= t).mean()) if len(usafa) else float("nan")
    qd = float((qst >= t).mean()) if len(qst) else float("nan")
    print(f"{t:5.2f} {fp:14.3f} {ud:13.3f} {qd:18.3f}")
print(f"\nconfounder score: median={np.median(conf):.3f} p95={np.percentile(conf,95):.3f} "
      f"p99={np.percentile(conf,99):.3f}")
print(f"USAFA drone score: median={np.median(usafa):.3f} min={usafa.min():.3f}")
