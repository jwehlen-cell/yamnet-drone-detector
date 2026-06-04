"""
Operating-point sweep for the +hardneg model (models/drone_classifier_binary_hardneg.keras).

Replays retrain_with_harvest's exact held-out split (seed=1337) and reports, for
a range of thresholds, the verified-neg false-positive rate (held-out, lower is
better) vs the field-drone recall (held-out). Shows how much recall is recovered
by lowering the threshold while keeping false positives suppressed.
"""
from __future__ import annotations

import re
import numpy as np
import tensorflow as tf
from train import SEED, load_dataset

VN = "qst_verified_negatives"
SENSOR_RE = re.compile(r"(SH\d{3})")
CAP_TRAIN_VN = 5000


def main() -> int:
    ds = load_dataset()
    X, y = ds["X"], ds["y_binary"]
    g = np.array(ds["group_labels"])
    is_vn = np.array([VN in k for k in g])
    is_df = np.array(["qst_detections" in k for k in g])

    # replay the exact rng sequence from retrain_with_harvest.main()
    rng = np.random.RandomState(SEED)
    vn_clips = np.unique(g[is_vn]); rng.shuffle(vn_clips)
    n_hold = max(1, int(0.25 * len(vn_clips)))
    hold_clips = set(vn_clips[:n_hold])
    vn_test = is_vn & np.array([k in hold_clips for k in g])
    vn_train_pool = is_vn & ~vn_test
    pool_idx = np.where(vn_train_pool)[0]
    if len(pool_idx) > CAP_TRAIN_VN:
        rng.choice(pool_idx, CAP_TRAIN_VN, replace=False)   # advance rng identically
    df_clips = np.unique(g[is_df]); rng.shuffle(df_clips)
    df_hold = set(df_clips[:max(1, int(0.25 * len(df_clips)))])
    df_test = is_df & np.array([k in df_hold for k in g])

    m = tf.keras.models.load_model("models/drone_classifier_binary_hardneg.keras", compile=False)
    s_vn = m.predict(X[vn_test], verbose=0).ravel()
    s_df = m.predict(X[df_test], verbose=0).ravel()
    print(f"held-out: {vn_test.sum()} verified-neg windows | {df_test.sum()} field-drone windows\n")
    print(f"{'thr':>5} {'vn_FP_rate':>11} {'field_recall':>13}")
    for thr in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        print(f"{thr:5.2f} {(s_vn >= thr).mean():11.3f} {(s_df >= thr).mean():13.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
