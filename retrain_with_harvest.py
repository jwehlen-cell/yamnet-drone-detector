"""
Retrain Angels Envy (binary head) with the 8h-harvested Kilo-verified false
positives as hard negatives, at scale, with a balanced subsample.

The harvest can yield 10k+ false-positive clips (~36% of the stream). Dumping
all of them would swamp the positive class and crater recall. So we CAP the
verified-neg windows used for training (CAP_TRAIN_VN) and hold the remainder
out as a large, honest false-positive-rate test set.

Reports (corrected shuffling fit, deployed threshold 0.7):
  held-out verified-neg FP rate     base -> +hardneg   (lower = better)
  field-drone (qst_detections) recall                  (must not crater)
  overall acc | clean-domain neg FP | leave-SH006-out generalization

Saves the +hardneg model -> models/drone_classifier_binary_hardneg.keras
(+ metrics json). Does NOT overwrite the shipped main weights.
"""
from __future__ import annotations

import json
import re

import numpy as np
import tensorflow as tf

from train import SEED, build_dense_head, load_dataset

VN = "qst_verified_negatives"
THR = 0.7
CAP_TRAIN_VN = 5000          # max verified-neg WINDOWS used in training
SENSOR_RE = re.compile(r"(SH\d{3})")
MODEL_OUT = "models/drone_classifier_binary_hardneg.keras"
METRICS_OUT = "models/metrics_binary_hardneg.json"


def fit(X, y, seed=SEED):
    rng = np.random.RandomState(seed)
    p = rng.permutation(len(y))
    X, y = X[p], y[p]
    np.random.seed(seed); tf.random.set_seed(seed)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    cw = {0: len(y) / (2.0 * n_neg), 1: len(y) / (2.0 * n_pos)}
    m = build_dense_head(num_classes=1)
    m.fit(X, y, validation_split=0.1, epochs=60, batch_size=128, class_weight=cw, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)])
    return m


def rate(m, X, thr=THR):
    return float((m.predict(X, verbose=0).ravel() >= thr).mean()) if len(X) else float("nan")


def main() -> int:
    ds = load_dataset()
    X, y = ds["X"], ds["y_binary"]
    g = np.array(ds["group_labels"])
    is_vn = np.array([VN in k for k in g])
    is_df = np.array(["qst_detections" in k for k in g])
    source = np.array(ds["source_labels"])
    sens = np.array([(SENSOR_RE.search(k).group(1) if SENSOR_RE.search(k) else "") for k in g])
    print(f"dataset: {len(y)} windows | verified-neg: {int(is_vn.sum())} "
          f"across {dict(zip(*np.unique(sens[is_vn], return_counts=True)))}")

    rng = np.random.RandomState(SEED)
    # clip-level split: hold out 25% of verified-neg CLIPS for FP eval
    vn_clips = np.unique(g[is_vn])
    rng.shuffle(vn_clips)
    n_hold = max(1, int(0.25 * len(vn_clips)))
    hold_clips = set(vn_clips[:n_hold])
    vn_test = is_vn & np.array([k in hold_clips for k in g])
    vn_train_pool = is_vn & ~vn_test
    # cap training verified-neg windows
    pool_idx = np.where(vn_train_pool)[0]
    if len(pool_idx) > CAP_TRAIN_VN:
        keep = set(rng.choice(pool_idx, CAP_TRAIN_VN, replace=False))
        vn_train = np.array([i in keep for i in range(len(y))])
    else:
        vn_train = vn_train_pool
    print(f"verified-neg: {int(vn_train.sum())} train (capped {CAP_TRAIN_VN}) | "
          f"{int(vn_test.sum())} held-out test ({len(hold_clips)} clips)")

    # held-out field-drone clips for recall (clip-level, 25%)
    df_clips = np.unique(g[is_df]); rng.shuffle(df_clips)
    df_hold = set(df_clips[:max(1, int(0.25 * len(df_clips)))])
    df_test = is_df & np.array([k in df_hold for k in g])
    # clean-domain negatives held out (extra, non-field): sample
    clean_neg = (source == "extra") & ~is_vn & ~is_df & (y == 0)
    cn_idx = np.where(clean_neg)[0]; cn_test = np.zeros(len(y), bool)
    cn_test[rng.choice(cn_idx, min(2000, len(cn_idx)), replace=False)] = True

    # exclude every held-out window from training
    held = vn_test | df_test | cn_test
    base_mask = ~is_vn & ~held
    hn_mask = (~is_vn & ~held) | vn_train

    print("training baseline (no verified negs)...", flush=True)
    m_base = fit(X[base_mask], y[base_mask])
    print("training +hardneg...", flush=True)
    m_hn = fit(X[hn_mask], y[hn_mask])

    res = {
        "verified_neg_FP@0.7": [rate(m_base, X[vn_test]), rate(m_hn, X[vn_test])],
        "field_drone_recall@0.7": [rate(m_base, X[df_test]), rate(m_hn, X[df_test])],
        "clean_neg_FP@0.7": [rate(m_base, X[cn_test]), rate(m_hn, X[cn_test])],
        "vn_test_n": int(vn_test.sum()), "df_test_n": int(df_test.sum()),
        "vn_train_n": int(vn_train.sum()), "cap": CAP_TRAIN_VN,
    }
    # leave-SH006-out generalization
    held6 = is_vn & (sens == "SH006")
    base6 = ~is_vn & ~held
    other_vn = vn_train & (sens != "SH006")
    hn6 = (~is_vn & ~held) | other_vn
    m_b6 = fit(X[base6], y[base6]); m_h6 = fit(X[hn6], y[hn6])
    res["SH006_unseen_FP@0.7"] = [rate(m_b6, X[held6]), rate(m_h6, X[held6])]

    print("\n=== AT-SCALE RETRAIN A/B (base -> +hardneg) ===")
    for k, v in res.items():
        if isinstance(v, list) and len(v) == 2 and all(isinstance(x, float) for x in v):
            print(f"  {k:28s} {v[0]:.3f} -> {v[1]:.3f}")
        else:
            print(f"  {k:28s} {v}")

    m_hn.save(MODEL_OUT)
    json.dump(res, open(METRICS_OUT, "w"), indent=2)
    print(f"\nsaved {MODEL_OUT} + {METRICS_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
