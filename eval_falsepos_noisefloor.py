"""
Noise-floor-controlled read on the 33-clip verified-negative retrain.

Fixes the validation-ordering footgun (shuffle train rows before fit) and
measures the verified-neg false-positive rate three ways on the SAME held-out
verified-neg clips:

    base@seedA, base@seedB  -> spread from training noise alone (data fixed)
    hardneg@seedA           -> effect of adding the verified negs

If |hardneg - base@seedA| is within the base@seedA vs base@seedB spread, the
33-clip snapshot is too small to move the head -- the effect is noise.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

from train import SEED, build_dense_head, load_dataset

VN = "qst_verified_negatives"


def fit(X_tr, y_tr, seed):
    rng = np.random.RandomState(seed)
    p = rng.permutation(len(y_tr))               # shuffle so val_split isn't the tail
    X_tr, y_tr = X_tr[p], y_tr[p]
    np.random.seed(seed); tf.random.set_seed(seed)
    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
    cw = {0: len(y_tr) / (2.0 * n_neg), 1: len(y_tr) / (2.0 * n_pos)}
    m = build_dense_head(num_classes=1)
    m.fit(X_tr, y_tr, validation_split=0.1, epochs=60, batch_size=128, class_weight=cw,
          verbose=0, callbacks=[
              tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
              tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)])
    return m


def fp(m, X):
    s = m.predict(X, verbose=0).ravel()
    return s.mean(), (s >= 0.7).mean()


def main() -> int:
    ds = load_dataset()
    X, y = ds["X"], ds["y_binary"]
    g = np.array(ds["group_labels"]); groups = g
    is_vn = np.array([VN in k for k in g])

    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(X, y, groups))
    base_tr = tr[~is_vn[tr]]
    vn_te = te[is_vn[te]]
    print(f"held-out verified-neg windows: {len(vn_te)}")

    mb1 = fit(X[base_tr], y[base_tr], 1)
    mb2 = fit(X[base_tr], y[base_tr], 2)   # same data, different seed -> noise floor
    mh1 = fit(X[tr], y[tr], 1)             # + verified negs

    for name, m in [("base@seedA", mb1), ("base@seedB", mb2), ("hardneg@seedA", mh1)]:
        mean, r07 = fp(m, X[vn_te])
        print(f"  {name:14s} verified-neg  mean={mean:.3f}  FP@0.7={r07:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
