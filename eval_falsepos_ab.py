"""
Isolated A/B for the Kilo-verified false-positive hard negatives
(data/extra/no_drone/qst_verified_negatives).

Question: do these 33 deployment false-positive clips, added as no_drone hard
negatives, suppress Angels Envy's false fires WITHOUT eroding real drone recall?

Two views (binary head, deployed threshold 0.7):

  A. Clip-level held-out (GroupShuffleSplit, seed=1337 like train.py)
       baseline : train on everything EXCEPT qst_verified_negatives
       +hardneg : train on everything (adds the verified negs that land in train)
     Scored on the identical held-out test set, broken out by:
       verified_neg (false-positive rate, lower=better) |
       qst_detections field drone (recall, must NOT drop) | overall

  B. Leave-SH006-out generalization (the strongest claim)
       Hold out EVERY SH006 verified-neg window (the worst false-firer).
       baseline : no verified negs at all
       +hardneg : verified negs from the OTHER sensors only (never SH006)
     If SH006's scores fall, the head generalized "vehicle noise != drone"
     to a sensor whose confounder clips it never trained on.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import tensorflow as tf

from train import SEED, build_dense_head, load_dataset

VN = "qst_verified_negatives"
THR = 0.7
SENSOR_RE = re.compile(r"(SH\d{3})")


def _fit(X_tr, y_tr, seed=SEED):
    # Shuffle before fit: Keras validation_split takes the LAST 10% without
    # shuffling, and appended buckets (verified negs) sit at the tail -- leaving
    # them unshuffled biases early stopping. Permute first.
    rng = np.random.RandomState(seed)
    p = rng.permutation(len(y_tr))
    X_tr, y_tr = X_tr[p], y_tr[p]
    np.random.seed(seed)
    tf.random.set_seed(seed)
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    cw = {0: len(y_tr) / (2.0 * n_neg), 1: len(y_tr) / (2.0 * n_pos)}
    m = build_dense_head(num_classes=1)
    m.fit(X_tr, y_tr, validation_split=0.1, epochs=60, batch_size=128, class_weight=cw,
          verbose=0, callbacks=[
              tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
              tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)])
    return m


def fp_rate(model, X, thr=THR):
    s = model.predict(X, verbose=0).ravel()
    return dict(n=len(s), mean=float(s.mean()),
               fp_at_0_7=float((s >= 0.7).mean()), fp_at_0_5=float((s >= 0.5).mean()))


def main() -> int:
    ds = load_dataset()
    X, y = ds["X"], ds["y_binary"]
    g = np.array(ds["group_labels"])
    groups = g
    is_vn = np.array([VN in k for k in g])
    is_drone_field = np.array(["qst_detections" in k for k in g])
    sensors = np.array([(SENSOR_RE.search(k).group(1) if SENSOR_RE.search(k) else "") for k in g])
    print(f"verified-neg windows: {is_vn.sum()}  across sensors "
          f"{dict(zip(*np.unique(sensors[is_vn], return_counts=True)))}")

    # ---- A. clip-level held-out -------------------------------------------
    print("\n=== A. clip-level held-out (seed=1337) ===")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(X, y, groups))
    base_tr = tr[~is_vn[tr]]
    m_base = _fit(X[base_tr], y[base_tr])
    m_hn = _fit(X[tr], y[tr])

    vn_te = te[is_vn[te]]
    df_te = te[is_drone_field[te]]
    print(f"held-out verified-neg windows: {len(vn_te)} | held-out field-drone windows: {len(df_te)}")
    if len(vn_te):
        b, h = fp_rate(m_base, X[vn_te]), fp_rate(m_hn, X[vn_te])
        print(f"  verified-neg FALSE-POSITIVE rate  base->hn:")
        print(f"    mean score   {b['mean']:.3f} -> {h['mean']:.3f}")
        print(f"    FP @0.7      {b['fp_at_0_7']:.3f} -> {h['fp_at_0_7']:.3f}")
        print(f"    FP @0.5      {b['fp_at_0_5']:.3f} -> {h['fp_at_0_5']:.3f}")
    if len(df_te):
        rb = float((m_base.predict(X[df_te], verbose=0).ravel() >= THR).mean())
        rh = float((m_hn.predict(X[df_te], verbose=0).ravel() >= THR).mean())
        print(f"  field-drone RECALL @0.7   base->hn:  {rb:.3f} -> {rh:.3f}  (must not drop)")
    # collateral check: overall + clean-domain negatives must be unharmed
    source = np.array(ds["source_labels"])
    pos_te = te[y[te] == 1]
    clean_neg_te = te[(y[te] == 0) & ~is_vn[te] & ~is_drone_field[te]]
    ob = float((m_base.predict(X[te], verbose=0).ravel() >= THR).astype(int) == y[te]).mean() if False else None
    for nm, idx, want in [("overall ACC", te, None),
                          ("all-drone RECALL@0.7", pos_te, "keep"),
                          ("clean-neg FP@0.7", clean_neg_te, "keep-low")]:
        pb = m_base.predict(X[idx], verbose=0).ravel()
        ph = m_hn.predict(X[idx], verbose=0).ravel()
        if nm.startswith("overall"):
            vb = ((pb >= THR).astype(int) == y[idx]).mean()
            vh = ((ph >= THR).astype(int) == y[idx]).mean()
        elif "RECALL" in nm:
            vb, vh = (pb >= THR).mean(), (ph >= THR).mean()
        else:
            vb, vh = (pb >= THR).mean(), (ph >= THR).mean()
        print(f"  {nm:22s} base->hn:  {vb:.3f} -> {vh:.3f}")

    # ---- B. leave-SH006-out -----------------------------------------------
    print("\n=== B. leave-SH006-out generalization ===")
    held = is_vn & (sensors == "SH006")
    base_mask = ~is_vn
    hn_mask = ~is_vn | (is_vn & (sensors != "SH006"))
    print(f"SH006 verified-neg held out: {held.sum()} windows | "
          f"other-sensor verified-neg added to +hardneg: {int((is_vn & (sensors!='SH006')).sum())}")
    m_base2 = _fit(X[base_mask], y[base_mask])
    m_hn2 = _fit(X[hn_mask], y[hn_mask])
    b, h = fp_rate(m_base2, X[held]), fp_rate(m_hn2, X[held])
    print(f"  SH006 (UNSEEN) false-positive rate  base->hn:")
    print(f"    mean score   {b['mean']:.3f} -> {h['mean']:.3f}")
    print(f"    FP @0.7      {b['fp_at_0_7']:.3f} -> {h['fp_at_0_7']:.3f}")
    print(f"    FP @0.5      {b['fp_at_0_5']:.3f} -> {h['fp_at_0_5']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
