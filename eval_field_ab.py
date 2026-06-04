"""
Fair A/B: does adding the field data (QST + USAFA) to the BINARY head help?

Both models are trained on the SAME clip-level train split and scored on the
SAME held-out test split (GroupShuffleSplit, seed=1337 — the exact split
train.py uses for binary). The ONLY difference is whether field-data windows
are present in the training set:

  baseline : train on non-field rows only (ERAU + pre-existing extras)
  +field   : train on all rows (adds qst_detections / usafa_dfec[_negatives])

Scoring is broken out by domain on the identical test set:
  overall | erau | extra_clean (bebop/mambo/esc50/...) | field (QST+USAFA)

This sidesteps the apples-to-oranges problem with the committed shipped weights
(whose own training set overlaps this test set).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from train import SEED, build_dense_head, load_dataset

FIELD_BUCKETS = {"qst_detections", "usafa_dfec", "usafa_dfec_negatives",
                 "qst_night_negatives", "qst_verified_negatives"}


def _fit(X_tr, y_tr):
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    cw = {0: len(y_tr) / (2.0 * n_neg), 1: len(y_tr) / (2.0 * n_pos)}
    model = build_dense_head(num_classes=1)
    model.fit(
        X_tr, y_tr, validation_split=0.1, epochs=60, batch_size=128,
        class_weight=cw, verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        ],
    )
    return model


def _scores(y_true, y_pred):
    return dict(
        n=int(len(y_true)), pos=int(y_true.sum()),
        acc=accuracy_score(y_true, y_pred),
        prec=precision_score(y_true, y_pred, zero_division=0),
        rec=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
    )


def main() -> int:
    ds = load_dataset()
    X, y = ds["X"], ds["y_binary"]
    groups = np.array(ds["group_labels"])
    source = np.array(ds["source_labels"])
    # NB: load_dataset collapses every negative row's subtype label to
    # "no_drone", so field NEGATIVE buckets must be detected from the group
    # key (which encodes the bucket path), not from subtype_labels.
    is_field = np.array([any(b in g for b in FIELD_BUCKETS) for g in ds["group_labels"]])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(X, y, groups))

    base_tr = tr[~is_field[tr]]           # shipped recipe: no field data in train
    print(f"train: baseline={len(base_tr)}  +field={len(tr)}   test={len(te)}")
    print(f"field windows in train: {int(is_field[tr].sum())}  | field windows in test: {int(is_field[te].sum())}")

    m_base = _fit(X[base_tr], y[base_tr])
    m_field = _fit(X[tr], y[tr])

    slices = {
        "overall":     np.ones(len(te), bool),
        "erau":        source[te] == "erau",
        "extra_clean": (source[te] == "extra") & ~is_field[te],
        "field":       is_field[te],
    }
    pb = (m_base.predict(X[te], verbose=0).ravel() >= 0.5).astype(int)
    pf = (m_field.predict(X[te], verbose=0).ravel() >= 0.5).astype(int)
    yte = y[te]

    print(f"\n{'slice':12s} {'n':>5s} {'pos':>5s} | "
          f"{'base_acc':>8s} {'fld_acc':>8s} | {'base_prec':>9s} {'fld_prec':>9s} | "
          f"{'base_rec':>8s} {'fld_rec':>8s} | {'base_f1':>7s} {'fld_f1':>7s}")
    for name, mask in slices.items():
        if not mask.any():
            continue
        b = _scores(yte[mask], pb[mask])
        f = _scores(yte[mask], pf[mask])
        print(f"{name:12s} {b['n']:5d} {b['pos']:5d} | "
              f"{b['acc']:8.3f} {f['acc']:8.3f} | {b['prec']:9.3f} {f['prec']:9.3f} | "
              f"{b['rec']:8.3f} {f['rec']:8.3f} | {b['f1']:7.3f} {f['f1']:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
