"""
Train drone / no-drone classifiers on ERAU YAMNet embeddings.

Produces:
  models/drone_classifier_binary.keras
  models/drone_classifier_binary.tflite
  models/drone_classifier_subtype.keras
  models/drone_classifier_subtype.tflite
  models/drone_classifier_subtype.labels.json
  models/metrics_binary.json
  models/metrics_subtype.json
  models/confusion_binary.png
  models/confusion_subtype.png

Usage:
  python train.py                         # trains both
  python train.py --mode binary           # only binary
  python train.py --mode subtype          # only subtype
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Extra dataset roots produced by ``embed_extra_audio.py``. Each root mirrors
# the ERAU on-disk shape (``<root>/drone/<subtype>/*.tfdata`` and
# ``<root>/no_drone/**/*.tfdata``), so the same loader handles both.
EXTRA_DATA_ROOTS: tuple[Path, ...] = (DATA_DIR / "extra",)

# Subtype labels we don't want as targets for the subtype head, even though
# we keep them as positives for the binary head. The mackenzie-jane
# visualization repo gives us one 5-sec WAV per drone model -> not enough
# evidence per model to learn a class. We bucket them under the literal
# folder name "visualization_samples" and drop that label here.
SUBTYPE_EXCLUDED_LABELS: frozenset[str] = frozenset({"visualization_samples"})

EMBEDDING_DIM = 1024
SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _list_tfdata_files(class_root: Path) -> list[Path]:
    return sorted(class_root.rglob("*.tfdata"))


def _parse_one_tfdata(path: Path) -> np.ndarray:
    """
    Load a single .tfdata file and return its embeddings as (T, 1024).

    ERAU's files are tf.io.serialize_tensor outputs of a (T, 1024) float32
    tensor, where T = 2 for one second of audio. We try parse_tensor first,
    then fall back to raw float32 buffer or TFRecord parsing for robustness.
    """
    raw_bytes = tf.io.read_file(str(path)).numpy()

    try:
        t = tf.io.parse_tensor(raw_bytes, out_type=tf.float32).numpy().astype(np.float32)
        if t.size % EMBEDDING_DIM == 0 and t.size > 0:
            return t.reshape(-1, EMBEDDING_DIM)
    except Exception:
        pass

    try:
        arr = np.frombuffer(raw_bytes, dtype=np.float32)
        if arr.size % EMBEDDING_DIM == 0 and arr.size > 0:
            return arr.reshape(-1, EMBEDDING_DIM)
    except Exception:
        pass

    try:
        for rec in tf.data.TFRecordDataset([str(path)]):
            for spec in (
                {"embedding": tf.io.FixedLenFeature([EMBEDDING_DIM], tf.float32)},
                {"embeddings": tf.io.FixedLenFeature([EMBEDDING_DIM], tf.float32)},
                {"embedding": tf.io.VarLenFeature(tf.float32)},
            ):
                try:
                    parsed = tf.io.parse_single_example(rec, spec)
                    val = parsed[next(iter(spec))]
                    if isinstance(val, tf.SparseTensor):
                        val = tf.sparse.to_dense(val)
                    arr = val.numpy().astype(np.float32).reshape(-1)
                    if arr.size % EMBEDDING_DIM == 0 and arr.size > 0:
                        return arr.reshape(-1, EMBEDDING_DIM)
                except Exception:
                    continue
            break
    except Exception:
        pass

    raise ValueError(f"Could not parse embeddings from {path}")


def load_split(class_root: Path) -> tuple[np.ndarray, list[str]]:
    """
    For one top-level class folder (e.g. data/drone), load all .tfdata files,
    average-pool the time dimension to a single 1024-vector per file, and
    return (X, subtype_names) where subtype_names is the immediate
    subdirectory name for each row (e.g. 'DJI_Matrice_M100').
    """
    files = _list_tfdata_files(class_root)
    if not files:
        raise SystemExit(f"No .tfdata files found under {class_root}")
    X = np.zeros((len(files), EMBEDDING_DIM), dtype=np.float32)
    subtypes: list[str] = []
    skipped = 0
    for i, p in enumerate(tqdm(files, desc=f"loading {class_root.name}")):
        try:
            emb = _parse_one_tfdata(p)
        except Exception:
            skipped += 1
            continue
        X[i] = emb.mean(axis=0)
        rel = p.relative_to(class_root)
        subtypes.append(rel.parts[0] if len(rel.parts) > 1 else class_root.name)
    if skipped:
        print(f"  Skipped {skipped} unparseable files in {class_root.name}")
        X = X[: len(subtypes)]
    return X, subtypes


def load_dataset(extra_roots: Iterable[Path] = EXTRA_DATA_ROOTS) -> dict:
    """
    Load the ERAU base dataset plus any extra roots that follow the same
    ``<root>/drone/<subtype>/`` and ``<root>/no_drone/`` layout. Extra roots
    that don't exist on disk are silently skipped, so the script still works
    on a fresh checkout without the extras downloaded.

    Returns a dict with keys ``X``, ``y_binary``, ``subtype_labels``,
    ``source_labels`` (one of ``"erau"`` or the extra root name), and
    per-subtype counts.
    """
    drone_root = DATA_DIR / "drone"
    no_drone_root = DATA_DIR / "no_drone"
    if not drone_root.is_dir() or not no_drone_root.is_dir():
        raise SystemExit(
            f"Expected {drone_root} and {no_drone_root}. "
            f"Run `python download_data.py` first."
        )

    X_drone, sub_drone = load_split(drone_root)
    X_nodrone, sub_nodrone = load_split(no_drone_root)
    source_drone = ["erau"] * len(X_drone)
    source_nodrone = ["erau"] * len(X_nodrone)

    for root in extra_roots:
        root = Path(root)
        extra_drone = root / "drone"
        extra_nodrone = root / "no_drone"
        if extra_drone.is_dir():
            xd, sd = load_split(extra_drone)
            X_drone = np.concatenate([X_drone, xd], axis=0)
            sub_drone += sd
            source_drone += [root.name] * len(xd)
        if extra_nodrone.is_dir():
            xn, sn = load_split(extra_nodrone)
            X_nodrone = np.concatenate([X_nodrone, xn], axis=0)
            sub_nodrone += sn
            source_nodrone += [root.name] * len(xn)

    X = np.concatenate([X_drone, X_nodrone], axis=0)
    y_binary = np.concatenate(
        [np.ones(len(X_drone), dtype=np.int32), np.zeros(len(X_nodrone), dtype=np.int32)]
    )
    subtype_labels = sub_drone + ["no_drone"] * len(X_nodrone)
    source_labels = source_drone + source_nodrone
    return {
        "X": X,
        "y_binary": y_binary,
        "subtype_labels": subtype_labels,
        "source_labels": source_labels,
        "drone_subtype_counts": {s: sub_drone.count(s) for s in sorted(set(sub_drone))},
        "n_drone": len(X_drone),
        "n_no_drone": len(X_nodrone),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def build_dense_head(num_classes: int) -> tf.keras.Model:
    """
    Two dense layers on top of pooled YAMNet embeddings.
    ERAU reports ~96% binary accuracy with this style of head.
    """
    inputs = tf.keras.Input(shape=(EMBEDDING_DIM,), name="yamnet_embedding")
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    if num_classes == 1:
        out = tf.keras.layers.Dense(1, activation="sigmoid", name="drone_prob")(x)
        loss = "binary_crossentropy"
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    else:
        out = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probs")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    model = tf.keras.Model(inputs, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=metrics)
    return model


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def _save_confusion(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4 + 0.4 * len(labels), 4 + 0.3 * len(labels)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _export_tflite(keras_path: Path, tflite_path: Path) -> None:
    model = tf.keras.models.load_model(keras_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    tflite_path.write_bytes(tflite_bytes)


def train_binary(ds: dict) -> dict:
    print("\n=== Binary classifier: drone vs no_drone ===")
    X, y = ds["X"], ds["y_binary"]
    sources = np.array(ds["source_labels"])
    indices = np.arange(len(X))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_test = X[idx_train], X[idx_test]
    sources_test = sources[idx_test]
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    class_weight = {0: len(y_train) / (2.0 * n_neg), 1: len(y_train) / (2.0 * n_pos)}

    model = build_dense_head(num_classes=1)
    model.summary(print_fn=print)

    ckpt = MODEL_DIR / "drone_classifier_binary.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt), save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=60, batch_size=128,
        class_weight=class_weight,
        callbacks=callbacks, verbose=2,
    )

    probs = model.predict(X_test, verbose=0).ravel()
    preds = (probs >= 0.5).astype(np.int32)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_label": "drone",
    }
    cm = confusion_matrix(y_test, preds)
    metrics["confusion_matrix"] = cm.tolist()

    # Per-source breakdown: how does the binary classifier do on each origin
    # dataset? Surfaces whether the new data helps or hurts ERAU eval.
    per_source: dict[str, dict] = {}
    for src in sorted(set(sources_test.tolist())):
        mask = sources_test == src
        if not mask.any():
            continue
        y_s = y_test[mask]
        p_s = preds[mask]
        per_source[src] = {
            "n": int(mask.sum()),
            "n_pos": int(y_s.sum()),
            "accuracy": float(accuracy_score(y_s, p_s)),
            "precision": float(precision_score(y_s, p_s, zero_division=0)),
            "recall": float(recall_score(y_s, p_s, zero_division=0)),
            "f1": float(f1_score(y_s, p_s, zero_division=0)),
        }
    metrics["per_source"] = per_source
    print(json.dumps(metrics, indent=2))

    _save_confusion(cm, ["no_drone", "drone"], MODEL_DIR / "confusion_binary.png",
                    "Binary confusion matrix")
    (MODEL_DIR / "metrics_binary.json").write_text(json.dumps(metrics, indent=2))

    tflite_path = MODEL_DIR / "drone_classifier_binary.tflite"
    _export_tflite(ckpt, tflite_path)
    print(f"Wrote {ckpt} and {tflite_path}")
    return metrics


def train_subtype(ds: dict) -> dict:
    print("\n=== Subtype classifier ===")
    # Exclude buckets that can't form a real class (e.g. one clip per drone
    # model from the visualization repo). They stay in the binary head but
    # are removed before the subtype split.
    keep_mask = np.array(
        [lab not in SUBTYPE_EXCLUDED_LABELS for lab in ds["subtype_labels"]]
    )
    subtype_labels_kept = [
        lab for lab, k in zip(ds["subtype_labels"], keep_mask) if k
    ]
    if len(subtype_labels_kept) != len(ds["subtype_labels"]):
        dropped = len(ds["subtype_labels"]) - len(subtype_labels_kept)
        print(
            f"  Excluding {dropped} rows with labels in {sorted(SUBTYPE_EXCLUDED_LABELS)}"
        )

    labels = sorted(set(subtype_labels_kept))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    y = np.array([label_to_idx[l] for l in subtype_labels_kept], dtype=np.int32)
    X = ds["X"][keep_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    counts = np.bincount(y_train, minlength=len(labels))
    class_weight = {i: (len(y_train) / (len(labels) * max(c, 1))) for i, c in enumerate(counts)}

    model = build_dense_head(num_classes=len(labels))
    model.summary(print_fn=print)

    ckpt = MODEL_DIR / "drone_classifier_subtype.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt), save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=60, batch_size=128,
        class_weight=class_weight,
        callbacks=callbacks, verbose=2,
    )

    probs = model.predict(X_test, verbose=0)
    preds = probs.argmax(axis=1)
    metrics = {
        "labels": labels,
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision_macro": float(precision_score(y_test, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "per_class_report": classification_report(
            y_test, preds, target_names=labels, output_dict=True, zero_division=0
        ),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    cm = confusion_matrix(y_test, preds)
    metrics["confusion_matrix"] = cm.tolist()
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("confusion_matrix", "per_class_report")}, indent=2))

    _save_confusion(cm, labels, MODEL_DIR / "confusion_subtype.png",
                    "Subtype confusion matrix")
    (MODEL_DIR / "metrics_subtype.json").write_text(json.dumps(metrics, indent=2))
    (MODEL_DIR / "drone_classifier_subtype.labels.json").write_text(json.dumps(labels, indent=2))

    tflite_path = MODEL_DIR / "drone_classifier_subtype.tflite"
    _export_tflite(ckpt, tflite_path)
    print(f"Wrote {ckpt} and {tflite_path}")
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("both", "binary", "subtype"), default="both")
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}, GPUs visible: {len(gpus)}")

    ds = load_dataset()
    print(f"\nLoaded {len(ds['X'])} embeddings ({EMBEDDING_DIM}-d each)")
    print(f"  drone:    {ds['n_drone']}")
    print(f"  no_drone: {ds['n_no_drone']}")
    print(f"  drone subtype counts: {ds['drone_subtype_counts']}")
    src_counts: dict[str, int] = {}
    for s in ds["source_labels"]:
        src_counts[s] = src_counts.get(s, 0) + 1
    print(f"  by source: {src_counts}")

    if args.mode in ("both", "binary"):
        train_binary(ds)
    if args.mode in ("both", "subtype"):
        train_subtype(ds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
