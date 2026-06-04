"""
Standalone drone-audio detector built on YAMNet embeddings.

Two-stage architecture:
  1. YAMNet (TF-Hub) maps raw 16 kHz mono audio to 1024-d embeddings per
     ~0.96 s frame.
  2. A dense classifier (trained from train.py) maps those embeddings to
     drone probability (binary) or drone-subtype probabilities (multiclass).

The detector is stateless: a single shared YAMNet model + classifier are
loaded once at construction, and `detect()` may be called repeatedly from
any external pipeline.

Typical usage:

    from drone_detector import DroneDetector
    detector = DroneDetector(
        model_path="models/drone_classifier_binary.keras",
        threshold=0.7,
    )
    result = detector.detect(audio, sample_rate)
    # -> {"drone_detected": bool, "confidence": float, "label": str}
"""

from __future__ import annotations

import json
import os
from math import gcd
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

YAMNET_TFHUB_URL = "https://tfhub.dev/google/yamnet/1"
YAMNET_SR = 16000
EMBEDDING_DIM = 1024


class DroneDetector:
    def __init__(
        self,
        model_path: str | os.PathLike,
        threshold: float = 0.45,
        labels_path: Optional[str | os.PathLike] = None,
        yamnet_url: str = YAMNET_TFHUB_URL,
    ) -> None:
        self.threshold = float(threshold)
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)

        self.yamnet = hub.load(yamnet_url)
        self.classifier = tf.keras.models.load_model(self.model_path)

        out_units = int(self.classifier.output_shape[-1])
        self.is_binary = out_units == 1

        if self.is_binary:
            self.labels = ["no_drone", "drone"]
        else:
            # Multiclass: load labels from sidecar JSON if available, otherwise fall
            # back to numeric labels.
            if labels_path is None:
                guess = self.model_path.with_suffix("").as_posix() + ".labels.json"
                labels_path = guess
            labels_path = Path(labels_path)
            if labels_path.exists():
                self.labels = json.loads(labels_path.read_text())
            else:
                self.labels = [f"class_{i}" for i in range(out_units)]

    # ------------------------------------------------------------------
    # Audio preprocessing
    # ------------------------------------------------------------------

    def _to_yamnet_input(self, audio: np.ndarray, sample_rate: int) -> tf.Tensor:
        audio = np.asarray(audio)
        if audio.dtype.kind in ("i", "u"):
            # int PCM -> normalize to [-1, 1]
            max_val = float(np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float32) / max_val
        else:
            audio = audio.astype(np.float32)

        if audio.ndim == 2:
            # (channels, samples) or (samples, channels) -> mono via mean over the
            # smaller axis (channels are almost always the smaller dimension).
            ch_axis = int(np.argmin(audio.shape))
            audio = audio.mean(axis=ch_axis)
        elif audio.ndim != 1:
            raise ValueError(f"Expected 1-D or 2-D audio, got shape {audio.shape}")

        if sample_rate != YAMNET_SR:
            from scipy.signal import resample_poly
            g = gcd(int(sample_rate), YAMNET_SR)
            up = YAMNET_SR // g
            down = int(sample_rate) // g
            audio = resample_poly(audio, up, down).astype(np.float32)

        # Clip to [-1, 1] to be safe.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak

        return tf.constant(audio, dtype=tf.float32)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _embed(self, wav: tf.Tensor) -> np.ndarray:
        """Run YAMNet, return mean-pooled (1, 1024) embedding."""
        _scores, embeddings, _spec = self.yamnet(wav)
        emb = embeddings.numpy()  # (T, 1024)
        if emb.size == 0:
            raise ValueError("Audio too short for YAMNet to produce embeddings.")
        return emb.mean(axis=0, keepdims=True).astype(np.float32)

    def detect(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Run drone detection on a single audio clip.

        Parameters
        ----------
        audio : np.ndarray
            Mono or multichannel waveform. int PCM or float in [-1, 1].
        sample_rate : int
            Sample rate of `audio`. Will be resampled to 16 kHz internally.

        Returns
        -------
        dict with keys:
          drone_detected : bool
          confidence     : float in [0, 1]
          label          : str
        """
        wav = self._to_yamnet_input(audio, sample_rate)
        emb = self._embed(wav)
        pred = self.classifier.predict(emb, verbose=0)

        if self.is_binary:
            confidence = float(pred[0, 0])
            detected = confidence >= self.threshold
            label = "drone" if detected else "no_drone"
            return {
                "drone_detected": bool(detected),
                "confidence": confidence,
                "label": label,
            }

        probs = pred[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        label = self.labels[idx]
        detected = (label != "no_drone") and (confidence >= self.threshold)
        return {
            "drone_detected": bool(detected),
            "confidence": confidence,
            "label": label,
        }
