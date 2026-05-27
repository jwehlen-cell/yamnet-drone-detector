# yamnet-drone-detector

A standalone, importable Python module for **drone vs. no-drone** audio
detection, built on top of [YAMNet](https://tfhub.dev/google/yamnet/1)
embeddings and trained on the Embry-Riddle Aeronautical University
YAMNet drone-embedding dataset.

## What this repo is

- A self-contained two-stage detector: **YAMNet → small dense classifier**.
- Trained weights are committed to this repo so it can be cloned and used
  immediately — no training step required.
- A clean Python API (`DroneDetector`) that takes a NumPy waveform and a
  sample rate and returns a verdict.
- A CLI (`detect_drone_cli.py`) for quick file-by-file testing.

## What this repo is not

- **Not** a streaming audio pipeline. It does not capture audio, schedule
  windows, persist results, or talk to any external service.
- **Not** wired into the [drone-audio-sensor](https://github.com/) project
  (or any other pipeline). It's intended to be pulled in as a library by an
  external pipeline that handles I/O, windowing, and reporting.

---

## Dataset and citation

This work uses the **ERAU YAMNet drone-embedding dataset**:

> Embry-Riddle Aeronautical University.
> *YAMNet Embeddings for Drone Detection.*
> Mendeley Data, V3 (2024). DOI: [10.17632/5dmcszvym4.3](https://doi.org/10.17632/5dmcszvym4.3)
> Mirror: <https://datacommons.erau.edu/datasets/5dmcszvym4/3>
> License: **CC BY 4.0**

The dataset contains 1-second segments of YAMNet 1024-dimensional
embeddings, organized as:

```
drone/
  DJI_Matrice_M100/
  Mavic_3/
  Mavic_Mini_2/
no_drone/
  ...
```

Per the dataset README, the top-level `drone/` and `no_drone/` folders are
the binary classification target; the drone subfolders are subtype labels.

---

## Installation

```bash
git clone https://github.com/<you>/yamnet-drone-detector.git
cd yamnet-drone-detector
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Works on CPU. If TensorFlow detects a GPU it will use it; otherwise CPU
inference is fine — the dense head is tiny and YAMNet itself is small.

---

## Using the detector in external code

```python
import soundfile as sf
from drone_detector import DroneDetector

detector = DroneDetector(
    model_path="models/drone_classifier_binary.keras",
    threshold=0.7,
)

audio, sr = sf.read("some_recording.wav")
result = detector.detect(audio, sr)

# result -> {"drone_detected": bool, "confidence": float, "label": str}
print(result)
```

`detect()` is stateless and safe to call repeatedly from any external
pipeline. The `DroneDetector` instance owns the loaded YAMNet and the
trained classifier — construct it once, call `detect()` many times.

### Subtype classifier

To classify *which* drone model is present, swap in the subtype model:

```python
detector = DroneDetector(
    model_path="models/drone_classifier_subtype.keras",
    threshold=0.7,
)
result = detector.detect(audio, sr)
# label ∈ {"DJI_Matrice_M100", "Mavic_3", "Mavic_Mini_2", "no_drone"}
```

The class-index → label mapping lives in
`models/drone_classifier_subtype.labels.json` and is loaded automatically.

### CLI

```bash
python detect_drone_cli.py --input recording.wav --threshold 0.7
```

Accepts WAV / FLAC / OGG (anything `soundfile` can read).

---

## Re-training from scratch

```bash
# 1. Pull the dataset (≈ a few hundred MB).
python download_data.py
# If the automated download fails, follow the printed manual instructions.

# 2. Train both classifiers. Produces models/*.keras + *.tflite + metrics_*.json.
python train.py
```

`train.py` flags:
- `--mode binary` — only train the binary classifier
- `--mode subtype` — only train the subtype classifier
- (default) — train both

Re-training overwrites the committed models. If you want to keep the
shipped weights, train on a branch.

---

## Architecture

```
   raw audio (any SR)
        │
        ▼
   resample → 16 kHz mono, [-1, 1]
        │
        ▼
   YAMNet (TF-Hub, frozen)
        │   embeddings: (T, 1024)
        ▼
   mean-pool over time
        │   (1, 1024)
        ▼
   Dense(256, relu) → Dropout(0.3)
   Dense( 64, relu) → Dropout(0.2)
   Dense(1, sigmoid)            ← binary
   Dense(K, softmax)            ← subtype (K = #classes)
        │
        ▼
   threshold → {drone_detected, confidence, label}
```

The same dense head architecture is used for both classifiers; only the
final layer differs. This matches the ~96% accuracy figure ERAU reports
in their dataset documentation.

---

## Model performance

Trained on the full ERAU dataset (9,108 1-second segments, 80/20 stratified
split, seed=1337). Both classifiers use the same dense head; only the
output layer differs.

| Model | Accuracy | Precision | Recall | F1 | Test set |
|-------|---------:|----------:|-------:|---:|---------:|
| Binary (drone vs no_drone) | **95.2%** | 93.4% | 93.0% | 93.2% | 1,822 |
| Subtype (macro avg)        | **95.1%** | 96.1% | 94.6% | 95.3% | 1,822 |

**Binary confusion matrix** (rows = true, cols = predicted):

|              | pred no_drone | pred drone |
|--------------|--------------:|-----------:|
| true no_drone| 1140          | 42         |
| true drone   |   45          | 595        |

**Subtype labels** (in classifier index order):
`matrice` (DJI Matrice M100), `mavic3` (Mavic 3),
`mavicmini` (Mavic Mini 2), `no_drone`. See
`models/metrics_subtype.json` for the per-class precision/recall report.

Visual confusion matrices: `models/confusion_binary.png`,
`models/confusion_subtype.png`. Full metric JSONs:
`models/metrics_binary.json`, `models/metrics_subtype.json`.

---

## Files

```
drone_detector.py           # Importable module exposing DroneDetector
detect_drone_cli.py         # Thin CLI wrapper for local testing
train.py                    # Training script (binary + subtype)
download_data.py            # Dataset downloader / instructions
deployment_info.json        # Summary: model paths, input format, provenance
requirements.txt
models/
  drone_classifier_binary.keras
  drone_classifier_binary.tflite
  drone_classifier_subtype.keras
  drone_classifier_subtype.tflite
  drone_classifier_subtype.labels.json
  metrics_binary.json
  metrics_subtype.json
  confusion_binary.png
  confusion_subtype.png
data/                       # gitignored; populated by download_data.py
```

---

## License

Code: MIT.
Dataset: CC BY 4.0 — see citation above. The committed trained weights
are derivative works of the dataset and inherit CC BY 4.0; attribute
Embry-Riddle Aeronautical University when redistributing.
