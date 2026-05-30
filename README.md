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

## Datasets and citation

The shipped weights are trained on a merged dataset spanning three public
sources:

1. **ERAU YAMNet drone-embedding dataset** (subtype labels
   `matrice`, `mavic3`, `mavicmini`, plus ~5.9k `no_drone` clips):

   > Embry-Riddle Aeronautical University.
   > *YAMNet Embeddings for Drone Detection.*
   > Mendeley Data, V3 (2024). DOI: [10.17632/5dmcszvym4.3](https://doi.org/10.17632/5dmcszvym4.3)
   > Mirror: <https://datacommons.erau.edu/datasets/5dmcszvym4/3>
   > License: **CC BY 4.0**

2. **saraalemadi/DroneAudioDataset** (adds subtype labels `bebop` and
   `mambo` and ~10k ESC-50/speech-noise negatives, of which we sample
   2,000 to keep the negative class balanced):

   > Sara Al-Emadi et al. *Audio Based Drone Detection and Identification
   > using Deep Learning*, IWCMC 2019.
   > Repo: <https://github.com/saraalemadi/DroneAudioDataset>
   > Negatives originate from ESC-50 (Piczak 2015) and the TensorFlow
   > Speech Commands dataset (Warden 2018).

3. **mackenzie-jane/drone-visualization** — 32 sample WAVs (one per
   drone model) from the 2025 *Multiclass Acoustic Dataset* arXiv paper.
   Used as additional **binary positives only** (one clip per model
   isn't enough to learn an individual subtype):

   > Chao Wang et al. *A Multiclass Acoustic Dataset and Interactive
   > Tool for Analyzing Drone Signatures in Real-World Environments.*
   > arXiv:2509.04715, 2025.
   > Repo: <https://github.com/mackenzie-jane/drone-visualization>

On-disk layout after running `download_data.py` + `embed_extra_audio.py`:

```
data/
  drone/                       # ERAU (.tfdata embeddings)
    matrice/  mavic3/  mavicmini/
  no_drone/                    # ERAU (.tfdata embeddings)
  extra/                       # Produced from raw WAV by embed_extra_audio.py
    drone/
      bebop/  mambo/  visualization_samples/
    no_drone/
      esc50_noise/
  extra_raw/                   # Cloned source WAV repos (gitignored)
    DroneAudioDataset/  drone-visualization/
```

Subtype subfolders are the subtype labels. `visualization_samples/` is
excluded from subtype training automatically (see
`SUBTYPE_EXCLUDED_LABELS` in `train.py`) but kept as positive examples
for the binary head.

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
# label ∈ {"bebop", "mambo", "matrice", "mavic3", "mavicmini", "no_drone"}
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
# 1. Pull the ERAU embedding dataset (≈ a few hundred MB).
python download_data.py

# 2. Clone the extra raw-audio datasets into data/extra_raw/ and convert
#    them to YAMNet embeddings under data/extra/.
mkdir -p data/extra_raw && cd data/extra_raw
git clone --depth 1 https://github.com/saraalemadi/DroneAudioDataset.git
git clone --depth 1 https://github.com/mackenzie-jane/drone-visualization.git
cd ../..
python embed_extra_audio.py    # writes data/extra/{drone,no_drone}/...

# 3. Train both classifiers. Produces models/*.keras + *.tflite + metrics_*.json.
python train.py
```

`train.py` flags:
- `--mode binary` — only train the binary classifier
- `--mode subtype` — only train the subtype classifier
- (default) — train both

If `data/extra/` is absent, training falls back to the ERAU-only set
(matching the original 4-class label list). Re-training overwrites the
committed models — train on a branch if you want to keep the shipped
weights.

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

Trained on 12,472 1-second YAMNet embeddings (9,108 ERAU + 3,364 from the
extra datasets), 80/20 stratified split, seed=1337. Both classifiers use
the same dense head; only the output layer differs.

| Model | Accuracy | Precision | Recall | F1 | Test set |
|-------|---------:|----------:|-------:|---:|---------:|
| Binary (drone vs no_drone) | **95.3%** | 93.0% | 94.2% | 93.6% | 2,495 |
| Subtype (macro avg)        | **93.6%** | 89.9% | 93.5% | 91.6% | 2,488 |

Binary per-source eval (`metrics_binary.json::per_source`):

| Source | n_test | accuracy | precision | recall | F1 |
|--------|-------:|---------:|----------:|-------:|---:|
| erau   | 1,811  | 95.5%    | 91.9%     | 95.4%  | 93.6% |
| extra  |   684  | 94.7%    | 95.6%     | 91.5%  | 93.5% |

**Subtype labels** (in classifier index order):
`bebop` (Parrot Bebop), `mambo` (Parrot Mambo), `matrice` (DJI Matrice
M100), `mavic3` (Mavic 3), `mavicmini` (Mavic Mini 2), `no_drone`.

Per-class subtype F1: bebop 0.88, mambo 0.86, matrice 0.88, mavic3 0.96,
mavicmini 0.95, no_drone 0.96. See `models/metrics_subtype.json` for
full precision/recall and confusion matrix.

Compared to the ERAU-only baseline (binary 95.2% / subtype 95.1% on
4 classes), the enhanced detector:

- maintains binary accuracy (+0.05 pp) on a 37% larger and more diverse
  test set — the negatives now include ESC-50 environmental sounds
  rather than only ERAU's no_drone clips, which the original head had
  never seen,
- expands the characterizer from 3 to 5 drone subtypes (Parrot Bebop
  and Parrot Mambo are net-new),
- trades ~1.5 pp on subtype accuracy for that expanded label set,
  with the per-class F1 stable for the previously-trained drones.

Visual confusion matrices: `models/confusion_binary.png`,
`models/confusion_subtype.png`. Full metric JSONs:
`models/metrics_binary.json`, `models/metrics_subtype.json`.

---

## Files

```
drone_detector.py           # Importable module exposing DroneDetector
detect_drone_cli.py         # Thin CLI wrapper for local testing
train.py                    # Training script (binary + subtype)
download_data.py            # ERAU dataset downloader
embed_extra_audio.py        # Raw WAV -> YAMNet embedding for extra datasets
evaluate_real_samples.py    # Runs the trained heads against a curated set of
                            #   real labeled drone WAVs (trained / untrained /
                            #   negative) and prints per-sample verdicts. Useful
                            #   smoke test after re-training.
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
data/                       # gitignored; populated by download_data.py + embed_extra_audio.py
```

---

## License

Code: MIT.
Datasets: ERAU is CC BY 4.0; saraalemadi/DroneAudioDataset and
mackenzie-jane/drone-visualization carry no explicit license file in the
upstream repos — cite the underlying conference / arXiv papers if you
redistribute downstream weights derived from them. The committed trained
weights are derivative works of all three datasets; attribute Embry-Riddle
Aeronautical University (ERAU dataset), Sara Al-Emadi et al. (IWCMC 2019),
and Wang et al. (arXiv:2509.04715) when redistributing.
