# Offline AudioSet (YAMNet) Classification + Visualization

This directory supports **offline audio event classification** using **YAMNet (AudioSet / 521 classes)** and provides lightweight **Streamlit dashboards** for stepping through predictions, visualizing category activity over time, and comparing uploaded annotation outputs against offline inference.

---

## What’s in this folder

- `offlineclassify.py` — CLI tool: audio file → `predictions.jsonl` / `predictions_full.npz`
- `viz_app.py` — Streamlit dashboard to explore classification results
- `dashboard_compare.py` — Streamlit dashboard to:
  - run offline inference on uploaded audio
  - inspect uploaded annotation files
  - compare uploaded annotations against offline inference
  - handle stereo audio in multiple modes
- `yamnet.tflite` — YAMNet TensorFlow Lite model
- `yamnet_class_map.csv` — Label map (521 AudioSet classes for YAMNet)
- `predictions.jsonl` — Example/output top-k predictions file
- `predictions_full.npz` — Optional full-vector output (`t_sec`, `scores`, `labels`)
- `Recording.wav` — Example audio file
- `requirements.txt` — Python dependencies
- `label_extractor.py` — Optional helper for extracting labels from TFLite metadata
- `metadata.json` — Optional project metadata / documentation

---

## Setup

### 1) Create an environment (recommended)

Using conda:

~~~bash
conda create -n yamnet_offline python=3.11 -y
conda activate yamnet_offline
pip install -r requirements.txt
~~~

**macOS note:** if `soundfile` fails to install, install `libsndfile` first:

~~~bash
brew install libsndfile
pip install soundfile
~~~

**macOS note:** `ffmpeg` may be required to process `m4a`, `mp3`, and `mp4` audio files:

~~~bash
brew install ffmpeg
~~~

**macOS note:** TensorFlow may require Apple-specific packages.  
If standard `tensorflow` fails to install, remove it from `requirements.txt` and install:

~~~bash
pip install tensorflow-macos tensorflow-metal
~~~

---

## 2) Running Offline Classification (CLI)

Run the classifier and write results to `predictions.jsonl`:

~~~bash
python offlineclassify.py --audio Recording.wav --sample_hz 1.0 --top_k 5
~~~

### CLI arguments

- `--audio` (required): Path to an audio file (`.wav`, `.flac`, etc.)
- `--sample_hz` (default: `1.0`): How often to emit predictions (frames per second)
  - Example: `2.0` → prediction every `0.5s`, `0.5` → prediction every `2s`
- `--top_k` (default: `5`): Number of top labels to save per frame
- `--window_s` (default: `0.975`): Window size (seconds) used for each prediction
- `--auto_window`: Override the window size using the model input length
- `--avg_frames` (default: `1`): Temporal smoothing (average over last `N` frames)
- `--out_json` (default: `predictions.jsonl`): Output path for top-k JSONL predictions
- `--out_npz` (default: `predictions_full.npz`): Output path for full score vectors
- `--model` (default: `yamnet.tflite`): Model path
- `--labels` (default: `yamnet_class_map.csv`): Label map path

### Example with smoothing and full-vector output

~~~bash
python offlineclassify.py \
  --audio Recording.wav \
  --sample_hz 1.0 \
  --top_k 10 \
  --avg_frames 3 \
  --auto_window \
  --out_json predictions.jsonl \
  --out_npz predictions_full.npz
~~~

### Output format

The default top-k output is **JSON Lines** (one JSON object per time step).

Each line looks like:

~~~json
{
  "t_sec": 12.487,
  "top": [
    {"label": "Speech", "score": 0.83},
    {"label": "Music", "score": 0.12},
    {"label": "Vehicle", "score": 0.04}
  ]
}
~~~

- `t_sec` is the center time of the analysis window
- `top` is the ranked list of top-k labels and their scores

If `--out_npz` is enabled, the CLI also writes a compressed `.npz` file containing:

- `t_sec` — frame-center timestamps
- `scores` — full prediction matrix of shape `[T, 521]`
- `labels` — YAMNet class labels

---

## 3) Visualizing Results (`viz_app.py`)

A localhost dashboard is provided via Streamlit for exploring saved results.

Run it with:

~~~bash
streamlit run viz_app.py
~~~

### What it includes

- Top-1 label timeline over time
- Confidence threshold slider
- Collapsed event-segment table (contiguous runs)
- Per-frame top-k inspection at selected timestamps

---

## 4) Upload + Compare Dashboard (`dashboard_compare.py`)

This dashboard supports three workflows:

1. Upload an audio file → run offline classification → step through predictions
2. Upload an annotations file (`JSONL` / `JSON` / `CSV` / `NPZ`) → inspect predictions
3. Upload audio + annotations → run offline classification and compare uploaded annotations against offline output side by side

Run it with:

~~~bash
streamlit run dashboard_compare.py
~~~

### Main features

- Upload audio and run offline inference directly in the dashboard
- Upload offline or real-time annotation files for inspection
- Compare uploaded annotations against offline inference frame by frame
- Confidence filtering for cleaner inspection
- Live playback-synced frame inspection (when supported by your Streamlit version)
- Download offline predictions as `JSONL` or full-vector `NPZ`

### Stereo audio support in `dashboard_compare.py`

The compare dashboard supports multiple stereo handling modes:

- **Mono mixdown** — mix channels to mono before classification
- **Stereo: LEFT/RIGHT separate** — classify each channel independently and output both
- **Stereo: LEFT/RIGHT average** — classify each channel independently and average the per-class scores for a combined output

When stereo outputs are available, the dashboard can also show:

- LEFT vs RIGHT side-by-side inspection
- LEFT vs RIGHT agreement metrics
- channel-specific output downloads
- stereo-aware comparisons against uploaded annotations

### Comparison metrics

The compare dashboard includes metrics such as:

- Frames compared
- Average overlap in top-k labels
- Top-1 match rate
- Top-2 match rate
- Average Jaccard similarity over top-k label sets
- Average absolute score difference on common labels

In stereo mode, the dashboard can also report **LEFT vs RIGHT channel agreement** metrics for the offline outputs.

### Label timeline visualization

The compare dashboard also includes label-over-time visualizations to help inspect how categories appear across the audio:

- label presence over time
- score-based or binary-presence views
- side-by-side timeline views when comparing two outputs

This is useful for spotting persistent labels, label transitions, and channel-specific events in stereo recordings.

---

## Supported audio formats

Recommended: `wav`

Also supported (depending on your local install / ffmpeg availability):

- `mp3`
- `mp4`
- `m4a`
- `flac`
- `ogg`
- `aiff`
- `aif`
- `aifc`

**Note:** `mp3` / `m4a` / `mp4` support typically requires `ffmpeg`.

On macOS:

~~~bash
brew install ffmpeg
~~~

---

## Supported annotation formats

The dashboard accepts:

- `JSONL` in the same format as `offlineclassify.py` output
- `JSON` containing a list of predictions or `{ "predictions": [...] }`
- `CSV` in several supported schemas
- `NPZ` full-vector files containing `t_sec`, `scores`, and `labels`

### Recommended annotation schema (`JSONL`)

One JSON object per line:

~~~json
{"t_sec":0.975,"top":[{"label":"Speech","score":0.83},{"label":"Music","score":0.12}]}
{"t_sec":1.975,"top":[{"label":"Speech","score":0.79},{"label":"Music","score":0.15}]}
~~~

### CSV compatibility

The dashboard supports several CSV layouts, including:

- long-form CSV: `t_sec, rank, label, score`
- minimal CSV: `t_sec, label, score`
- wide top-k CSV: `t_sec, label_1, score_1, ...`
- Android real-time wide exports with capture metadata and one column per label

Android-style wide CSV files are automatically converted into the internal long-form schema for comparison.

---

## Additional notes

- YAMNet expects **16 kHz** waveform input; audio is resampled accordingly before inference
- `dashboard_compare.py` can handle stereo audio using selectable channel-handling modes
- `metadata.json` is not required for inference
- `yamnet_class_map.csv` should match the model’s class ordering exactly
- `predictions.jsonl` stores only the saved **top-k** labels per frame
- `predictions_full.npz` stores the full **521-dimensional** score vectors when enabled