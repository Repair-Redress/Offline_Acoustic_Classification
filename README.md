# Offline AudioSet (YAMNet) Classification + Visualization

This directory supports **offline audio event classification** using **YAMNet (AudioSet / 521 classes)** and provides several **Streamlit apps** for exploring saved results, stepping through predictions, and comparing uploaded annotation outputs against offline inference.

The app suite now uses a **multipage Streamlit navigation setup** with a root entrypoint (`dashboard_app.py`) and page files under `pages/`, so page titles in the sidebar can be controlled without renaming the underlying Python files.

---

## What’s in this folder

- `dashboard_app.py` — **main Streamlit entrypoint** using Streamlit's navigation API
- `pages/compare_app.py` — comparison dashboard page to:
  - run offline inference on uploaded audio
  - inspect uploaded annotation files
  - compare uploaded annotations against offline inference
  - handle stereo audio in multiple modes
- `pages/help_app.py` — help/reference page for:
  - metric definitions
  - parameter descriptions
  - model / ontology notes
  - supported input formats
- `pages/viz_app.py` — visualization page for exploring classification results
- `offlineclassify.py` — CLI tool: audio file → `predictions.jsonl` / `predictions_full.npz`
- `yamnet.tflite` — YAMNet TensorFlow Lite model
- `yamnet_class_map.csv` — Label map (521 AudioSet classes for YAMNet)
- `requirements.txt` — Python dependencies
- `label_extractor.py` — optional helper for extracting labels from TFLite metadata
- `metadata.json` — optional project metadata / documentation

---

## Setup

### 1) Create an environment

Using conda:

~~~bash
conda create -n yamnet_offline python=3.11 -y
conda activate yamnet_offline
pip install -r requirements.txt
~~~

### macOS notes

If `soundfile` fails to install:

~~~bash
brew install libsndfile
pip install soundfile
~~~

If you want support for `mp3`, `m4a`, and `mp4` audio:

~~~bash
brew install ffmpeg
~~~

If standard TensorFlow installation fails on Apple Silicon, you may need:

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

- `--audio` (required): path to an audio file (`.wav`, `.flac`, etc.)
- `--sample_hz` (default: `1.0`): how often to emit predictions (frames per second)
  - `2.0` → prediction every `0.5s`
  - `0.5` → prediction every `2s`
- `--top_k` (default: `5`): number of top labels to save per frame
- `--window_s` (default: `0.975`): window size (seconds) used for each prediction
- `--auto_window`: override the window size using the model input length
- `--avg_frames` (default: `1`): temporal smoothing over the last `N` frames
- `--out_json` (default: `predictions.jsonl`): output path for top-k JSONL predictions
- `--out_npz` (default: `predictions_full.npz`): output path for full score vectors
- `--model` (default: `yamnet.tflite`): model path
- `--labels` (default: `yamnet_class_map.csv`): label map path

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

Example:

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

## 3) Running the Streamlit app suite

The app now uses a **main entrypoint** with explicit page navigation.

Run it with:

~~~bash
streamlit run dashboard_app.py
~~~

This launches a multipage app with sidebar navigation.

### Pages

#### Comparison Dashboard
Implemented in `pages/compare_app.py`.

Supports three workflows:

1. Upload an audio file → run offline classification → step through predictions
2. Upload an annotations file (`JSONL` / `JSON` / `CSV` / `NPZ`) → inspect predictions
3. Upload audio + annotations → run offline classification and compare uploaded annotations against offline output side by side

Main features:

- upload audio and run offline inference directly in the app
- upload offline or real-time annotation files for inspection
- compare uploaded annotations against offline inference frame by frame
- confidence filtering for cleaner inspection
- live playback-synced frame inspection (when supported by your Streamlit version)
- download offline predictions as `JSONL` or full-vector `NPZ`
- reduced UI clutter through collapsible sections and tabbed comparison views

Stereo support includes:

- **Mono mixdown** — mix channels to mono before classification
- **Stereo: LEFT/RIGHT separate** — classify each channel independently
- **Stereo: LEFT/RIGHT average** — classify each channel independently and average per-class scores

When stereo outputs are available, the app can show:

- LEFT vs RIGHT side-by-side inspection
- LEFT vs RIGHT agreement metrics
- channel-specific output downloads
- stereo-aware comparisons against uploaded annotations

#### Help / Reference
Implemented in `pages/help_app.py`.

Provides documentation for:

- metric definitions
- parameter meanings
- YAMNet / AudioSet background
- supported input formats
- notes on stereo handling and comparison behavior

#### Visualization Page
Implemented in `pages/viz_app.py`.

Provides a lightweight interface for exploring saved classification results, including:

- top-label inspection over time
- confidence filtering
- event/activity exploration
- per-frame top-k inspection

---

## 4) Comparison metrics

The comparison dashboard includes metrics such as:

- Frames compared
- Average overlap in top-k labels
- Top-1 match rate
- Top-2 match rate
- Average Jaccard similarity over top-k label sets
- Average absolute score difference on common labels

In stereo mode, the app can also report **LEFT vs RIGHT channel agreement** metrics for offline outputs.

The help page includes definitions and interpretation notes for these metrics.

---

## 5) Label timeline visualization

The comparison and visualization pages include label-over-time visualizations to help inspect how categories appear across the audio:

- label presence over time
- score-based or binary-presence views
- side-by-side timeline views when comparing two outputs

This is useful for spotting:

- persistent labels
- label transitions
- channel-specific events in stereo recordings
- noisy or inconsistent outputs

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

The comparison dashboard accepts:

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

## Upload size configuration

File upload size limit is handled through Streamlit's config file:

`.streamlit/config.toml`

~~~toml
[server]
maxUploadSize = 1024
~~~

This version sets the max upload size at 1024MB.

---

## Additional notes

- YAMNet expects **16 kHz** waveform input; audio is resampled accordingly before inference
- uploaded audio is decoded and cached where possible to reduce repeated load cost
- uploaded annotation parsing is cached where possible to improve responsiveness
- stale results are cleared when uploaded files change, so the dashboard stays in sync with current inputs
- `yamnet_class_map.csv` should match the model’s class ordering exactly
- `predictions.jsonl` stores only the saved **top-k** labels per frame
- `predictions_full.npz` stores the full **521-dimensional** score vectors when enabled
- `metadata.json` is not required for inference