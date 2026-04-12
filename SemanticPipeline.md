# Semantic Pipeline

## Overview

`semantic_pipeline.py` is a context-aware extension of the offline YAMNet classification workflow in this project.

Instead of only producing frame-level AudioSet labels, it:

- runs offline acoustic inference on an audio recording
- ingests optional contextual metadata from a `context.json` file
- applies rule-based context weighting to the raw class scores
- aggregates results into higher-level semantic outputs
- exports both machine-readable predictions and a human-readable summary

This pipeline is designed to support semantic understanding of a recording rather than only top-k sound event detection.

## What The Pipeline Does

The pipeline has five main stages:

1. Audio loading
   - Loads audio from common formats.
   - Uses `soundfile` first.
   - Falls back to `pydub`/`ffmpeg` for formats like `.m4a`.

2. Acoustic inference
   - Converts audio to mono.
   - Resamples to 16 kHz.
   - Frames the waveform into YAMNet-sized windows.
   - Runs TFLite inference and produces frame-level class scores.

3. Context ingestion
   - Optionally loads structured metadata from `context.json`.
   - Reads scene tags, location hints, weather hints, and recording notes.

4. Context fusion
   - Builds an adjustment vector over YAMNet labels.
   - Boosts or suppresses classes based on the provided context.
   - Keeps the raw model output and the context-adjusted output separate.

5. Semantic summarization
   - Aggregates predictions across time.
   - Filters out low-value labels when building scene summaries.
   - Extracts salient segments.
   - Produces a clip-level semantic interpretation and confidence signal.

## Main Outputs

Running the pipeline produces three files by default:

- `*_semantic_predictions.jsonl`
  - Per-frame predictions.
  - Includes `raw_top` and `adjusted_top` labels for each timestamp.

- `*_semantic_predictions.npz`
  - Full score matrices.
  - Includes:
    - `t_sec`
    - `raw_scores`
    - `adjusted_scores`
    - `labels`
    - `adjustment_vector`

- `*_semantic_summary.json`
  - Clip-level semantic interpretation.
  - Includes:
    - dominant labels
    - dominant non-silence labels
    - salient segments
    - semantic scene label
    - narrative explanation
    - context signals used
    - top rule impacts
    - scene confidence
    - audio stats

## Example Command

```bash
python semantic_pipeline.py \
  --audio ../Recordings/example.m4a \
  --context_json sample_context.json \
  --auto_window \
  --top_k 5
```

## Context JSON

The context file is optional but recommended when semantic interpretation matters.

Typical fields:

```json
{
  "tags": ["outdoor", "urban", "roadside"],
  "scene": {
    "environment": "urban",
    "setting": "roadside",
    "indoor_outdoor": "outdoor"
  },
  "location": {
    "display_name": "Avenida Carrera 7, Chapinero, Bogota, Colombia",
    "category": "highway",
    "city": "Bogota"
  },
  "weather": {
    "summary": "light rain and windy",
    "condition": "rain",
    "precipitation_mm": 1.8,
    "windspeed_10m": 18.5
  },
  "recording": {
    "device": "Android phone",
    "notes": "Captured near a busy road during afternoon traffic."
  }
}
```

## Important Design Choices

### Raw and adjusted scores are both preserved

The pipeline does not overwrite the acoustic model output.

It keeps:

- raw YAMNet scores
- context-adjusted scores

This makes the system easier to inspect and audit.

### The context layer is rule-based

At this stage, semantic fusion is not done with a large language model in the inference loop.

Instead, the pipeline uses:

- structured context signals
- interpretable rule-based boosts and suppressions
- explicit adjustment weights

This was chosen to keep the system offline-friendly and easier to debug.

### The summary is intentionally conservative

The pipeline now tries to avoid overclaiming.

It:

- filters out low-value labels such as `Silence` for semantic scene summaries
- downweights contradictory indoor labels for strong outdoor context
- computes a scene confidence score
- can fall back to a low-confidence interpretation when evidence is weak

## Current Limitations

- YAMNet predictions can still be noisy for quiet or ambiguous recordings.
- Context priors are hand-built and still need tuning.
- Some classes in AudioSet remain semantically awkward at scene level.
- The final semantic summary is only as strong as the acoustic evidence.
- This is not yet a retrained or fine-tuned model.

## Recommended Use

Use this pipeline when you want:

- offline sound event classification
- context-aware reinterpretation of acoustic labels
- segment-level event inspection
- a higher-level semantic summary of a recording

Use the outputs carefully when:

- the recording is very quiet
- the top class is mostly `Silence`
- non-silence confidence is low
- context and audio evidence disagree strongly

## Suggested Next Steps

- integrate semantic outputs into the Streamlit comparison/visualization tools
- keep tuning context rules on more recordings
- add better scene-level confidence calibration
- create evaluation examples with human-checked expected scenes
- explore fine-tuning or retraining only after the rule-based baseline is stable
