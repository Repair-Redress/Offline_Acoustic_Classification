# Semantic Pipeline

## Purpose

The semantic pipeline turns an audio recording plus contextual metadata into a higher-level interpretation of the recording. It still uses YAMNet for offline acoustic evidence, but it does not stop at frame-level AudioSet labels. The pipeline combines acoustic predictions with metadata such as timestamp, location, weather, scene hints, and manual notes, then produces:

- a clip-level semantic scene summary
- dominant non-silence sound labels
- salient audio chunks with start/end times
- a narrative explanation of the recording as a whole
- an auditable record of which context signals affected the scores

The goal is to answer “what is happening in this recording, in context?” rather than only “which YAMNet classes scored highest?”

## How To Use The Page

Open **Semantic Classification** from the app navigation.

1. Upload an audio file.
   - Supported formats include `.wav`, `.mp3`, `.m4a`, `.flac`, and `.ogg`.
   - The audio is converted to mono, resampled to 16 kHz, and classified locally with YAMNet.

2. Upload a labels / metadata file.
   - A standard YAMNet class-map CSV or JSON can provide the 521 label names.
   - Android-style wide CSV exports are also supported when they contain columns such as `Capture location`, `Capture time`, and one column per YAMNet class.
   - The page can parse semicolon-delimited files where labels contain commas.
   - If the file includes latitude, longitude, and timestamp metadata, those values are used for automatic context enrichment.

3. Optionally expand **Model parameters**.
   - `Sample rate (frames/sec)` controls how often YAMNet is evaluated over the audio.
   - `Temporal averaging` smooths predictions over recent frames.
   - `Top-k labels` controls the number of labels shown in summaries.
   - `Use model-derived window length` keeps the analysis window aligned with the loaded TFLite model.

4. Optionally expand **Context settings**.
   - Automatic context fetching is enabled by default.
   - The page uses latitude, longitude, and timestamp from the uploaded metadata when available.
   - You can override those values or add manual scene, location, weather, recording, and tag context.
   - Reverse geocoding uses OpenStreetMap Nominatim.
   - Weather context uses Open-Meteo, including historical weather when the timestamp is in the past.

5. Click **Run Semantic Classification**.
   - The app runs acoustic inference.
   - It fetches any available external context.
   - It applies rule-based context adjustment.
   - It displays the semantic output and salient chunks.

## Main Page Outputs

The page shows a compact semantic result first:

- **Semantic scene**: a short scene-level interpretation such as `urban roadside soundscape`, `outdoor natural ambience`, or `low-confidence ambient recording`.
- **Duration**: the analyzed audio duration.
- **Scene confidence**: a bounded score based on non-silence evidence, audio energy, and near-silent fraction.
- **Narrative**: a sentence-level explanation comparing raw acoustic evidence, context-adjusted evidence, scene tags, and context cues.

The page then shows:

- **Dominant non-silence labels**: the strongest labels after filtering low-value scene labels such as `Silence` when they would not help semantic interpretation.
- **Salient chunks**: relevant time spans with start time, end time, descriptive tags, description, peak score, mean score, and frame count.
- **Top rule impacts**: the largest boosts or suppressions applied by context rules.
- **Top-1 timeline preview**: a quick frame-level comparison of raw and adjusted top labels.
- Download buttons for JSONL predictions, NPZ matrices, and the semantic summary JSON.

## Pipeline Stages

### 1. Audio Inference

The app loads the uploaded audio, converts it to mono, rescales it if needed, resamples to 16 kHz, and splits it into overlapping windows. Each window is passed through the YAMNet TFLite model to produce frame-level class scores over the 521 AudioSet labels.

The raw YAMNet scores are preserved. Context adjustment creates a second score matrix, but it does not overwrite the raw acoustic evidence.

### 2. Metadata And Context Collection

The pipeline collects context from three places:

- uploaded labels / metadata file
- automatic API enrichment
- optional manual context fields

Typical metadata includes:

```json
{
  "location": {
    "latitude": 42.6109233,
    "longitude": -76.6360494,
    "display_name": "Lat: 42.6109233, Lon: -76.6360494"
  },
  "recording": {
    "timestamp": "2/13/2026_11:53:48:916"
  }
}
```

The app understands ISO timestamps and Android-style capture timestamps such as:

```text
2/13/2026_11:53:48:916
```

When coordinates are available, the context enrichment layer can add:

- reverse-geocoded place details
- inferred scene tags such as `outdoor`, `urban`, or `roadside`
- weather summary, condition, precipitation, wind, cloud cover, and matched hour

### 3. Rule-Based Context Adjustment

Context adjustment is performed by `build_context_adjustment()` in `semantic_pipeline.py`.
The adjustment groups and rules are stored in `context_rules.json`, and the dashboard includes a **Context Rules** page for editing that structured repo-backed file.

The process is intentionally interpretable:

1. Extract context signals.
   - The pipeline reads tags, scene fields, location fields, weather fields, and recording notes.
   - Each signal becomes a normalized text value with a source and confidence.
   - Examples: `outdoor`, `roadside`, `rain`, `windy`, `urban`, `office`, `industrial`.

2. Build semantic label groups.
   - YAMNet labels are grouped using pattern sets such as `traffic`, `road`, `nature`, `rain`, `wind`, `office`, `construction`, `industrial`, `speech`, and `residential`.
   - A group is not a new model class. It is a set of YAMNet label indices matching text patterns.

3. Match context rules.
   - Each rule has trigger signals and one or more boost/suppress actions.
   - Example: rainy weather boosts rain/water/wind-related labels.
   - Example: urban scene boosts traffic/road/crowd/construction labels and suppresses nature labels.
   - Example: outdoor scene boosts nature/traffic/wind labels and suppresses indoor room labels.

4. Create an adjustment vector.
   - The pipeline starts with weight `1.0` for every YAMNet label.
   - Matching rules multiply selected label weights up or down.
   - Rule confidence controls how strongly each factor is applied.
   - Final weights are clipped to a conservative range so context can guide the model without fully replacing acoustic evidence.

5. Apply adjusted scores.
   - Create an adjusted score matrix by applying adjustment vector
   - Scores are clipped to `[0.0, 1.0]`.
   - Raw scores remain available for comparison.

The summary includes **Top rule impacts**, which lists the largest label-level adjustments, the rule name, whether it was a boost or suppression, the label affected, the weight, and the matched context terms.
Rule edits saved from the dashboard apply to future semantic classification runs after validation.

## Semantic Summary

After scoring, `build_semantic_summary()` aggregates the recording:

- computes dominant adjusted labels over time
- computes dominant non-silence labels
- chooses salient labels from the strongest non-silence evidence
- detects contiguous time spans where those labels exceed a dynamic threshold
- infers scene tags from context and labels
- computes scene confidence from non-silence strength, top label strength, RMS, and near-silent fraction
- chooses a semantic scene label
- writes a narrative explanation

The salient chunks are sorted by peak score and include both timing and score evidence. These chunks are intended to identify the portions of the recording most relevant to the semantic interpretation.

## Why Raw And Adjusted Scores Both Matter

Context is useful, but it can be wrong or incomplete. For that reason, the pipeline keeps raw acoustic output and context-adjusted output side by side.

This makes it possible to inspect cases where:

- context strengthened a plausible event
- context suppressed a contradictory label
- raw YAMNet evidence disagreed with metadata
- the recording was too quiet for confident interpretation

## Current Limitations

- The context rules are hand-built in `context_rules.json` and should be tuned as more recordings are reviewed.
- API enrichment depends on available coordinates, timestamp quality, and network access.
- Weather and reverse-geocoding add context, not ground truth.
- YAMNet can still produce noisy labels for quiet, clipped, or ambiguous recordings.
- The pipeline is not a retrained scene model; it is an interpretable context-aware layer on top of YAMNet.

---

## Complete Rule-Based Adjustment Guide

This section is an exact guide to how `build_context_adjustment()` computes context-adjusted YAMNet scores.

### Complete Rule Table

| Rule | Trigger signals | Boosts | Suppresses | Direct label suppressions |
|---|---|---|---|---|
| `rainy_weather` | `rain`, `drizzle`, `storm`, `wet`, `precipitation`, `weathercode_rain` | `rain: 1.35`, `water: 1.18`, `wind: 1.10` | `construction: 0.94` | none |
| `windy_weather` | `windy`, `gust`, `wind`, `weathercode_wind` | `wind: 1.30`, `nature: 1.08`, `traffic: 1.05` | none | none |
| `urban_scene` | `urban`, `city`, `downtown`, `road`, `street`, `avenue`, `traffic corridor` | `traffic: 1.30`, `road: 1.18`, `crowd: 1.08`, `construction: 1.10` | `nature: 0.88` | none |
| `rural_scene` | `rural`, `farmland`, `countryside`, `village`, `field` | `nature: 1.28`, `animals: 1.12`, `wind: 1.06` | `traffic: 0.82`, `construction: 0.90` | none |
| `indoor_scene` | `indoor`, `indoors`, `inside`, `room`, `hallway` | `indoors: 1.22`, `office: 1.08`, `speech: 1.04` | `birds: 0.72`, `wind: 0.80`, `rain: 0.84` | none |
| `office_scene` | `office`, `meeting room`, `workplace`, `lab`, `classroom` | `office: 1.35`, `speech: 1.08`, `indoors: 1.08` | `nature: 0.82` | none |
| `outdoor_scene` | `outdoor`, `outdoors`, `outside`, `open air` | `nature: 1.18`, `traffic: 1.06`, `wind: 1.08` | `office: 0.82` | `inside, small room: 0.58`, `inside, large room or hall: 0.62`, `reverberation: 0.80` |
| `industrial_scene` | `industrial`, `factory`, `warehouse`, `plant`, `machinery` | `industrial: 1.35`, `construction: 1.16`, `traffic: 1.05` | `birds: 0.84` | none |
| `residential_scene` | `residential`, `home`, `apartment`, `neighborhood` | `residential: 1.28`, `speech: 1.06`, `animals: 1.05` | `industrial: 0.86` | none |

### Inputs And Outputs

The adjustment step receives the ordered YAMNet `labels` list and the merged `context` dictionary. It returns:

- `adjustment_vector`: one multiplicative weight per YAMNet label
- `impacts`: an audit trail of every label-level boost or suppression
- `context_signals`: normalized context values considered by the rules

The raw YAMNet score matrix is adjusted later with:

```text
adjusted_scores = clip(raw_scores * adjustment_vector, 0.0, 1.0)
```

Raw scores are preserved.

### Text Normalization

All label and context matching uses this normalization:

- convert to string
- trim whitespace
- lowercase
- replace underscores with spaces
- collapse repeated whitespace

Example:

```text
"Traffic_Corridor" -> "traffic corridor"
```

### Context Signal Extraction

The pipeline extracts signals from these context fields:

| Context source | Fields read | Confidence |
|---|---|---:|
| `tags`, `scene_tags`, `manual_tags` | every tag value | `1.00` |
| `scene` | `environment`, `setting`, `place_type`, `indoor_outdoor`, `land_use` | `0.95` |
| `location` | `display_name`, `name`, `category`, `type`, `road`, `neighbourhood`, `suburb`, `city` | `0.80` |
| `weather` | `summary`, `description`, `weathercode_label`, `condition` | `0.95` |
| `recording` | `device`, `notes`, `source` | `0.60` |

Numeric weather fields create additional signals:

| Weather field | Condition | Added signal | Confidence |
|---|---:|---|---:|
| `precipitation_mm` | `> 0` | `precipitation` | `min(1.0, 0.5 + value / 10.0)` |
| `rain_mm` | `> 0` | `rain` | `min(1.0, 0.5 + value / 10.0)` |
| `windspeed_10m` | `>= 10` | `wind` | `min(1.0, value / 40.0)` |
| `windgusts_10m` | `>= 10` | `gust` | `min(1.0, value / 40.0)` |
| `cloudcover` | `>= 75` | `cloudy` | `min(1.0, value / 100.0)` |

### Label Groups

YAMNet labels are assigned to semantic groups by substring matching against normalized label text. A single label may belong to multiple groups.

| Group | Matching terms |
|---|---|
| `speech` | `speech`, `conversation`, `narration`, `whisper`, `babbling`, `shout`, `screaming` |
| `music` | `music`, `singing`, `choir`, `piano`, `guitar`, `drum`, `musical instrument` |
| `birds` | `bird`, `chirp`, `tweet`, `owl`, `crow`, `pigeon` |
| `wind` | `wind`, `rustling leaves` |
| `rain` | `rain`, `thunderstorm`, `thunder`, `raindrop` |
| `water` | `water`, `stream`, `ocean`, `waves`, `pour`, `drip`, `rain` |
| `traffic` | `car`, `bus`, `motor vehicle`, `truck`, `motorcycle`, `traffic`, `engine`, `vehicle horn`, `car horn`, `siren` |
| `road` | `vehicle horn`, `car horn`, `honking`, `siren`, `brake`, `skidding`, `traffic noise` |
| `construction` | `jackhammer`, `drill`, `hammer`, `power tool`, `construction`, `chainsaw` |
| `industrial` | `machine`, `engine`, `compressor`, `mechanical fan`, `tools` |
| `indoors` | `typing`, `printer`, `keyboard`, `clicking`, `door`, `drawer`, `cutlery`, `dishes`, `speech` |
| `office` | `typing`, `clicking`, `printer`, `speech`, `telephone`, `computer keyboard` |
| `residential` | `dog`, `cat`, `speech`, `television`, `vacuum cleaner`, `doorbell` |
| `nature` | `bird`, `insect`, `wind`, `water`, `frog`, `cricket`, `rain` |
| `animals` | `dog`, `cat`, `bird`, `frog`, `insect`, `livestock` |
| `crowd` | `crowd`, `cheering`, `clapping`, `applause`, `speech` |

### Rule Matching

Every adjustment rule has trigger signals. A rule fires when any trigger token appears inside any normalized context signal.

When a rule fires:

- `rule_conf` is the maximum confidence among the matched context signals.
- `matched_terms` records the matched context signal values.
- boosts and suppressions are applied to every YAMNet label in the affected label group.

### Adjustment Formulas

All label weights start at `1.0`.

Boost formula:

```text
effective = 1.0 + (factor - 1.0) * rule_conf
weight[label] *= effective
```

Suppression formula:

```text
effective = 1.0 - (1.0 - factor) * rule_conf
weight[label] *= effective
```

If no context signals exist, the function returns the unmodified all-ones vector.

If context signals exist, final weights are clipped:

```text
weight[label] = clip(weight[label], 0.3, 2.5)
```

### Audit Trail

Every rule action creates a `RuleImpact` entry:

| Field | Meaning |
|---|---|
| `rule` | rule that fired |
| `source` | `boost` or `suppress` |
| `label` | YAMNet label affected |
| `weight` | effective multiplier for that single action |
| `matched_terms` | normalized context signals that triggered the rule |

The UI table **Top adjustments** shows the largest impacts sorted by distance from neutral weight `1.0`.

### Worked Example

Suppose context contains:

```json
{
  "tags": ["outdoor", "urban", "roadside"],
  "weather": {
    "summary": "overcast and windy",
    "windspeed_10m": 18.5
  }
}
```

The extracted signals include:

- `outdoor`, confidence `1.00`
- `urban`, confidence `1.00`
- `roadside`, confidence `1.00`
- `overcast and windy`, confidence `0.95`
- numeric `wind`, confidence `min(1.0, 18.5 / 40.0) = 0.4625`

Rules that can fire include:

- `outdoor_scene`, because `outdoor` matches
- `urban_scene`, because `urban` matches
- `windy_weather`, because `windy` and/or `wind` match

For a YAMNet label in the `traffic` group:

```text
urban_scene traffic boost:
effective = 1.0 + (1.30 - 1.0) * 1.00 = 1.30

outdoor_scene traffic boost:
effective = 1.0 + (1.06 - 1.0) * 1.00 = 1.06

windy_weather traffic boost:
effective = 1.0 + (1.05 - 1.0) * 0.95 = 1.0475

combined weight before clipping:
1.30 * 1.06 * 1.0475 = 1.443455
```

For a YAMNet label in the `nature` group:

```text
urban_scene nature suppression:
effective = 1.0 - (1.0 - 0.88) * 1.00 = 0.88

outdoor_scene nature boost:
effective = 1.0 + (1.18 - 1.0) * 1.00 = 1.18

windy_weather nature boost:
effective = 1.0 + (1.08 - 1.0) * 0.95 = 1.076

combined weight before clipping:
0.88 * 1.18 * 1.076 = 1.1177984
```

For the direct label `inside, small room` when outdoor context is present:

```text
outdoor_scene direct suppression:
effective = 1.0 - (1.0 - 0.58) * 1.00 = 0.58
```

This is why the adjusted output can favor traffic and outdoor ambience while reducing contradictory indoor-room evidence, without discarding the original acoustic scores.
