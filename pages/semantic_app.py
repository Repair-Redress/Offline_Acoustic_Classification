import io
import csv
import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from context_enrichment import enrich_context, merge_context
from semantic_pipeline import (
    Interpreter,
    build_context_adjustment,
    build_semantic_summary,
    compute_audio_stats,
    get_model_expected_len,
    load_labels,
    top_k_labels,
    to_mono,
    classify_waveform,
)


st.set_page_config(page_title="Semantic Classification", layout="wide")
st.title("Semantic Classification Dashboard")
st.write("Upload an audio file, add optional metadata, and generate a context-aware semantic summary.")
st.markdown(
    """
    <style>
    .semantic-summary-grid {
        display: grid;
        grid-template-columns: minmax(280px, 1.7fr) minmax(180px, 0.75fr) minmax(180px, 0.75fr);
        gap: 0.85rem;
        margin: 0.35rem 0 0.85rem 0;
    }
    .semantic-summary-card {
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-radius: 8px;
        padding: 1rem 1.05rem;
        background: rgba(128, 128, 128, 0.08);
        min-height: 118px;
    }
    .semantic-summary-label {
        font-size: 0.86rem;
        font-weight: 650;
        opacity: 0.72;
        margin-bottom: 0.45rem;
    }
    .semantic-summary-value {
        font-size: 1.85rem;
        line-height: 1.12;
        font-weight: 700;
        overflow-wrap: anywhere;
    }
    .semantic-summary-value.scene {
        font-size: 2rem;
    }
    .semantic-summary-subtext {
        margin-top: 0.45rem;
        font-size: 0.82rem;
        opacity: 0.66;
    }
    @media (max-width: 900px) {
        .semantic-summary-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_interpreter(model_path: str):
    if Interpreter is None:
        raise RuntimeError("Install tensorflow or tflite_runtime to run YAMNet inference.")
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


@st.cache_data(show_spinner=False)
def load_labels_cached(labels_path: str):
    return load_labels(labels_path)


@st.cache_data(show_spinner=False)
def enrich_context_cached(
    latitude: float | None,
    longitude: float | None,
    timestamp_text: str,
    metadata_json: str,
    include_geocoding: bool,
    include_weather: bool,
) -> dict:
    metadata = json.loads(metadata_json) if metadata_json else {}
    return enrich_context(
        latitude=latitude,
        longitude=longitude,
        timestamp_text=timestamp_text,
        metadata=metadata,
        include_geocoding=include_geocoding,
        include_weather=include_weather,
    )


def read_audio_from_upload(audio_bytes: bytes, filename: str) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        waveform, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
        return np.asarray(waveform, dtype=np.float32), int(sr)
    except Exception:
        pass

    if AudioSegment is None:
        raise RuntimeError("pydub is required for this audio format. Install requirements.txt or use WAV/FLAC input.")

    ext = Path(filename).suffix.lower().lstrip(".")
    if ext == "m4a":
        ext = "mp4"
    segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext or None)
    sr = int(segment.frame_rate)
    channels = int(segment.channels)
    raw = np.array(segment.get_array_of_samples())
    if channels > 1:
        raw = raw.reshape((-1, channels))
    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        waveform = raw.astype(np.float32) / max(abs(info.min), info.max)
    else:
        waveform = raw.astype(np.float32)
    return waveform, sr


def parse_metadata_upload(uploaded_file) -> dict:
    if uploaded_file is None:
        return {}
    try:
        return json.loads(uploaded_file.getvalue().decode("utf-8"))
    except Exception as exc:
        st.error(f"Metadata JSON could not be parsed: {exc}")
        return {}


def normalize_metadata_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def is_blank_metadata_value(value) -> bool:
    if value in ("", None):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def lookup_metadata_value(data: dict, *keys: str):
    if not isinstance(data, dict):
        return None
    normalized = {normalize_metadata_key(key): value for key, value in data.items()}
    for key in keys:
        value = normalized.get(normalize_metadata_key(key))
        if not is_blank_metadata_value(value):
            return value
    return None


def sniff_csv_delimiter(text: str) -> str:
    sample = text[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        if first_line.count(";") > first_line.count(","):
            return ";"
        return ","


def parse_location_coordinates(value) -> dict:
    if is_blank_metadata_value(value):
        return {}
    text = str(value)
    lat_match = re.search(r"\bLat(?:itude)?\s*:\s*([-+]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    lon_match = re.search(r"\b(?:Lon|Long|Longitude)\s*:\s*([-+]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    coordinates = {}
    if lat_match:
        coordinates["latitude"] = float(lat_match.group(1))
    if lon_match:
        coordinates["longitude"] = float(lon_match.group(1))
    return coordinates


TIMESTAMP_KEYS = (
    "timestamp",
    "capture_time",
    "capture time",
    "recorded_at",
    "recording_timestamp",
    "recording timestamp",
    "datetime",
    "date_time",
    "time",
)

LATITUDE_KEYS = (
    "latitude",
    "lat",
    "location_latitude",
    "location latitude",
    "location_lat",
    "location lat",
    "capture_latitude",
    "capture latitude",
    "gps_latitude",
    "gps lat",
)

LONGITUDE_KEYS = (
    "longitude",
    "long",
    "lon",
    "lng",
    "location_longitude",
    "location longitude",
    "location_long",
    "location long",
    "location_lon",
    "location lon",
    "location_lng",
    "location lng",
    "capture_longitude",
    "capture longitude",
    "gps_longitude",
    "gps long",
)

LOCATION_NAME_KEYS = (
    "capture_location",
    "capture location",
    "location",
    "display_name",
    "display name",
    "location_name",
    "location name",
)


def load_labels_from_upload(uploaded_file) -> list[str]:
    if uploaded_file is None:
        return load_labels_cached(labels_path)

    name = uploaded_file.name.lower()
    text = uploaded_file.getvalue().decode("utf-8")
    if name.endswith(".csv"):
        rows = list(csv.DictReader(io.StringIO(text), delimiter=sniff_csv_delimiter(text)))
        if not rows:
            raise ValueError("Labels CSV is empty.")

        fieldnames = list(rows[0].keys())
        if any(name in fieldnames for name in ("display_name", "name", "label")):
            labels = []
            for row in rows:
                labels.append(row.get("display_name") or row.get("name") or row.get("label") or row[list(row.keys())[-1]])
            return labels

        metadata_columns = {
            *(normalize_metadata_key(key) for key in TIMESTAMP_KEYS),
            *(normalize_metadata_key(key) for key in LATITUDE_KEYS),
            *(normalize_metadata_key(key) for key in LONGITUDE_KEYS),
            *(normalize_metadata_key(key) for key in LOCATION_NAME_KEYS),
        }
        label_columns = [col for col in fieldnames if col and normalize_metadata_key(col) not in metadata_columns]
        if label_columns:
            return label_columns

    if name.endswith(".json"):
        metadata = json.loads(text)
        candidates = [
            metadata.get("labels"),
            metadata.get("class_labels"),
            metadata.get("class_map"),
            metadata.get("classes"),
            (metadata.get("yamnet") or {}).get("labels"),
            (metadata.get("yamnet") or {}).get("class_map"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            if isinstance(candidate, list) and all(isinstance(x, str) for x in candidate):
                return candidate
            if isinstance(candidate, list) and all(isinstance(x, dict) for x in candidate):
                labels = [x.get("display_name") or x.get("name") or x.get("label") for x in candidate]
                if all(labels):
                    return labels
            if isinstance(candidate, dict):
                keys_are_intlike = all(str(k).isdigit() for k in candidate.keys())
                if keys_are_intlike:
                    return [candidate[str(i)] if str(i) in candidate else candidate[i] for i in range(len(candidate))]

    raise ValueError("Labels file must be a CSV class map or JSON metadata file containing labels.")


def metadata_from_label_file(uploaded_file) -> dict:
    if uploaded_file is None:
        return {}

    name = uploaded_file.name.lower()
    try:
        if name.endswith(".json"):
            data = json.loads(uploaded_file.getvalue().decode("utf-8"))
            return data if isinstance(data, dict) else {}

        if name.endswith(".csv"):
            raw = uploaded_file.getvalue()
            text = raw.decode("utf-8", errors="replace")
            df = pd.read_csv(io.BytesIO(raw), sep=sniff_csv_delimiter(text), nrows=1)
            if df.empty:
                return {}
            row = df.iloc[0].to_dict()
            metadata = {}
            timestamp = lookup_metadata_value(row, *TIMESTAMP_KEYS)
            display_name = lookup_metadata_value(row, *LOCATION_NAME_KEYS)
            latitude = lookup_metadata_value(row, *LATITUDE_KEYS)
            longitude = lookup_metadata_value(row, *LONGITUDE_KEYS)

            if timestamp is not None:
                metadata["timestamp"] = str(timestamp)
            if display_name is not None:
                metadata["location"] = {"display_name": str(display_name)}
                metadata["location"].update(parse_location_coordinates(display_name))
            if latitude is not None:
                metadata["latitude"] = latitude
            if longitude is not None:
                metadata["longitude"] = longitude
            return metadata
    except Exception:
        return {}

    return {}


def context_from_metadata(metadata: dict) -> dict:
    context = {}
    latitude = lookup_metadata_value(metadata, *LATITUDE_KEYS)
    longitude = lookup_metadata_value(metadata, *LONGITUDE_KEYS)
    display_name = lookup_metadata_value(metadata, *LOCATION_NAME_KEYS)
    timestamp = lookup_metadata_value(metadata, *TIMESTAMP_KEYS)

    location = {}
    if display_name is not None:
        location["display_name"] = str(display_name)
    if latitude is not None:
        location["latitude"] = latitude
    if longitude is not None:
        location["longitude"] = longitude
    nested_location = metadata.get("location") if isinstance(metadata.get("location"), dict) else {}
    context["location"] = {k: v for k, v in merge_context(location, nested_location).items() if not is_blank_metadata_value(v)}

    recording = {}
    nested_recording = metadata.get("recording") if isinstance(metadata.get("recording"), dict) else {}
    if timestamp is not None:
        recording["timestamp"] = str(timestamp)
    context["recording"] = {k: v for k, v in merge_context(recording, nested_recording).items() if not is_blank_metadata_value(v)}

    return {k: v for k, v in context.items() if v}


def first_present(data: dict, *keys: str):
    for key in keys:
        if key in data and data[key] not in ("", None):
            return data[key]
    return None


def nested_first_present(data: dict, section: str, *keys: str):
    nested = data.get(section)
    if isinstance(nested, dict):
        return first_present(nested, *keys)
    return None


def coerce_float(value) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def context_from_form(metadata: dict) -> tuple[dict, dict]:
    default_latitude = coerce_float(lookup_metadata_value(metadata, *LATITUDE_KEYS))
    if default_latitude is None and isinstance(metadata.get("location"), dict):
        default_latitude = coerce_float(lookup_metadata_value(metadata["location"], *LATITUDE_KEYS))

    default_longitude = coerce_float(lookup_metadata_value(metadata, *LONGITUDE_KEYS))
    if default_longitude is None and isinstance(metadata.get("location"), dict):
        default_longitude = coerce_float(lookup_metadata_value(metadata["location"], *LONGITUDE_KEYS))

    default_timestamp = str(
        lookup_metadata_value(metadata, *TIMESTAMP_KEYS)
        or (lookup_metadata_value(metadata["recording"], *TIMESTAMP_KEYS) if isinstance(metadata.get("recording"), dict) else None)
        or ""
    )

    with st.expander("Context settings", expanded=False):
        st.markdown("**Automatic context**")
        auto_col1, auto_col2, auto_col3 = st.columns([1.1, 1, 1])
        with auto_col1:
            auto_context = st.checkbox("Fetch context from location and time", value=True)
        with auto_col2:
            include_geocoding = st.checkbox("Reverse geocode location", value=True, disabled=not auto_context)
        with auto_col3:
            include_weather = st.checkbox("Fetch weather", value=True, disabled=not auto_context)

        loc_col1, loc_col2, loc_col3 = st.columns(3)
        with loc_col1:
            latitude_text = st.text_input(
                "Latitude",
                value="" if default_latitude is None else str(default_latitude),
                placeholder="4.6455",
                disabled=not auto_context,
            )
        with loc_col2:
            longitude_text = st.text_input(
                "Longitude",
                value="" if default_longitude is None else str(default_longitude),
                placeholder="-74.0613",
                disabled=not auto_context,
            )
        with loc_col3:
            timestamp_text = st.text_input(
                "Recording timestamp",
                value=default_timestamp,
                placeholder="2025-07-04T18:30:00-05:00",
                disabled=not auto_context,
            )

        st.markdown("**Manual context**")
        tags_raw = st.text_input("Tags", placeholder="outdoor, urban, roadside")

        scene_col, location_col, weather_col, recording_col = st.columns(4)
        with scene_col:
            st.markdown("Scene")
            environment = st.text_input("Environment", placeholder="urban")
            setting = st.text_input("Setting", placeholder="roadside")
            indoor_outdoor = st.selectbox("Indoor / outdoor", ["", "indoor", "outdoor"])
            place_type = st.text_input("Place type", placeholder="street")

        with location_col:
            st.markdown("Location")
            display_name = st.text_input("Display name", placeholder="Avenida Carrera 7, Bogota")
            category = st.text_input("Category", placeholder="highway")
            city = st.text_input("City", placeholder="Bogota")
            neighbourhood = st.text_input("Neighbourhood", placeholder="Chapinero")

        with weather_col:
            st.markdown("Weather")
            weather_summary = st.text_input("Weather summary", placeholder="light rain and windy")
            condition = st.text_input("Condition", placeholder="rain")
            precipitation_mm = st.number_input("Precipitation (mm)", min_value=0.0, value=0.0, step=0.1)
            windspeed_10m = st.number_input("Wind speed (m/s)", min_value=0.0, value=0.0, step=0.1)

        with recording_col:
            st.markdown("Recording")
            device = st.text_input("Device", placeholder="Android phone")
            notes = st.text_area("Notes", placeholder="Captured near a busy road during afternoon traffic.")

    tags = [item.strip() for item in tags_raw.split(",") if item.strip()]
    scene = {
        "environment": environment.strip(),
        "setting": setting.strip(),
        "indoor_outdoor": indoor_outdoor.strip(),
        "place_type": place_type.strip(),
    }
    location = {
        "display_name": display_name.strip(),
        "category": category.strip(),
        "city": city.strip(),
        "neighbourhood": neighbourhood.strip(),
    }
    weather = {
        "summary": weather_summary.strip(),
        "condition": condition.strip(),
    }
    if precipitation_mm > 0:
        weather["precipitation_mm"] = float(precipitation_mm)
    if windspeed_10m > 0:
        weather["windspeed_10m"] = float(windspeed_10m)

    recording = {
        "device": device.strip(),
        "notes": notes.strip(),
    }

    context = {}
    if tags:
        context["tags"] = tags
    if any(scene.values()):
        context["scene"] = {k: v for k, v in scene.items() if v}
    if any(location.values()):
        context["location"] = {k: v for k, v in location.items() if v}
    if any(weather.values()):
        context["weather"] = {k: v for k, v in weather.items() if v not in ("", None)}
    if any(recording.values()):
        context["recording"] = {k: v for k, v in recording.items() if v}
    enrichment_request = {
        "enabled": bool(auto_context),
        "include_geocoding": bool(include_geocoding),
        "include_weather": bool(include_weather),
        "latitude": coerce_float(latitude_text),
        "longitude": coerce_float(longitude_text),
        "timestamp_text": timestamp_text.strip(),
    }
    return context, enrichment_request


def dual_jsonl_bytes(
    t_sec: np.ndarray,
    raw_scores: np.ndarray,
    adjusted_scores: np.ndarray,
    labels: list[str],
    top_k: int,
) -> bytes:
    lines = []
    for t, raw_row, adjusted_row in zip(t_sec, raw_scores, adjusted_scores):
        lines.append(
            json.dumps(
                {
                    "t_sec": float(t),
                    "raw_top": top_k_labels(raw_row, labels, top_k),
                    "adjusted_top": top_k_labels(adjusted_row, labels, top_k),
                }
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def summary_json_bytes(summary, context: dict, adjustment_vector: np.ndarray, labels: list[str], audio_stats: dict) -> bytes:
    payload = {
        "semantic_summary": {
            "scene_tags": summary.scene_tags,
            "dominant_labels": summary.dominant_labels,
            "dominant_non_silence_labels": summary.dominant_non_silence_labels,
            "salient_segments": summary.salient_segments,
            "semantic_scene": summary.semantic_scene,
            "narrative": summary.narrative,
            "evidence": summary.evidence,
        },
        "context": context,
        "top_adjustment_weights": top_k_labels(adjustment_vector, labels, 15),
        "audio_stats": audio_stats,
    }
    return json.dumps(payload, indent=2).encode("utf-8")


model_path = "yamnet.tflite"
labels_path = "yamnet_class_map.csv"

input_col1, input_col2 = st.columns(2)
with input_col1:
    audio_file = st.file_uploader("Audio file", type=["wav", "mp3", "m4a", "flac", "ogg"])
with input_col2:
    label_metadata_file = st.file_uploader(
        "Labels / metadata file",
        type=["csv", "json"],
        help="Use a YAMNet class-map CSV/JSON, or a metadata file that includes label names and recording context.",
    )

uploaded_metadata = metadata_from_label_file(label_metadata_file)

try:
    window_default = float(get_model_expected_len(model_path) / 16000.0)
    model_runtime_available = Interpreter is not None
except RuntimeError as exc:
    window_default = 0.975
    model_runtime_available = False
    st.warning(str(exc))

with st.expander("Model parameters", expanded=False):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        sample_hz = st.slider("Sample rate (frames/sec)", min_value=1.0, max_value=10.0, value=1.0, step=0.5)
    with col_b:
        avg_frames = st.slider("Temporal averaging", min_value=1, max_value=10, value=1, step=1)
    with col_c:
        top_k = st.slider("Top-k labels", min_value=3, max_value=10, value=5, step=1)

    auto_window = st.checkbox("Use model-derived window length", value=True)
    window_s = st.number_input("Window length (seconds)", min_value=0.1, value=window_default, step=0.025, disabled=auto_window)

manual_context, enrichment_request = context_from_form(uploaded_metadata)
run_clicked = st.button(
    "Run Semantic Classification",
    type="primary",
    use_container_width=True,
    disabled=audio_file is None or not model_runtime_available,
)

if run_clicked and audio_file is not None:
    with st.spinner("Running semantic pipeline..."):
        labels = load_labels_from_upload(label_metadata_file)
        interpreter = load_interpreter(model_path)
        audio_bytes = audio_file.getvalue()
        waveform, sr = read_audio_from_upload(audio_bytes, audio_file.name)
        waveform = to_mono(np.asarray(waveform)).astype(np.float32)
        audio_stats = compute_audio_stats(waveform, sr)
        effective_window_s = window_default if auto_window else float(window_s)
        t_sec, raw_scores = classify_waveform(
            interpreter,
            waveform,
            sr,
            sample_hz=float(sample_hz),
            window_s=effective_window_s,
            avg_frames=int(avg_frames),
        )
        if len(labels) != raw_scores.shape[1]:
            st.warning(
                f"Uploaded label file has {len(labels)} labels, but the model produced {raw_scores.shape[1]} scores. "
                "Using the bundled YAMNet class map for score alignment."
            )
            labels = load_labels_cached(labels_path)
            if len(labels) != raw_scores.shape[1]:
                raise ValueError("Bundled labels also do not match the model output shape.")

        context = dict(manual_context)
        automatic_context = {}
        if enrichment_request["enabled"]:
            latitude = enrichment_request["latitude"]
            longitude = enrichment_request["longitude"]
            if latitude is not None and longitude is not None:
                try:
                    automatic_context = enrich_context_cached(
                        latitude,
                        longitude,
                        enrichment_request["timestamp_text"],
                        json.dumps(uploaded_metadata, sort_keys=True),
                        enrichment_request["include_geocoding"],
                        enrichment_request["include_weather"],
                    )
                except Exception as exc:
                    st.warning(f"Automatic context fetch failed: {exc}")
            else:
                st.info("Add latitude and longitude to fetch automatic context.")
        metadata_context = context_from_metadata(uploaded_metadata)
        if isinstance(uploaded_metadata.get("context"), dict):
            metadata_context = merge_context(metadata_context, uploaded_metadata["context"])
        context = merge_context(metadata_context, merge_context(automatic_context, context))

        adjustment_vector, impacts, context_signals = build_context_adjustment(labels, context)
        adjusted_scores = np.clip(raw_scores * adjustment_vector[None, :], 0.0, 1.0)
        summary = build_semantic_summary(
            t_sec=t_sec,
            raw_scores=raw_scores,
            adjusted_scores=adjusted_scores,
            labels=labels,
            top_k=int(top_k),
            context=context,
            impacts=impacts,
            context_signals=context_signals,
            audio_stats=audio_stats,
        )

    st.subheader("Semantic Output")
    scene_text = html.escape(summary.semantic_scene.title())
    duration_text = html.escape(f"{audio_stats['duration_sec']:.1f} s")
    confidence = float(summary.evidence["scene_confidence"])
    confidence_text = html.escape(f"{confidence:.0%}")
    st.markdown(
        f"""
        <div class="semantic-summary-grid">
            <div class="semantic-summary-card">
                <div class="semantic-summary-label">Semantic scene</div>
                <div class="semantic-summary-value scene">{scene_text}</div>
                <div class="semantic-summary-subtext">Whole-recording interpretation</div>
            </div>
            <div class="semantic-summary-card">
                <div class="semantic-summary-label">Duration</div>
                <div class="semantic-summary-value">{duration_text}</div>
                <div class="semantic-summary-subtext">Analyzed audio length</div>
            </div>
            <div class="semantic-summary-card">
                <div class="semantic-summary-label">Scene confidence</div>
                <div class="semantic-summary-value">{confidence_text}</div>
                <div class="semantic-summary-subtext">Non-silence evidence strength</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(summary.narrative)
    if summary.scene_tags:
        st.caption("Scene tags: " + ", ".join(summary.scene_tags))

    with st.expander("Final context used by the semantic model", expanded=False):
        st.code(json.dumps(context, indent=2), language="json")

    st.markdown("**Dominant non-silence labels**")
    st.dataframe(pd.DataFrame(summary.dominant_non_silence_labels), use_container_width=True, hide_index=True)

    st.markdown("**Salient chunks**")
    chunks_df = pd.DataFrame(summary.salient_segments)
    if not chunks_df.empty:
        chunks_df["descriptive_tags"] = chunks_df["label"]
        chunks_df["description"] = chunks_df.apply(
            lambda row: (
                f"{row['label']} is salient from {row['start_sec']:.2f}s to {row['end_sec']:.2f}s "
                f"(peak={row['peak_score']:.3f})."
            ),
            axis=1,
        )
        chunks_df = chunks_df[
            ["start_sec", "end_sec", "descriptive_tags", "description", "peak_score", "mean_score", "frame_count"]
        ].rename(columns={"start_sec": "start_time_sec", "end_sec": "end_time_sec"})
    st.dataframe(chunks_df, use_container_width=True, hide_index=True)

    st.markdown("**Top adjustments**")
    st.dataframe(pd.DataFrame(summary.evidence["top_rule_impacts"]), use_container_width=True, hide_index=True)

    frame_times = pd.DataFrame(
        {
            "t_sec": t_sec,
            "raw_top_1": [items[0]["label"] for items in (top_k_labels(row, labels, 1) for row in raw_scores)],
            "adjusted_top_1": [items[0]["label"] for items in (top_k_labels(row, labels, 1) for row in adjusted_scores)],
        }
    )
    with st.expander("Top-1 timeline preview", expanded=False):
        timeline_plot_df = frame_times.melt(
            id_vars="t_sec",
            value_vars=["raw_top_1", "adjusted_top_1"],
            var_name="score_source",
            value_name="label",
        )
        timeline_plot_df["score_source"] = timeline_plot_df["score_source"].map(
            {"raw_top_1": "Raw", "adjusted_top_1": "Context-adjusted"}
        )
        timeline_fig = px.scatter(
            timeline_plot_df,
            x="t_sec",
            y="label",
            color="score_source",
            symbol="score_source",
            labels={"t_sec": "Time (s)", "label": "Top-1 label", "score_source": "Score source"},
            height=380,
        )
        timeline_fig.update_traces(marker={"size": 8})
        timeline_fig.update_layout(margin={"l": 10, "r": 10, "t": 10, "b": 10})
        st.plotly_chart(timeline_fig, use_container_width=True)
        st.dataframe(frame_times, use_container_width=True, hide_index=True)

    download_col1, download_col2, download_col3 = st.columns(3)
    with download_col1:
        st.download_button(
            "Download semantic predictions JSONL",
            data=dual_jsonl_bytes(t_sec, raw_scores, adjusted_scores, labels, int(top_k)),
            file_name=f"{Path(audio_file.name).stem}_semantic_predictions.jsonl",
            mime="application/jsonl",
            use_container_width=True,
        )
    with download_col2:
        npz_buffer = io.BytesIO()
        np.savez_compressed(
            npz_buffer,
            t_sec=t_sec.astype(np.float32),
            raw_scores=raw_scores.astype(np.float32),
            adjusted_scores=adjusted_scores.astype(np.float32),
            labels=np.array(labels, dtype=object),
            adjustment_vector=adjustment_vector.astype(np.float32),
        )
        st.download_button(
            "Download semantic matrix NPZ",
            data=npz_buffer.getvalue(),
            file_name=f"{Path(audio_file.name).stem}_semantic_predictions.npz",
            mime="application/octet-stream",
            use_container_width=True,
        )
    with download_col3:
        st.download_button(
            "Download semantic summary JSON",
            data=summary_json_bytes(summary, context, adjustment_vector, labels, audio_stats),
            file_name=f"{Path(audio_file.name).stem}_semantic_summary.json",
            mime="application/json",
            use_container_width=True,
        )
