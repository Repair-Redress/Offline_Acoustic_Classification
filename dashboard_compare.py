import io
import json
import math
from datetime import datetime
import re
#from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import soundfile as sf
import scipy.signal
from pydub import AudioSegment

try:
    import tflite_runtime.interpreter as tflite
    InterpreterCls = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    InterpreterCls = tf.lite.Interpreter


# -----------------------------
# Utilities: labels + parsing
# -----------------------------

def _normalize_label(s: str) -> str:
    # normalize label names for matching (case/whitespace)
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _parse_android_capture_time(s: str) -> Optional[datetime]:
    """
    Android 'Capture time' examples:
      1/20/2026_14:41:9:91
      1/20/2026_14:41:11:440
    Interprets last field as milliseconds when 4 time parts exist.
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None

    s = s.replace("_", " ")
    parts = s.split()
    if len(parts) != 2:
        return None

    date_part, time_part = parts
    # parse date
    dt_date = None
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            dt_date = datetime.strptime(date_part, fmt).date()
            break
        except Exception:
            continue
    if dt_date is None:
        return None

    tparts = time_part.split(":")
    if len(tparts) < 2:
        return None

    try:
        hh = int(tparts[0])
        mm = int(tparts[1])
        ss = int(tparts[2]) if len(tparts) >= 3 else 0
        ms = int(tparts[3]) if len(tparts) >= 4 else 0
        # clamp/pad ms
        if ms < 0:
            ms = 0
        if ms > 999999: #guard for malformed formats
            ms = ms % 1000
        # if it's >999 but plausible microseconds, coerce:
        if ms > 999:
            ms = int(str(ms)[:3])
    except Exception:
        return None

    return datetime(dt_date.year, dt_date.month, dt_date.day, hh, mm, ss, ms * 1000)


def _detect_android_wide_csv(df: pd.DataFrame) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Returns (is_android, capture_time_col, capture_location_col).
    Android schema typically has 'Capture time' and 'Capture location' plus many label columns.
    """
    cols_norm = {_normalize_label(c): c for c in df.columns}
    time_col = None
    loc_col = None

    # common names
    for key in ["capture time", "capture_time", "time"]:
        if key in cols_norm:
            time_col = cols_norm[key]
            break

    for key in ["capture location", "capture_location", "location"]:
        if key in cols_norm:
            loc_col = cols_norm[key]
            break

    # heuristic: android wide schema has a time column + many numeric label columns
    if time_col is None:
        return (False, None, None)

    # count numeric-like columns excluding obvious metadata
    meta_like = set()
    for c in df.columns:
        lc = _normalize_label(c)
        if "capture" in lc or "location" in lc or lc in ("lat", "lon", "latitude", "longitude"):
            meta_like.add(c)

    numeric_like = 0
    for c in df.columns:
        if c in meta_like:
            continue
        ser = pd.to_numeric(df[c], errors="coerce")
        # android label columns should be mostly numeric
        if ser.notna().mean() >= 0.6:
            numeric_like += 1

    # YAMNet has 521 labels; android export will usually have ~521 dims of numeric columns
    is_android = numeric_like >= 50
    return (is_android, time_col, loc_col)


def convert_android_wide_csv_to_long(
    df: pd.DataFrame,
    android_top_k: int = 5,
    canonical_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Converts Android wide CSV (one column per label) into long-form:
      t_sec, rank, label, score
    Also preserves capture_time / capture_location if present.
    """
    is_android, time_col, loc_col = _detect_android_wide_csv(df)
    if not is_android or time_col is None:
        raise ValueError("Not an Android wide-label CSV format.")

    # Build canonical label mapping (optional; helps if there are minor whitespace/case differences)
    canon_map = None
    if canonical_labels:
        canon_map = {_normalize_label(x): x for x in canonical_labels}

    # Identify label columns: numeric-like and not metadata
    meta_like = set()
    for c in df.columns:
        lc = _normalize_label(c)
        if c == time_col or c == loc_col:
            meta_like.add(c)
        if "capture" in lc or "location" in lc or lc in ("lat", "lon", "latitude", "longitude"):
            meta_like.add(c)

    label_cols = []
    # If we have canonical labels, only accept columns that match them (after normalization)
    if canon_map is not None:
        canon_keys = set(canon_map.keys())
        for c in df.columns:
            if c in meta_like:
                continue
            if _normalize_label(c) in canon_keys:
                label_cols.append(c)
    else: # Fallback heuristic: numeric-like columns
        for c in df.columns:
            if c in meta_like:
                continue
            ser = pd.to_numeric(df[c], errors="coerce")
            if ser.notna().mean() >= 0.6:
                label_cols.append(c)

    if not label_cols:
        raise ValueError("Android CSV detected, but no numeric label columns were found.")

    # Parse capture times into t_sec (relative to first valid time)
    dts = [ _parse_android_capture_time(x) for x in df[time_col].tolist() ]
    valid = [dt for dt in dts if dt is not None]
    if not valid:
        # fallback: use row index as seconds
        t_sec = np.arange(len(df), dtype=np.float32)
        base_dt = None
    else:
        base_dt = valid[0]
        t_sec = np.array([
            (dt - base_dt).total_seconds() if dt is not None else np.nan
            for dt in dts
        ], dtype=np.float32)

        # fill missing with interpolation then forward/backfill
        t_ser = pd.Series(t_sec)
        t_ser = t_ser.interpolate(limit_direction="both").fillna(method="ffill").fillna(method="bfill")
        t_sec = t_ser.to_numpy(dtype=np.float32)

    # Convert each row’s 521 scores into top-k long rows
    scores_mat = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    rows = []
    for i in range(scores_mat.shape[0]):
        s = scores_mat[i]
        # top-k indices
        k = min(int(android_top_k), len(label_cols))
        idx = np.argsort(-s)[:k]

        cap_time_val = df.iloc[i][time_col]
        cap_loc_val = df.iloc[i][loc_col] if (loc_col is not None and loc_col in df.columns) else None

        for r, j in enumerate(idx, start=1):
            raw_label = label_cols[int(j)]
            norm = _normalize_label(raw_label)
            label = canon_map.get(norm, raw_label) if canon_map else raw_label
            rows.append({
                "t_sec": float(t_sec[i]),
                "rank": r,
                "label": str(label),
                "score": float(s[int(j)]),
                "capture_time": str(cap_time_val),
                "capture_location": str(cap_loc_val) if cap_loc_val is not None else ""
            })

    out = pd.DataFrame(rows).sort_values(["t_sec", "rank"]).reset_index(drop=True)
    out.attrs["schema"] = "android_wide_converted"
    return out

def read_csv_bytes_auto_sep(b: bytes) -> pd.DataFrame:
    """
    Android exports are often semicolon-delimited to avoid quoting label names containing commas.
    This detects delimiter from the header line and reads accordingly.
    """
    sample = b[:8192].decode("utf-8", errors="ignore")
    first_line = sample.splitlines()[0] if sample.splitlines() else ""

    # Choose delimiter by count in header (robust even when labels contain commas)
    candidates = [";", ",", "\t"]
    delim = max(candidates, key=lambda d: first_line.count(d))
    if first_line.count(delim) == 0:
        delim = ","  # fallback

    return pd.read_csv(io.BytesIO(b), sep=delim)

def read_audio_any_format(audio_bytes: bytes, filename: str) -> tuple[np.ndarray, int]:
    """
    Returns: (samples, sample_rate)
    samples is float32, shape [n] or [n, channels]
    Tries soundfile first; falls back to pydub (ffmpeg) for mp3/m4a/etc.
    """
    # 1) Try soundfile (fast + accurate for wav/flac/ogg/aiff)
    try:
        x, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
        return np.asarray(x, dtype=np.float32), int(sr)
    except Exception:
        pass

    # 2) Fallback: pydub (requires ffmpeg), supports mp3/m4a and more
    ext = (filename.split(".")[-1] or "").lower()
    fmt = ext
    # pydub/ffmpeg often treats m4a as mp4 container
    if fmt == "m4a":
        fmt = "mp4"

    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt if fmt else None)

    sr = seg.frame_rate
    channels = seg.channels
    arr = np.array(seg.get_array_of_samples())

    # reshape to [n, channels] if multi-channel
    if channels > 1:
        arr = arr.reshape((-1, channels))

    # convert integer PCM -> float32 in [-1, 1]
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        x = arr.astype(np.float32) / max(abs(info.min), info.max)
    else:
        x = arr.astype(np.float32)

    return x, int(sr)

@st.cache_data
def load_yamnet_labels_csv(path: str) -> List[str]:
    """Loads canonical YAMNet class map CSV (expects 'display_name' col)."""
    df = pd.read_csv(path)
    # canonical file uses display_name
    if "display_name" in df.columns:
        labels = df["display_name"].astype(str).tolist()
    elif "name" in df.columns:
        labels = df["name"].astype(str).tolist()
    else:
        # fallback: last column
        labels = df[df.columns[-1]].astype(str).tolist()
    return labels


def parse_predictions_jsonl_bytes(b: bytes) -> pd.DataFrame:
    """
    Parses JSONL in your format:
      {"t_sec": 12.3, "top": [{"label": "...", "score": 0.4}, ...]}
    Returns long-form df: columns [t_sec, rank, label, score]
    """
    rows = []
    for line in b.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        t = float(obj["t_sec"])
        top = obj.get("top", [])
        for r, item in enumerate(top, start=1):
            rows.append({
                "t_sec": t,
                "rank": r,
                "label": str(item["label"]),
                "score": float(item["score"])
            })
    if not rows:
        return pd.DataFrame(columns=["t_sec", "rank", "label", "score"])
    return pd.DataFrame(rows).sort_values(["t_sec", "rank"]).reset_index(drop=True)


def parse_predictions_json_bytes(b: bytes) -> pd.DataFrame:
    """
    Supports a few JSON shapes:
      - list of {"t_sec":..., "top":[...]}
      - {"predictions": [ ...same... ]}
    """
    obj = json.loads(b.decode("utf-8"))
    if isinstance(obj, dict) and "predictions" in obj:
        obj = obj["predictions"]
    if not isinstance(obj, list):
        raise ValueError("JSON annotations must be a list of prediction objects or {predictions:[...]}")

    rows = []
    for entry in obj:
        t = float(entry["t_sec"])
        top = entry.get("top", [])
        for r, item in enumerate(top, start=1):
            rows.append({"t_sec": t, "rank": r, "label": str(item["label"]), "score": float(item["score"])})
    return pd.DataFrame(rows).sort_values(["t_sec", "rank"]).reset_index(drop=True)


def parse_predictions_csv_bytes(
    b: bytes,
    android_top_k: int = 5,
    canonical_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Flexible CSV parser. Supports:
      A) long-form: t_sec,rank,label,score
      B) minimal:  t_sec,label,score  (assumes rank=1)
      C) wide:     t_sec,label_1,score_1,label_2,score_2,...
      D) Android wide: Capture time/location + one column per label (scores)
    """
    df = read_csv_bytes_auto_sep(b)
    cols = set(pd.Index(df.columns).map(str).str.lower())

    # Case A/B: already long-ish
    if {"t_sec", "label", "score"}.issubset(cols):
        out = df.rename(columns={c: c.lower() for c in df.columns})
        if "rank" not in out.columns:
            out["rank"] = 1
        out = out[["t_sec", "rank", "label", "score"]].copy()
        out["t_sec"] = out["t_sec"].astype(float)
        out["rank"] = out["rank"].astype(int)
        out["score"] = out["score"].astype(float)
        out["label"] = out["label"].astype(str)
        return out.sort_values(["t_sec", "rank"]).reset_index(drop=True)

    # Case C: wide format with label_i/score_i
    if "t_sec" in cols:
        lower_cols = [c.lower() for c in df.columns]
        pairs = []
        for c in lower_cols:
            if c.startswith("label_"):
                suffix = c.split("label_")[-1]
                score_col = f"score_{suffix}"
                if score_col in lower_cols:
                    pairs.append((c, score_col))
        if pairs:
            rows = []
            for _, row in df.iterrows():
                t = float(row[[c for c in df.columns if c.lower() == "t_sec"][0]])
                for r, (lc, sc) in enumerate(pairs, start=1):
                    label = row[[c for c in df.columns if c.lower() == lc][0]]
                    score = row[[c for c in df.columns if c.lower() == sc][0]]
                    if pd.isna(label) or pd.isna(score):
                        continue
                    rows.append({"t_sec": t, "rank": r, "label": str(label), "score": float(score)})
            return pd.DataFrame(rows).sort_values(["t_sec", "rank"]).reset_index(drop=True)

    # Case D: Android wide schema fallback
    try:
        return convert_android_wide_csv_to_long(
            df,
            android_top_k=android_top_k,
            canonical_labels=canonical_labels,
        )
    except Exception:
        pass

    raise ValueError(
        "Unrecognized CSV annotations format.\n"
        "Supported: (1) t_sec,rank,label,score  (2) t_sec,label,score  (3) t_sec,label_1,score_1,...  "
        "(4) Android wide CSV with Capture time + label columns."
    )


def parse_annotations(uploaded, android_top_k: int = 5, canonical_labels: Optional[List[str]] = None) -> pd.DataFrame:
    """Auto-detect .jsonl/.json/.csv into standard long-form df. Also supports Android wide CSV."""
    b = uploaded.getvalue()
    name = (uploaded.name or "").lower()

    if name.endswith(".jsonl"):
        return parse_predictions_jsonl_bytes(b)
    if name.endswith(".json"):
        return parse_predictions_json_bytes(b)
    if name.endswith(".csv"):
        return parse_predictions_csv_bytes(b, android_top_k=android_top_k, canonical_labels=canonical_labels)

    # fallback: try jsonl then json
    try:
        return parse_predictions_jsonl_bytes(b)
    except Exception:
        try:
            return parse_predictions_json_bytes(b)
        except Exception as e:
            raise ValueError(f"Unsupported annotations file: {uploaded.name}. Use .jsonl, .json, or .csv") from e



def to_top1(df_long: pd.DataFrame) -> pd.DataFrame:
    """Returns df with one row per frame: t_sec, top1, score1"""
    if df_long.empty:
        return pd.DataFrame(columns=["t_sec", "top1", "score1"])
    top1 = df_long[df_long["rank"] == 1][["t_sec", "label", "score"]].copy()
    top1 = top1.rename(columns={"label": "top1", "score": "score1"}).sort_values("t_sec").reset_index(drop=True)
    return top1


def nearest_frame_time(times: np.ndarray, t: float) -> float:
    """Return nearest time in sorted times array."""
    if len(times) == 0:
        return float("nan")
    idx = np.searchsorted(times, t)
    if idx == 0:
        return float(times[0])
    if idx >= len(times):
        return float(times[-1])
    before = times[idx - 1]
    after = times[idx]
    return float(before if abs(t - before) <= abs(t - after) else after)


# -----------------------------
# Offline classification core
# -----------------------------

def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    # soundfile returns (n, channels) if always_2d=True; but we use always_2d=False, so handle both
    if x.ndim == 2 and x.shape[1] in (1, 2, 3, 4):
        return np.mean(x, axis=1)
    return np.mean(x, axis=-1)


def resample(x: np.ndarray, sr_in: int, sr_out: int = 16000) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = math.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return scipy.signal.resample_poly(x, up, down).astype(np.float32)


def frame_waveform(x: np.ndarray, sr: int, window_s: float, hop_s: float) -> Tuple[np.ndarray, np.ndarray]:
    win = int(round(window_s * sr))
    hop = int(round(hop_s * sr))
    if win <= 0 or hop <= 0:
        raise ValueError("window_s and hop_s must be > 0")

    frames = []
    centers = []
    n = len(x)
    i = 0
    while i < n:
        chunk = x[i:i + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        frames.append(chunk)
        centers.append((i + win / 2) / sr)
        i += hop

    return np.stack(frames).astype(np.float32), np.array(centers, dtype=np.float32)


def temporal_average(scores: np.ndarray, avg_frames: int) -> np.ndarray:
    if avg_frames <= 1:
        return scores
    out = np.zeros_like(scores)
    csum = np.cumsum(scores, axis=0)
    for i in range(scores.shape[0]):
        j0 = max(0, i - avg_frames + 1)
        window_sum = csum[i] - (csum[j0 - 1] if j0 > 0 else 0)
        out[i] = window_sum / (i - j0 + 1)
    return out


@st.cache_resource
def load_interpreter(model_path: str):
    interp = InterpreterCls(model_path=model_path)
    interp.allocate_tensors()
    return interp


def run_tflite_scores(interp, frames: np.ndarray, progress: Optional[st.progress] = None) -> np.ndarray:
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    scores = []
    n = len(frames)
    for i, f in enumerate(frames):
        x = f.astype(np.float32)
        # some models accept [win], some [1, win]
        if len(input_details[0]["shape"]) == 2:
            x = x[None, :]
        interp.set_tensor(input_details[0]["index"], x)
        interp.invoke()
        y = interp.get_tensor(output_details[0]["index"])
        y = np.squeeze(y)  # [521]
        scores.append(y)

        if progress is not None and (i % 10 == 0 or i == n - 1):
            progress.progress((i + 1) / n)

    return np.stack(scores).astype(np.float32)


def classify_audio_bytes(
    audio_bytes: bytes,
    audio_name: str,
    model_path: str,
    labels: List[str],
    sample_hz: float = 1.0,
    window_s: float = 0.975,
    avg_frames: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      t_centers: [T]
      scores:    [T, 521]
      labels:    list[str] length 521
    """
    # Read audio (WAV/FLAC typically). MP3 may not work with soundfile.
    x, sr = read_audio_any_format(audio_bytes, filename=audio_name)
    x = np.asarray(x)
    x = to_mono(x).astype(np.float32)

    # Normalize if looks like int16 scale
    if np.max(np.abs(x)) > 1.5:
        x = x / 32768.0

    x = resample(x, sr, 16000)

    interp = load_interpreter(model_path)
    input_details = interp.get_input_details()

    # Model expects a fixed number of waveform samples (e.g., 15600 for YAMNet TFLite)
    expected_len = int(input_details[0]["shape"][-1])

    # Force window length to match the model exactly
    effective_window_s = expected_len / 16000.0

    # Optional: warn if user-set window_s doesn't match
    if abs(window_s - effective_window_s) > 1e-6:
        st.warning(
            f"window_s={window_s} does not match model input. "
            f"Overriding to {effective_window_s:.6f}s ({expected_len} samples)."
        )

    hop_s = 1.0 / float(sample_hz)
    frames, t_centers = frame_waveform(x, 16000, effective_window_s, hop_s)

    prog = st.progress(0.0, text="Running offline classification…")
    raw_scores = run_tflite_scores(interp, frames, progress=prog)
    prog.empty()

    scores = temporal_average(raw_scores, avg_frames)

    if scores.shape[1] != len(labels):
        raise ValueError(f"Mismatch: model outputs {scores.shape[1]} classes but labels has {len(labels)}.")

    return t_centers, scores, labels


def scores_to_long_df(t_centers: np.ndarray, scores: np.ndarray, labels: List[str], top_k: int) -> pd.DataFrame:
    rows = []
    for t, s in zip(t_centers, scores):
        idx = np.argsort(-s)[:top_k]
        for r, i in enumerate(idx, start=1):
            rows.append({"t_sec": float(t), "rank": r, "label": labels[int(i)], "score": float(s[int(i)])})
    return pd.DataFrame(rows).sort_values(["t_sec", "rank"]).reset_index(drop=True)


def df_long_to_jsonl_bytes(df_long: pd.DataFrame) -> bytes:
    """Convert long-form df back to JSONL in the standard format."""
    if df_long.empty:
        return b""
    out_lines = []
    for t, grp in df_long.groupby("t_sec", sort=True):
        grp = grp.sort_values("rank")
        top = [{"label": row["label"], "score": float(row["score"])} for _, row in grp.iterrows()]
        out_lines.append(json.dumps({"t_sec": float(t), "top": top}))
    return ("\n".join(out_lines) + "\n").encode("utf-8")


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Offline vs Real-time AudioSet (YAMNet) Viewer", layout="wide")
st.title("Offline vs Real-time AudioSet (YAMNet) Dashboard")

st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "Choose a workflow",
    [
        "1) Upload audio → run OFFLINE classification → step through",
        "2) Upload annotations → step through",
        "3) Upload audio + REAL-TIME annotations → run OFFLINE + compare side-by-side",
    ],
)

st.sidebar.header("Model / labels")
model_path = st.sidebar.text_input("Model path", value="yamnet.tflite")
labels_path = st.sidebar.text_input("Label map path", value="yamnet_class_map.csv")

# Load labels once
try:
    labels = load_yamnet_labels_csv(labels_path)
except Exception as e:
    st.sidebar.error(f"Could not load labels from {labels_path}: {e}")
    labels = []

st.sidebar.header("Offline params")
sample_hz = st.sidebar.number_input("sample_hz (predictions / sec)", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
window_s = st.sidebar.number_input("window_s (seconds)", min_value=0.1, max_value=5.0, value=0.975, step=0.025, disabled=True)
avg_frames = st.sidebar.number_input("avg_frames (smoothing)", min_value=1, max_value=50, value=1, step=1)
top_k = st.sidebar.number_input("top_k (saved per frame)", min_value=1, max_value=30, value=5, step=1)

st.sidebar.header("Display")
min_conf = st.sidebar.slider("Min confidence (filter)", 0.0, 1.0, 0.0, 0.01)


def stepper_view(df_long: pd.DataFrame, title: str, audio_bytes: Optional[bytes] = None):
    """Reusable "step-through" UI for a prediction timeline."""
    st.subheader(title)

    if df_long.empty:
        st.info("No predictions loaded.")
        return

    top1 = to_top1(df_long)
    if top1.empty:
        st.info("No rank=1 rows found in predictions.")
        return

    # Filtered view for plotting
    top1_plot = top1[top1["score1"] >= min_conf].copy()

    colA, colB = st.columns([2, 1], gap="large")
    with colA:
        try:
            import plotly.express as px
            fig = px.scatter(top1_plot, x="t_sec", y="top1", color="top1", hover_data=["score1"], height=450)
            fig.update_layout(showlegend=False, xaxis_title="Time (s)", yaxis_title="Top-1 label")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("Plotly not available; install 'plotly' for interactive plots.")

    with colB:
        if audio_bytes is not None:
            st.caption("Audio")
            st.audio(audio_bytes)

        times = top1["t_sec"].to_numpy(dtype=float)
        tmin, tmax = float(times.min()), float(times.max())

        # stepping controls
        if "step_idx" not in st.session_state:
            st.session_state.step_idx = 0

        # map idx to time
        st.session_state.step_idx = int(np.clip(st.session_state.step_idx, 0, len(times) - 1))

        btnc1, btnc2, btnc3 = st.columns([1, 1, 1])
        if btnc1.button("◀ Prev"):
            st.session_state.step_idx = max(0, st.session_state.step_idx - 1)
        if btnc2.button("Next ▶"):
            st.session_state.step_idx = min(len(times) - 1, st.session_state.step_idx + 1)
        if btnc3.button("Reset"):
            st.session_state.step_idx = 0

        t_pick = st.slider("Time (s)", tmin, tmax, float(times[st.session_state.step_idx]))
        t_near = nearest_frame_time(times, t_pick)
        st.session_state.step_idx = int(np.where(times == t_near)[0][0])

        # show frame
        frame = df_long[df_long["t_sec"] == t_near].sort_values("rank").copy()
        frame = frame[frame["score"] >= min_conf] if min_conf > 0 else frame
        st.write(f"Frame @ **{t_near:.3f}s** (filtered by min_conf={min_conf:.2f})")
        st.dataframe(frame, use_container_width=True, height=260)


def side_by_side_view(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_name: str,
    right_name: str,
    audio_bytes: Optional[bytes] = None
):
    st.subheader("Side-by-side comparison")

    if df_left.empty or df_right.empty:
        st.info("Need both datasets loaded.")
        return

    top1_L = to_top1(df_left)
    top1_R = to_top1(df_right)
    if top1_L.empty or top1_R.empty:
        st.info("Missing rank=1 rows in one of the datasets.")
        return

    times_L = top1_L["t_sec"].to_numpy(dtype=float)
    times_R = top1_R["t_sec"].to_numpy(dtype=float)

    tmin = float(min(times_L.min(), times_R.min()))
    tmax = float(max(times_L.max(), times_R.max()))

    col0, col1, col2 = st.columns([1, 1, 1], gap="large")

    with col0:
        if audio_bytes is not None:
            st.caption("Audio")
            st.audio(audio_bytes)
        t_pick = st.slider("Compare at time (s)", tmin, tmax, tmin)
        tL = nearest_frame_time(times_L, t_pick)
        tR = nearest_frame_time(times_R, t_pick)

        # quick summary
        rowL = top1_L[top1_L["t_sec"] == tL].iloc[0]
        rowR = top1_R[top1_R["t_sec"] == tR].iloc[0]
        st.markdown(
            f"""
            **Nearest frames**  
            - {left_name}: **{tL:.3f}s** → **{rowL['top1']}** ({rowL['score1']:.3f})  
            - {right_name}: **{tR:.3f}s** → **{rowR['top1']}** ({rowR['score1']:.3f})
            """
        )

    with col1:
        st.caption(left_name)
        frameL = df_left[df_left["t_sec"] == tL].sort_values("rank")
        frameL = frameL[frameL["score"] >= min_conf] if min_conf > 0 else frameL
        st.dataframe(frameL, use_container_width=True, height=360)

    with col2:
        st.caption(right_name)
        frameR = df_right[df_right["t_sec"] == tR].sort_values("rank")
        frameR = frameR[frameR["score"] >= min_conf] if min_conf > 0 else frameR
        st.dataframe(frameR, use_container_width=True, height=360)

    # Optional: quick timeline plots
    st.divider()
    st.subheader("Top-1 timelines")

    try:
        import plotly.express as px
        c1, c2 = st.columns(2, gap="large")
        with c1:
            p = top1_L[top1_L["score1"] >= min_conf].copy()
            fig = px.scatter(p, x="t_sec", y="top1", color="top1", hover_data=["score1"], height=420)
            fig.update_layout(showlegend=False, title=left_name, xaxis_title="Time (s)", yaxis_title="Top-1 label")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            p = top1_R[top1_R["score1"] >= min_conf].copy()
            fig = px.scatter(p, x="t_sec", y="top1", color="top1", hover_data=["score1"], height=420)
            fig.update_layout(showlegend=False, title=right_name, xaxis_title="Time (s)", yaxis_title="Top-1 label")
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Install 'plotly' for interactive timeline plots.")


# -----------------------------
# Mode implementations
# -----------------------------

if mode.startswith("1)"):
    st.write("Upload an audio file, run offline YAMNet classification, then step through predictions.")

    audio_up = st.file_uploader("Upload audio (WAV recommended)", type=["wav", "flac", "ogg", "aiff", "aif", "aifc", "mp3", "m4a", "mp4"])
    if audio_up is None:
        st.info("Upload an audio file to begin.")
        st.stop()

    audio_bytes = audio_up.getvalue()
    st.audio(audio_bytes)

    if not labels:
        st.error("Label map not loaded. Fix label map path in the sidebar.")
        st.stop()

    run_btn = st.button("Run offline classification", type="primary")
    if run_btn:
        try:
            t_centers, scores, labels_out = classify_audio_bytes(
                audio_bytes=audio_bytes,
                audio_name=audio_up.name,
                model_path=model_path,
                labels=labels,
                sample_hz=float(sample_hz),
                window_s=float(window_s),
                avg_frames=int(avg_frames),
            )
            df_long = scores_to_long_df(t_centers, scores, labels_out, top_k=int(top_k))
            st.session_state["offline_df"] = df_long
        except Exception as e:
            st.exception(e)
            st.stop()

    df_off = st.session_state.get("offline_df", pd.DataFrame())
    if not df_off.empty:
        stepper_view(df_off, "OFFLINE predictions", audio_bytes=audio_bytes)

        # download
        out_bytes = df_long_to_jsonl_bytes(df_off)
        st.download_button(
            "Download offline predictions.jsonl",
            data=out_bytes,
            file_name="predictions.jsonl",
            mime="application/jsonl",
        )

elif mode.startswith("2)"):
    st.write("Upload an annotations file (JSONL/JSON/CSV) and step through predictions.")
    ann_up = st.file_uploader("Upload annotations file", type=["jsonl", "json", "csv"])
    audio_up = st.file_uploader("(Optional) Upload audio for playback", type=["wav", "flac", "ogg", "aiff", "aif", "aifc", "mp3", "m4a", "mp4"])

    audio_bytes = audio_up.getvalue() if audio_up is not None else None

    if ann_up is None:
        st.info("Upload an annotations file to begin.")
        st.stop()

    try:
        df_ann = parse_annotations(ann_up, android_top_k=int(top_k), canonical_labels=labels)
    except Exception as e:
        st.exception(e)
        st.stop()

    st.success(f"Loaded {len(df_ann)} rows from {ann_up.name}")
    if df_ann.attrs.get("schema") == "android_wide_converted":
        st.info("Detected Android wide CSV. Converted to offline-compatible schema (t_sec/top-k).")
        st.download_button(
            "Download converted JSONL (offline schema)",
            data=df_long_to_jsonl_bytes(df_ann),
            file_name="converted_from_android.jsonl",
            mime="application/jsonl",
        )
    stepper_view(df_ann, f"Annotations: {ann_up.name}", audio_bytes=audio_bytes)

elif mode.startswith("3)"):
    st.write("Upload audio + real-time annotations, run offline classification, then compare side-by-side.")

    audio_up = st.file_uploader("Upload audio (WAV recommended)", type=["wav", "flac", "ogg", "aiff", "aif", "aifc", "mp3", "m4a", "mp4"])
    ann_up = st.file_uploader("Upload REAL-TIME output annotations (JSONL/JSON/CSV)", type=["jsonl", "json", "csv"])

    if audio_up is None or ann_up is None:
        st.info("Upload both an audio file and an annotations file to begin.")
        st.stop()

    audio_bytes = audio_up.getvalue()
    st.audio(audio_bytes)

    # Parse real-time annotations
    try:
        df_rt = parse_annotations(ann_up, android_top_k=int(top_k), canonical_labels=labels)
    except Exception as e:
        st.exception(e)
        st.stop()

    if not labels:
        st.error("Label map not loaded. Fix label map path in the sidebar.")
        st.stop()

    run_btn = st.button("Run OFFLINE and compare", type="primary")
    if run_btn:
        try:
            t_centers, scores, labels_out = classify_audio_bytes(
                audio_bytes=audio_bytes,
                audio_name=audio_up.name,
                model_path=model_path,
                labels=labels,
                sample_hz=float(sample_hz),
                window_s=float(window_s),
                avg_frames=int(avg_frames),
            )
            df_off = scores_to_long_df(t_centers, scores, labels_out, top_k=int(top_k))
            st.session_state["compare_off_df"] = df_off
            st.session_state["compare_rt_df"] = df_rt
        except Exception as e:
            st.exception(e)
            st.stop()

    df_off = st.session_state.get("compare_off_df", pd.DataFrame())
    df_rt2 = st.session_state.get("compare_rt_df", pd.DataFrame())

    if not df_off.empty and not df_rt2.empty:
        side_by_side_view(df_rt2, df_off, left_name=f"REAL-TIME ({ann_up.name})", right_name="OFFLINE (this run)", audio_bytes=audio_bytes)

        # Downloads
        out_bytes = df_long_to_jsonl_bytes(df_off)
        st.download_button(
            "Download OFFLINE predictions.jsonl (this run)",
            data=out_bytes,
            file_name="offline_predictions.jsonl",
            mime="application/jsonl",
        )