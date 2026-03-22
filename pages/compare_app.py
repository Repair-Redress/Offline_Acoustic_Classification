import io
import json
import math
from datetime import datetime
import re
#from dataclasses import dataclass
from typing import List, Optional, Tuple
import base64
import hashlib
from audio_time_player import audio_time_player

import numpy as np
import pandas as pd
import streamlit as st

def get_interpreter_cls():
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter

# -----------------------------
# Utilities: labels + parsing
# -----------------------------

def _guess_mime(filename: str) -> str:
    ext = (filename.split(".")[-1] or "").lower()
    if ext in ("wav",): return "audio/wav"
    if ext in ("mp3",): return "audio/mpeg"
    if ext in ("m4a", "mp4"): return "audio/mp4"
    if ext in ("flac",): return "audio/flac"
    if ext in ("ogg",): return "audio/ogg"
    return "audio/wav"

def playback_time_seconds(audio_bytes: bytes, audio_name: str, key: str, poll_ms: int = 250) -> float:
    src_key = hashlib.md5(audio_bytes).hexdigest()
    loaded_key = f"_audio_loaded_{key}"
    needs_key = f"_audio_needs_{key}"

    # Default to True so the first mount ALWAYS gets audio
    needs_audio = bool(st.session_state.get(needs_key, True))
    send_audio = needs_audio or (st.session_state.get(loaded_key) != src_key)
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii") if send_audio else ""

    state = audio_time_player(
        audio_b64=audio_b64,
        mime=_guess_mime(audio_name),
        poll_ms=poll_ms,
        height=90,
        key=key,
    )

    # remember whether frontend says it needs audio
    st.session_state[needs_key] = bool(state.get("needs_audio", False))
    # only mark "loaded" when we actually sent the audio payload
    if send_audio:
        st.session_state[loaded_key] = src_key
    # (optional) debug:
    # st.caption(state)
    return float(state.get("t", 0.0))

def apply_time_shift(df_long: pd.DataFrame, shift_sec: float) -> pd.DataFrame:
    """Shift t_sec by a constant (seconds). Also shifts attrs['full_t_sec'] if present."""
    if df_long.empty or shift_sec == 0:
        return df_long
    out = df_long.copy()
    out["t_sec"] = out["t_sec"].astype(float) + float(shift_sec)
    # preserve attrs
    out.attrs = dict(df_long.attrs)
    # shift full timeline if present
    if out.attrs.get("full_t_sec") is not None:
        out.attrs["full_t_sec"] = np.asarray(out.attrs["full_t_sec"], dtype=np.float32) + np.float32(shift_sec)
    # sort/reset while preserving attrs
    out2 = out.sort_values(["t_sec", "rank"]).reset_index(drop=True)
    out2.attrs = dict(out.attrs)
    return out2

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

def parse_predictions_npz_bytes(b: bytes, top_k: int) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """
    Reads .npz with keys: t_sec [T], scores [T,521], labels [521]
    Returns:
      df_long (top-k long)
      t_sec (np array)
      scores (np array)
      labels (list)
    """
    with io.BytesIO(b) as bio:
        npz = np.load(bio, allow_pickle=True)
        t = npz["t_sec"].astype(np.float32)
        scores = npz["scores"].astype(np.float32)
        labels_arr = npz["labels"]
        labels = [str(x) for x in labels_arr.tolist()]
    df_long = scores_to_long_df(t, scores, labels, top_k=int(top_k))
    df_long.attrs["schema"] = "offline_npz_full"
    df_long.attrs["full_scores"] = scores
    df_long.attrs["full_t_sec"] = t
    df_long.attrs["full_labels"] = labels
    return df_long, t, scores, labels

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
    Attaches attrs:
      full_scores [T,C], full_t_sec [T], full_labels [C]
    """
    is_android, time_col, loc_col = _detect_android_wide_csv(df)
    if not is_android or time_col is None:
        raise ValueError("Not an Android wide-label CSV format.")

    # Canonical label map (optional)
    canon_map = None
    if canonical_labels:
        canon_map = {_normalize_label(x): x for x in canonical_labels}

    # Identify metadata-like columns
    meta_like = set()
    for c in df.columns:
        lc = _normalize_label(c)
        if c == time_col or c == loc_col:
            meta_like.add(c)
        if "capture" in lc or "location" in lc or lc in ("lat", "lon", "latitude", "longitude"):
            meta_like.add(c)

    # Choose label columns; if canonical provided, align to canonical ordering
    if canon_map is not None:
        df_col_by_norm = {_normalize_label(c): c for c in df.columns if c not in meta_like}
        label_cols = []
        full_labels = []
        for canon in canonical_labels:  # canonical_labels is not None here
            n = _normalize_label(canon)
            if n in df_col_by_norm:
                label_cols.append(df_col_by_norm[n])
                full_labels.append(canon)
    else:
        label_cols = []
        for c in df.columns:
            if c in meta_like:
                continue
            ser = pd.to_numeric(df[c], errors="coerce")
            if ser.notna().mean() >= 0.6:
                label_cols.append(c)
        full_labels = list(label_cols)

    if not label_cols:
        raise ValueError("Android CSV detected, but no numeric label columns were found.")

    # compute t_sec from Capture time
    dts = [_parse_android_capture_time(x) for x in df[time_col].tolist()]
    valid = [dt for dt in dts if dt is not None]

    if not valid:
        # fallback: treat row index as seconds
        t_sec = np.arange(len(df), dtype=np.float32)
    else:
        base_dt = valid[0]
        t_sec = np.array(
            [(dt - base_dt).total_seconds() if dt is not None else np.nan for dt in dts],
            dtype=np.float32,
        )
        # fill gaps: interpolate, then ffill/bfill
        t_ser = pd.Series(t_sec).interpolate(limit_direction="both")
        t_ser = t_ser.ffill().bfill()
        t_sec = t_ser.to_numpy(dtype=np.float32)

    # Full score matrix aligned to full_labels order
    scores_mat = (
        df[label_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    # Convert each row to top-k long rows
    rows = []
    for i in range(scores_mat.shape[0]):
        s = scores_mat[i]
        k = min(int(android_top_k), len(full_labels))
        idx = np.argsort(-s)[:k]

        cap_time_val = df.iloc[i][time_col]
        cap_loc_val = df.iloc[i][loc_col] if (loc_col is not None and loc_col in df.columns) else None

        for r, j in enumerate(idx, start=1):
            rows.append({
                "t_sec": float(t_sec[i]),
                "rank": r,
                "label": str(full_labels[int(j)]),
                "score": float(s[int(j)]),
                "capture_time": str(cap_time_val),
                "capture_location": str(cap_loc_val) if cap_loc_val is not None else ""
            })

    out = pd.DataFrame(rows).sort_values(["t_sec", "rank"]).reset_index(drop=True)
    out.attrs["schema"] = "android_wide_converted"
    out.attrs["full_scores"] = scores_mat
    out.attrs["full_t_sec"] = t_sec
    out.attrs["full_labels"] = full_labels
    return out


def npz_bytes_from_scores(t_sec: np.ndarray, scores: np.ndarray, labels_list: List[str]) -> bytes:
    bio = io.BytesIO()
    np.savez_compressed(
        bio,
        t_sec=t_sec.astype(np.float32),
        scores=scores.astype(np.float32),
        labels=np.array(labels_list, dtype=object),
    )
    return bio.getvalue()

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
    from pydub import AudioSegment
    # 1) Try soundfile (fast + accurate for wav/flac/ogg/aiff)
    try:
        import soundfile as sf
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

@st.cache_data(show_spinner=False)
def read_audio_any_format_cached(audio_bytes: bytes, filename: str) -> tuple[np.ndarray, int]:
    return read_audio_any_format(audio_bytes, filename)



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

@st.cache_data(show_spinner=False)
def parse_annotations_bytes(name: str, data: bytes, android_top_k: int, canonical_labels: tuple[str, ...]):
    lower = name.lower()
    canonical_labels = list(canonical_labels)

    if lower.endswith(".jsonl"):
        return parse_predictions_jsonl_bytes(data)
    if lower.endswith(".json"):
        return parse_predictions_json_bytes(data)
    if lower.endswith(".csv"):
        return parse_predictions_csv_bytes(data, android_top_k=android_top_k, canonical_labels=canonical_labels)

    try:
        return parse_predictions_jsonl_bytes(data)
    except Exception:
        return parse_predictions_json_bytes(data)

def to_top1(df_long: pd.DataFrame) -> pd.DataFrame:
    """Returns df with one row per frame: t_sec, top1, score1"""
    if df_long.empty:
        return pd.DataFrame(columns=["t_sec", "top1", "score1"])
    top1 = df_long[df_long["rank"] == 1][["t_sec", "label", "score"]].copy()
    top1 = top1.rename(columns={"label": "top1", "score": "score1"}).sort_values("t_sec").reset_index(drop=True)
    return top1

def topk_at_time(df_long: pd.DataFrame, t: float, k: int) -> pd.DataFrame:
    frame = df_long[df_long["t_sec"] == t].sort_values("rank")
    return frame.head(k)

def nearest_times_for_reference(ref_times: np.ndarray, candidate_times: np.ndarray) -> np.ndarray:
    if len(ref_times) == 0 or len(candidate_times) == 0:
        return np.array([], dtype=float)
    
    idx = np.searchsorted(candidate_times, ref_times)
    idx = np.clip(idx, 0, len(candidate_times) - 1)

    prev_idx = np.clip(idx - 1, 0, len(candidate_times) - 1)
    next_idx = idx

    prev_vals = candidate_times[prev_idx]
    next_vals = candidate_times[next_idx]

    choose_prev = np.abs(ref_times - prev_vals) <= np.abs(ref_times - next_vals)
    return np.where(choose_prev, prev_vals, next_vals)

def compute_compare_metrics(df_rt: pd.DataFrame, df_off: pd.DataFrame, k: int = 10) -> dict:
    """
    Metrics use top-k from both timelines (unless full 521 vectors are available elsewhere).
    """
    if df_rt.empty or df_off.empty:
        return {}
    
    rt_times = np.sort(df_rt["t_sec"].unique().astype(float))
    off_times = np.sort(df_off["t_sec"].unique().astype(float))
    matched_off_times = nearest_times_for_reference(rt_times, off_times)
    pairs = list(zip(rt_times, matched_off_times))

    same_top1 = 0
    same_top2 = 0
    jacc_12 = []
    same_in_topk = []
    common_score_absdiff = []

    # precompute per-time dicts for faster lookup
    rt_groups = {t: g.sort_values("rank") for t, g in df_rt.groupby("t_sec")}
    off_groups = {t: g.sort_values("rank") for t, g in df_off.groupby("t_sec")}

    for tR, tO in pairs:
        gR = rt_groups.get(tR)
        gO = off_groups.get(tO)
        if gR is None or gO is None:
            continue

        gRk = gR.head(k)
        gOk = gO.head(k)

        # label sets
        setR = set(gRk["label"].astype(str))
        setO = set(gOk["label"].astype(str))

        inter = setR & setO
        union = setR | setO

        same_in_topk.append(len(inter))
        jacc_12.append(len(inter) / len(union) if len(union) else 0.0)

        # top1/top2 match
        r1 = gRk.iloc[0]["label"] if len(gRk) >= 1 else None
        o1 = gOk.iloc[0]["label"] if len(gOk) >= 1 else None
        r2 = gRk.iloc[1]["label"] if len(gRk) >= 2 else None
        o2 = gOk.iloc[1]["label"] if len(gOk) >= 2 else None
        same_top1 += int(r1 == o1)
        same_top2 += int(r2 == o2)

        # score diffs on common labels (avg abs diff over intersection)
        if inter:
            r_scores = dict(zip(gRk["label"].astype(str), gRk["score"].astype(float)))
            o_scores = dict(zip(gOk["label"].astype(str), gOk["score"].astype(float)))
            diffs = [abs(r_scores[l] - o_scores[l]) for l in inter if l in r_scores and l in o_scores]
            if diffs:
                common_score_absdiff.append(float(np.mean(diffs)))

    n = max(1, len(pairs))
    out = {
        "frames_compared": len(pairs),
        "avg_same_in_topk": float(np.mean(same_in_topk)) if same_in_topk else 0.0,
        "top1_match_rate": float(same_top1 / n),
        "top2_match_rate": float(same_top2 / n),
        "avg_jaccard_topk": float(np.mean(jacc_12)) if jacc_12 else 0.0,
        "avg_abs_score_diff_on_common_labels": float(np.mean(common_score_absdiff)) if common_score_absdiff else 0.0,
    }
    return out

def compute_stddev_of_category_means(df_long: pd.DataFrame, k: int = 10) -> float:
    """
    Approx: per-label mean score using only appearances in top-k,
    then take stddev across labels.
    """
    if df_long.empty:
        return float("nan")
    dfk = df_long[df_long["rank"] <= k].copy()
    means = dfk.groupby("label")["score"].mean()
    return float(means.std()) if len(means) else float("nan")

def compute_full_vector_pair_metrics(
    t_a: np.ndarray,
    scores_a: np.ndarray,
    t_b: np.ndarray,
    scores_b: np.ndarray,
) -> dict:
    """
    Frame-aligned full-vector comparison metrics.
    Requires matching frame grids.
    """
    if scores_a is None or scores_b is None:
        return {}

    if len(t_a) != len(t_b) or not np.allclose(t_a, t_b, atol=1e-4):
        return {}

    if scores_a.shape != scores_b.shape:
        return {}

    eps = 1e-8
    dot = np.sum(scores_a * scores_b, axis=1)
    norm_a = np.linalg.norm(scores_a, axis=1)
    norm_b = np.linalg.norm(scores_b, axis=1)
    cosine = dot / np.maximum(norm_a * norm_b, eps)

    top1_a = np.argmax(scores_a, axis=1)
    top1_b = np.argmax(scores_b, axis=1)

    abs_diff = np.abs(scores_a - scores_b)
    l2_diff = np.linalg.norm(scores_a - scores_b, axis=1)

    return {
        "mean_cosine_similarity_full": float(np.mean(cosine)),
        "full_top1_match_rate": float(np.mean(top1_a == top1_b)),
        "mean_abs_score_diff_full": float(np.mean(abs_diff)),
        "mean_l2_score_diff_full": float(np.mean(l2_diff)),
    }


def compute_stereo_channel_metrics(result: dict, top_k_compare: int = 5) -> dict:
    """
    Compare LEFT vs RIGHT outputs inside a stereo result dict.
    Uses:
      - existing top-k set/rank metrics
      - full 521-dim score metrics when available
    """
    outputs = result.get("outputs", {})
    if "left" not in outputs or "right" not in outputs:
        return {}

    df_left = result_output_to_df(result, "left", top_k=int(top_k_compare))
    df_right = result_output_to_df(result, "right", top_k=int(top_k_compare))

    topk_metrics = compute_compare_metrics(df_left, df_right, k=int(top_k_compare))
    topk_metrics["k_used"] = int(top_k_compare)

    full_metrics = compute_full_vector_pair_metrics(
        outputs["left"]["t_sec"],
        outputs["left"]["scores"],
        outputs["right"]["t_sec"],
        outputs["right"]["scores"],
    )

    return {**topk_metrics, **full_metrics}


def show_stereo_channel_metrics(result: dict, top_k_compare: int = 5, heading: str = "Stereo channel agreement (LEFT vs RIGHT)"):
    m = compute_stereo_channel_metrics(result, top_k_compare=top_k_compare)
    if not m:
        return

    st.subheader(heading)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Frames compared", m.get("frames_compared", 0))
    c2.metric(f"Top-1 match", f"{100 * m.get('top1_match_rate', 0.0):.1f}%")
    c3.metric(f"Avg same in Top-{m['k_used']}", f"{m.get('avg_same_in_topk', 0.0):.2f}")
    c4.metric(f"Avg Jaccard(Top-{m['k_used']})", f"{m.get('avg_jaccard_topk', 0.0):.3f}")
    c5.metric("Mean cosine (full)", f"{m.get('mean_cosine_similarity_full', float('nan')):.3f}")
    c6.metric("Mean |Δscore| (full)", f"{m.get('mean_abs_score_diff_full', float('nan')):.4f}")

    st.caption(
        f"Full-vector Top-1 match: {100 * m.get('full_top1_match_rate', 0.0):.1f}% • "
        f"Mean L2 distance: {m.get('mean_l2_score_diff_full', float('nan')):.4f} • "
        f"Avg |Δscore| on common Top-{m['k_used']} labels: "
        f"{m.get('avg_abs_score_diff_on_common_labels', 0.0):.4f}"
    )

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

def split_audio_channels(x: np.ndarray) -> dict[str, np.ndarray]:
    """
    Returns either:
      {"mono": [n]}
    or
      {"left": [n], "right": [n]}
    """
    x = np.asarray(x)

    if x.ndim == 1:
        return {"mono": x.astype(np.float32)}

    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D audio array, got shape={x.shape}")

    # Prefer [samples, channels]. If it looks like [channels, samples], transpose.
    if x.shape[1] > 8 and x.shape[0] <= 8:
        x = x.T

    n_channels = x.shape[1]

    if n_channels == 1:
        return {"mono": x[:, 0].astype(np.float32)}

    if n_channels >= 2:
        return {
            "left": x[:, 0].astype(np.float32),
            "right": x[:, 1].astype(np.float32),
        }

    raise ValueError(f"Could not interpret audio shape {x.shape}")


def classify_single_waveform(
    interp,
    x: np.ndarray,
    sr: int,
    sample_hz: float,
    expected_len: int,
    avg_frames: int,
    progress=None,
    progress_start: float = 0.0,
    progress_end: float = 1.0,
    progress_text: str = "Running offline classification…",
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)

    if x.size == 0:
        raise ValueError("Empty waveform.")

    # Normalize if looks like int16-ish scale
    if np.max(np.abs(x)) > 1.5:
        x = x / 32768.0

    x = resample(x, sr, 16000)

    effective_window_s = expected_len / 16000.0
    hop_s = 1.0 / float(sample_hz)

    frames, t_centers = frame_waveform(x, 16000, effective_window_s, hop_s)
    raw_scores = run_tflite_scores(
        interp,
        frames,
        progress=progress,
        progress_start=progress_start,
        progress_end=progress_end,
        progress_text=progress_text,
    )
    scores = temporal_average(raw_scores, avg_frames)
    return t_centers, scores


def result_output_to_df(result: dict, output_key: str, top_k: int) -> pd.DataFrame:
    out = result["outputs"][output_key]
    df = scores_to_long_df(out["t_sec"], out["scores"], result["labels"], top_k=int(top_k))
    _strip_attrs_inplace(df)
    return df

def resample(x: np.ndarray, sr_in: int, sr_out: int = 16000) -> np.ndarray:
    import scipy.signal
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
    InterpreterCls = get_interpreter_cls()
    interp = InterpreterCls(model_path=model_path)
    interp.allocate_tensors()
    return interp


def run_tflite_scores(
    interp,
    frames: np.ndarray,
    progress=None,
    progress_start: float = 0.0,
    progress_end: float = 1.0,
    progress_text: str = "Running offline classification…",
) -> np.ndarray:
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
            frac = progress_start + (progress_end - progress_start) * ((i + 1) / n)
            progress.progress(frac, text=progress_text)

    return np.stack(scores).astype(np.float32)


def classify_audio_bytes(
    audio_bytes: bytes,
    audio_name: str,
    model_path: str,
    labels: List[str],
    sample_hz: float = 1.0,
    window_s: float = 0.975,
    avg_frames: int = 1,
    channel_mode: str = "mono_mixdown",  # mono_mixdown | stereo_separate | stereo_average
) -> dict :
    """
    Returns a dict:
    {
      "requested_mode": ...,
      "effective_mode": ...,
      "labels": [...],
      "outputs": {
          "mono": {"t_sec":..., "scores":...}
          or "left"/"right"
          or "left"/"right"/"avg"
      },
      "primary_key": "mono" or "avg" or None
    }
    """

    x, sr = read_audio_any_format_cached(audio_bytes, filename=audio_name)
    x = np.asarray(x)

    interp = load_interpreter(model_path)
    input_details = interp.get_input_details()
    expected_len = int(input_details[0]["shape"][-1])
    effective_window_s = expected_len / 16000.0

    if abs(window_s - effective_window_s) > 1e-6:
        st.warning(
            f"window_s={window_s} does not match model input. "
            f"Overriding to {effective_window_s:.6f}s ({expected_len} samples)."
        )

    channels = split_audio_channels(x)
    requested_mode = channel_mode
    is_stereo = ("left" in channels and "right" in channels)

    if channel_mode != "mono_mixdown" and not is_stereo:
        st.info("Input is not stereo; falling back to mono mixdown.")
        channel_mode = "mono_mixdown"

    outputs = {}
    primary_key = None

    if channel_mode == "mono_mixdown":
        mono = to_mono(x).astype(np.float32)
        prog = st.progress(0.0, text="Running offline classification…")
        t_centers, scores = classify_single_waveform(
            interp,
            mono,
            sr,
            sample_hz=float(sample_hz),
            expected_len=expected_len,
            avg_frames=int(avg_frames),
            progress=prog,
            progress_start=0.0,
            progress_end=1.0,
            progress_text="Running mono mixdown classification…",
        )
        prog.empty()

        outputs["mono"] = {"t_sec": t_centers, "scores": scores}
        primary_key = "mono"

    elif channel_mode == "stereo_separate":
        prog = st.progress(0.0, text="Running LEFT channel classification…")

        t_left, s_left = classify_single_waveform(
            interp,
            channels["left"],
            sr,
            sample_hz=float(sample_hz),
            expected_len=expected_len,
            avg_frames=int(avg_frames),
            progress=prog,
            progress_start=0.0,
            progress_end=0.5,
            progress_text="Running LEFT channel classification…",
        )

        t_right, s_right = classify_single_waveform(
            interp,
            channels["right"],
            sr,
            sample_hz=float(sample_hz),
            expected_len=expected_len,
            avg_frames=int(avg_frames),
            progress=prog,
            progress_start=0.5,
            progress_end=1.0,
            progress_text="Running RIGHT channel classification…",
        )
        prog.empty()

        if len(t_left) != len(t_right) or not np.allclose(t_left, t_right, atol=1e-4):
            raise ValueError("Left/right channel frame grids do not align.")

        outputs["left"] = {"t_sec": t_left, "scores": s_left}
        outputs["right"] = {"t_sec": t_right, "scores": s_right}
        primary_key = None

    elif channel_mode == "stereo_average":
        prog = st.progress(0.0, text="Running LEFT channel classification…")

        t_left, s_left = classify_single_waveform(
            interp,
            channels["left"],
            sr,
            sample_hz=float(sample_hz),
            expected_len=expected_len,
            avg_frames=int(avg_frames),
            progress=prog,
            progress_start=0.0,
            progress_end=0.5,
            progress_text="Running LEFT channel classification…",
        )

        t_right, s_right = classify_single_waveform(
            interp,
            channels["right"],
            sr,
            sample_hz=float(sample_hz),
            expected_len=expected_len,
            avg_frames=int(avg_frames),
            progress=prog,
            progress_start=0.5,
            progress_end=1.0,
            progress_text="Running RIGHT channel classification…",
        )
        prog.empty()

        if len(t_left) != len(t_right) or not np.allclose(t_left, t_right, atol=1e-4):
            raise ValueError("Left/right channel frame grids do not align.")

        s_avg = 0.5 * (s_left + s_right)

        outputs["left"] = {"t_sec": t_left, "scores": s_left}
        outputs["right"] = {"t_sec": t_right, "scores": s_right}
        outputs["avg"] = {"t_sec": t_left, "scores": s_avg}
        primary_key = "avg"

    else:
        raise ValueError(f"Unknown channel_mode={channel_mode}")

    for name, out in outputs.items():
        if out["scores"].shape[1] != len(labels):
            raise ValueError(
                f"Mismatch for {name}: model outputs {out['scores'].shape[1]} classes "
                f"but labels has {len(labels)}."
            )

    return {
        "requested_mode": requested_mode,
        "effective_mode": channel_mode,
        "labels": labels,
        "outputs": outputs,
        "primary_key": primary_key,
    }


def scores_to_long_df(t_centers: np.ndarray, scores: np.ndarray, labels: List[str], top_k: int) -> pd.DataFrame:
    if len(t_centers) == 0 or scores.size == 0:
        return pd.DataFrame(columns=["t_sec", "rank", "label", "score"])

    top_k = min(int(top_k), scores.shape[1])

    idx = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
    top_scores = np.take_along_axis(scores, idx, axis=1)

    order = np.argsort(-top_scores, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    top_scores = np.take_along_axis(top_scores, order, axis=1)

    n_frames = len(t_centers)
    out = pd.DataFrame({
        "t_sec": np.repeat(t_centers.astype(float), top_k),
        "rank": np.tile(np.arange(1, top_k + 1), n_frames),
        "label": np.array(labels, dtype=object)[idx.reshape(-1)],
        "score": top_scores.reshape(-1).astype(float),
    })
    return out


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


def plot_label_presence_over_time(
    df_long: pd.DataFrame,
    title: str,
    min_score: float = 0.0,
    max_labels: int = 25,
    binary: bool = False,
):
    """
    Heatmap over time:
      x = time
      y = label
      z = score or binary presence
    Uses only labels present in the saved top-k rows.
    """
    if df_long.empty:
        st.info("No data available for timeline plot.")
        return

    plot_df = df_long.copy()
    if min_score > 0:
        plot_df = plot_df[plot_df["score"] >= float(min_score)].copy()

    if plot_df.empty:
        st.info("No labels remain after applying the confidence filter.")
        return

    summary = (
        plot_df.groupby("label")
        .agg(
            frames_present=("t_sec", "nunique"),
            mean_score=("score", "mean"),
            max_score=("score", "max"),
        )
        .sort_values(["frames_present", "mean_score", "max_score"], ascending=False)
    )

    keep_labels = summary.head(int(max_labels)).index.tolist()
    plot_df = plot_df[plot_df["label"].isin(keep_labels)].copy()

    if binary:
        plot_df["value"] = 1.0
    else:
        plot_df["value"] = plot_df["score"].astype(float)

    mat = plot_df.pivot_table(
        index="label",
        columns="t_sec",
        values="value",
        aggfunc="max",
        fill_value=0.0,
    )

    mat = mat.reindex(index=keep_labels)

    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=go.Heatmap(
                z=mat.to_numpy(),
                x=mat.columns.astype(float),
                y=mat.index.tolist(),
                colorbar_title="Present" if binary else "Score",
                zmin=0.0,
                zmax=1.0 if binary else None,
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Audio category label",
            height=max(350, 18 * len(mat.index) + 140),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("Plotly heatmap unavailable. Install plotly to see the label timeline plot.")
        st.dataframe(mat, use_container_width=True)

# --- UI Helper Functs ---

def _small_frame_for_display(df_long: pd.DataFrame, t_near: float, min_conf: float) -> pd.DataFrame:
    frame = (
        df_long[df_long["t_sec"] == t_near]
        .sort_values("rank")[["rank", "label", "score"]]
        .copy()
        .reset_index(drop=True)
    )
    if min_conf > 0:
        frame = frame[frame["score"] >= min_conf].reset_index(drop=True)

    # IMPORTANT: prevent Streamlit trying to serialize huge/non-JSON attrs
    frame.attrs = {}
    return frame

def _strip_attrs_inplace(df: pd.DataFrame):
    schema = df.attrs.get("schema", None)
    df.attrs = {}
    if schema is not None:
        df.attrs["schema"] = schema

def _safe_key(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", s)

def _cmp_live_fragment(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_name: str,
    right_name: str,
    audio_bytes: bytes,
    audio_name: str,
    min_conf: float,
    poll_ms: int,
    epoch: int,
):
    # Make the component key change on mode switch so old iframe dies cleanly
    player_key = f"cmp_player_{_safe_key(left_name)}_{_safe_key(right_name)}_{epoch}"

    t_pick = playback_time_seconds(
        audio_bytes,
        audio_name=(audio_name or "audio.wav"),
        key=player_key,
        poll_ms=poll_ms,
    )
    st.caption(f"Playback time: {t_pick:.2f}s")

    # Times arrays (fast)
    times_L = np.sort(df_left.loc[df_left["rank"] == 1, "t_sec"].unique().astype(float))
    times_R = np.sort(df_right.loc[df_right["rank"] == 1, "t_sec"].unique().astype(float))
    if len(times_L) == 0 or len(times_R) == 0:
        st.info("No frames.")
        return

    tL = nearest_frame_time(times_L, float(t_pick))
    tR = nearest_frame_time(times_R, float(t_pick))

    # Top-1 rows (guard just in case)
    top1_L_rows = df_left[(df_left["rank"] == 1) & (df_left["t_sec"] == tL)][["label", "score"]]
    top1_R_rows = df_right[(df_right["rank"] == 1) & (df_right["t_sec"] == tR)][["label", "score"]]
    if top1_L_rows.empty or top1_R_rows.empty:
        st.info("Missing top-1 row for one of the selected frames.")
        return

    top1_L = top1_L_rows.iloc[0]
    top1_R = top1_R_rows.iloc[0]

    st.markdown(
        f"""
        **Nearest frames**  
        - {left_name}: **{tL:.3f}s** → **{top1_L['label']}** ({float(top1_L['score']):.3f})  
        - {right_name}: **{tR:.3f}s** → **{top1_R['label']}** ({float(top1_R['score']):.3f})
        """
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.caption(left_name)
        frameL = _small_frame_for_display(df_left, tL, min_conf)
        st.table(frameL)

    with c2:
        st.caption(right_name)
        frameR = _small_frame_for_display(df_right, tR, min_conf)
        st.table(frameR)

def _live_one_fragment(
    df_long: pd.DataFrame,
    title: str,
    audio_bytes: bytes,
    audio_name: str,
    min_conf: float,
    poll_ms: int,
    epoch: int,
):
    # Key changes when mode changes (epoch), which kills old iframes.
    player_key = f"one_player_{_safe_key(title)}_{epoch}"

    t_pick = playback_time_seconds(
        audio_bytes,
        audio_name=(audio_name or "audio.wav"),
        key=player_key,
        poll_ms=poll_ms,
    )
    st.caption(f"Playback time: {t_pick:.2f}s")

    times = np.sort(df_long.loc[df_long["rank"] == 1, "t_sec"].unique().astype(float))
    if len(times) == 0:
        st.info("No frames.")
        return

    t_near = nearest_frame_time(times, float(t_pick))
    frame = _small_frame_for_display(df_long, t_near, min_conf)

    st.write(f"Frame @ **{t_near:.3f}s** (nearest to playback)")
    st.table(frame)

def load_annotation_df(uploaded, top_k: int, labels: List[str]):
    data = uploaded.getvalue()
    name = uploaded.name

    if name.lower().endswith(".npz"):
        df, t_full, scores_full, labels_full = parse_predictions_npz_bytes(data, top_k=top_k)
        return df, {
            "full_t_sec": t_full,
            "full_scores": scores_full,
            "full_labels": labels_full,
        }

    df = parse_annotations_bytes(
        name=name,
        data=data,
        android_top_k=top_k,
        canonical_labels=tuple(labels),
    )
    return df, {}

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Offline vs Real-time AudioSet (YAMNet) Viewer", layout="wide")
st.title("Offline vs Real-time AudioSet (YAMNet) Dashboard")

st.caption("See the Help page for metric definitions, parameter explanations, and AudioSet/YAMNet notes.")

st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "Choose a workflow",
    [
        "Offline from audio",
        "View annotations",
        "Compare RT vs offline",
    ]
)

if "_last_mode" not in st.session_state:
    st.session_state["_last_mode"] = mode
if "_mode_epoch" not in st.session_state:
    st.session_state["_mode_epoch"] = 0

if st.session_state["_last_mode"] != mode:
    st.session_state["_last_mode"] = mode
    st.session_state["_mode_epoch"] += 1

st.sidebar.header("Channel handling")
channel_mode = st.sidebar.selectbox(
    "Stereo handling",
    options=["stereo_separate", "stereo_average", "mono_mixdown"],
    format_func=lambda x: {
        "stereo_separate": "Stereo: LEFT/RIGHT separate",
        "stereo_average": "Stereo: LEFT/RIGHT average",
        "mono_mixdown": "Mono",
    }[x],
)

st.sidebar.header("Audio Playback Settings")
poll_ms = st.sidebar.slider("Playback sync rate (ms)", 100, 2000, 1000, 50)

model_path = "yamnet.tflite"
labels_path = "yamnet_class_map.csv"
sample_hz = 1.0
window_s = 0.975
avg_frames = 1
top_k = 5
labels = []

st.sidebar.header("Other Settings/Information")
with st.sidebar.expander("Advanced model settings", expanded=False):
    st.markdown("**Model / labels**")
    model_path = st.text_input("Model path", value="yamnet.tflite")
    labels_path = st.text_input("Label map path", value="yamnet_class_map.csv")

    st.markdown("**Offline params**")
    st.caption("sample_hz: 1.0")
    st.caption("window_s: 0.975")
    st.caption("avg_frames: 1")
    st.caption("top_k: 5")

# Load labels once
try:
    labels = load_yamnet_labels_csv(labels_path)
except Exception as e:
    st.sidebar.error(f"Could not load labels from {labels_path}: {e}")
    labels = []

with st.sidebar.expander("Timeline settings", expanded=False):
    st.markdown("**Timeline plot**")
    timeline_binary = st.checkbox(
        "Show binary presence instead of score",
        value=False,
    )
    timeline_max_labels = st.slider(
        "Max labels in timeline plot",
        min_value=5,
        max_value=60,
        value=25,
        step=1,
    )
    st.markdown("**Display**")
    min_conf = st.slider("Min confidence (filter)", 0.0, 1.0, 0.0, 0.01)


with st.sidebar.expander("Comparison settings", expanded=False):
    st.markdown("**Real-time alignment**")
    rt_time_shift = st.number_input(
        "Real-time time shift (sec) applied to uploaded RT annotations",
        value=-0.975/2,  # default: capture time is END of window -> subtract half window
        step=0.01,
        disabled=True
    )


# --- Wrap as fragments (recompute wrapping when poll_ms changes) ---
HAS_FRAGMENT = hasattr(st, "fragment")
if HAS_FRAGMENT:
    _cmp_live_fragment = st.fragment(_cmp_live_fragment)
    _live_one_fragment = st.fragment(_live_one_fragment)

def stepper_view(df_long: pd.DataFrame, title: str, audio_bytes: Optional[bytes] = None, audio_name: str = ""):
    st.subheader(title)

    if df_long.empty:
        st.info("No predictions loaded.")
        return

    top1 = to_top1(df_long)
    if top1.empty:
        st.info("No rank=1 rows found in predictions.")
        return

    rendered_live_audio = False

    if audio_bytes is not None:
        if not HAS_FRAGMENT:
            st.warning("Your Streamlit version doesn't support fragments; using slider fallback.")
        else:
            _live_one_fragment(
                df_long=df_long,
                title=title,
                audio_bytes=audio_bytes,
                audio_name=audio_name,
                min_conf=min_conf,
                poll_ms=poll_ms,
                epoch=int(st.session_state["_mode_epoch"]),
            )
            rendered_live_audio = True

    if not rendered_live_audio:
        times = top1["t_sec"].to_numpy(dtype=float)
        tmin, tmax = float(times.min()), float(times.max())
        t_pick = st.slider("Scrub time (s)", tmin, tmax, tmin, key=f"scrub_fallback_{title}")
        t_near = nearest_frame_time(times, float(t_pick))
        frame = _small_frame_for_display(df_long, t_near, min_conf)
        st.write(f"Frame @ **{t_near:.3f}s** (nearest to scrub time)")
        st.table(frame)

    with st.expander("Label presence over time", expanded=False):
        plot_label_presence_over_time(
            df_long,
            title=f"{title} — label timeline",
            min_score=min_conf,
            max_labels=timeline_max_labels,
            binary=timeline_binary,
        )


def side_by_side_view(df_left, df_right, left_name, right_name, audio_bytes=None, audio_name=""):
    st.subheader("Side-by-side comparison")

    if df_left.empty or df_right.empty:
        st.info("Need both datasets loaded.")
        return

    rendered_live_audio = False

    if audio_bytes is not None:
        if not HAS_FRAGMENT:
            st.warning("Your Streamlit version doesn't support fragments; using slider fallback.")
        else:
            epoch = int(st.session_state.get("_mode_epoch", 0))
            _cmp_live_fragment(
                df_left=df_left,
                df_right=df_right,
                left_name=left_name,
                right_name=right_name,
                audio_bytes=audio_bytes,
                audio_name=audio_name,
                min_conf=min_conf,
                poll_ms=int(poll_ms),
                epoch=epoch,
            )
            rendered_live_audio = True

    if not rendered_live_audio:
        top1_L = to_top1(df_left)
        top1_R = to_top1(df_right)

        times_L = top1_L["t_sec"].to_numpy(dtype=float)
        times_R = top1_R["t_sec"].to_numpy(dtype=float)
        tmin = float(min(times_L.min(), times_R.min()))
        tmax = float(max(times_L.max(), times_R.max()))

        t_pick = st.slider("Scrub time (s)", tmin, tmax, tmin, key=f"cmp_fallback_{left_name}_{right_name}")
        tL = nearest_frame_time(times_L, float(t_pick))
        tR = nearest_frame_time(times_R, float(t_pick))

        rowL = top1_L[top1_L["t_sec"] == tL].iloc[0]
        rowR = top1_R[top1_R["t_sec"] == tR].iloc[0]

        st.markdown(
            f"""
            **Nearest frames**  
            - {left_name}: **{tL:.3f}s** → **{rowL['top1']}** ({rowL['score1']:.3f})  
            - {right_name}: **{tR:.3f}s** → **{rowR['top1']}** ({rowR['score1']:.3f})
            """
        )

        c1, c2 = st.columns(2, gap="large")
        with c1:
            frameL = _small_frame_for_display(df_left, tL, min_conf)
            st.table(frameL)
        with c2:
            frameR = _small_frame_for_display(df_right, tR, min_conf)
            st.table(frameR)

    with st.expander("Label presence over time", expanded=False):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            plot_label_presence_over_time(
                df_left,
                title=f"{left_name} — label timeline",
                min_score=min_conf,
                max_labels=timeline_max_labels,
                binary=timeline_binary,
            )
        with c2:
            plot_label_presence_over_time(
                df_right,
                title=f"{right_name} — label timeline",
                min_score=min_conf,
                max_labels=timeline_max_labels,
                binary=timeline_binary,
            )

# -----------------------------
# Mode implementations
# -----------------------------

if mode == "Offline from audio":
    st.write("Upload an audio file, run offline YAMNet classification, then step through predictions.")

    audio_up = st.file_uploader("Upload audio (WAV recommended)", type=["wav", "flac", "ogg", "aiff", "aif", "aifc", "mp3", "m4a", "mp4"])
    if audio_up is None:
        st.info("Upload an audio file to begin.")
        st.stop()

    audio_bytes = audio_up.getvalue()
    current_audio_hash = hashlib.md5(audio_bytes).hexdigest()
    if st.session_state.get("offline_audio_hash") != current_audio_hash:
        st.session_state.pop("offline_result", None)
        st.session_state["offline_audio_hash"] = current_audio_hash

    if not labels:
        st.error("Label map not loaded. Fix label map path in the sidebar.")
        st.stop()

    run_btn = st.button("Run offline classification", type="primary")
    if run_btn:
        try:
            result = classify_audio_bytes(
                audio_bytes=audio_bytes,
                audio_name=audio_up.name,
                model_path=model_path,
                labels=labels,
                sample_hz=float(sample_hz),
                window_s=float(window_s),
                avg_frames=int(avg_frames),
                channel_mode=channel_mode,
            )
            st.session_state["offline_result"] = result
        except Exception as e:
            st.exception(e)
            st.stop()

    offline_result = st.session_state.get("offline_result", None)

    if offline_result is not None:
        mode_used = offline_result["effective_mode"]

        if "left" in offline_result["outputs"] and "right" in offline_result["outputs"]:
            show_stereo_channel_metrics(
                offline_result,
                top_k_compare=int(top_k),
                heading="Stereo channel agreement (LEFT vs RIGHT)",
            )

        if mode_used == "stereo_separate":
            df_left = result_output_to_df(offline_result, "left", top_k=int(top_k))
            df_right = result_output_to_df(offline_result, "right", top_k=int(top_k))

            side_by_side_view(
                df_left,
                df_right,
                left_name="OFFLINE LEFT",
                right_name="OFFLINE RIGHT",
                audio_bytes=audio_bytes,
                audio_name=audio_up.name,
            )

            c1, c2 = st.columns(2)

            with c1:
                st.download_button(
                    "Download LEFT predictions.jsonl",
                    data=df_long_to_jsonl_bytes(df_left),
                    file_name="predictions_left.jsonl",
                    mime="application/jsonl",
                )
                st.download_button(
                    "Download LEFT full vectors (.npz)",
                    data=npz_bytes_from_scores(
                        offline_result["outputs"]["left"]["t_sec"],
                        offline_result["outputs"]["left"]["scores"],
                        offline_result["labels"],
                    ),
                    file_name="predictions_left_full.npz",
                    mime="application/octet-stream",
                )

            with c2:
                st.download_button(
                    "Download RIGHT predictions.jsonl",
                    data=df_long_to_jsonl_bytes(df_right),
                    file_name="predictions_right.jsonl",
                    mime="application/jsonl",
                )
                st.download_button(
                    "Download RIGHT full vectors (.npz)",
                    data=npz_bytes_from_scores(
                        offline_result["outputs"]["right"]["t_sec"],
                        offline_result["outputs"]["right"]["scores"],
                        offline_result["labels"],
                    ),
                    file_name="predictions_right_full.npz",
                    mime="application/octet-stream",
                )
        else:
            primary_key = offline_result["primary_key"]
            df_primary = result_output_to_df(offline_result, primary_key, top_k=int(top_k))

            title = "OFFLINE predictions"
            if primary_key == "avg":
                title += " (averaged stereo output)"
            elif primary_key == "mono":
                title += " (mono mixdown)"

            stepper_view(df_primary, title, audio_bytes=audio_bytes, audio_name=audio_up.name)

            st.download_button(
                f"Download {primary_key.upper()} predictions.jsonl",
                data=df_long_to_jsonl_bytes(df_primary),
                file_name=f"predictions_{primary_key}.jsonl",
                mime="application/jsonl",
            )

            st.download_button(
                f"Download {primary_key.upper()} full vectors (.npz)",
                data=npz_bytes_from_scores(
                    offline_result["outputs"][primary_key]["t_sec"],
                    offline_result["outputs"][primary_key]["scores"],
                    offline_result["labels"],
                ),
                file_name=f"predictions_{primary_key}_full.npz",
                mime="application/octet-stream",
            )

            if mode_used == "stereo_average":
                with st.expander("Inspect LEFT vs RIGHT channel predictions", expanded=False):
                    df_left = result_output_to_df(offline_result, "left", top_k=int(top_k))
                    df_right = result_output_to_df(offline_result, "right", top_k=int(top_k))
                    side_by_side_view(
                        df_left,
                        df_right,
                        left_name="OFFLINE LEFT",
                        right_name="OFFLINE RIGHT",
                        audio_bytes=None,
                        audio_name="",
                    )


elif mode == "View annotations":
    st.write("Upload an annotations file (JSONL/JSON/CSV) and step through predictions.")
    ann_up = st.file_uploader("Upload annotations file", type=["jsonl", "json", "csv", "npz"])
    audio_up = st.file_uploader("(Optional) Upload audio for playback", type=["wav", "flac", "ogg", "aiff", "aif", "aifc", "mp3", "m4a", "mp4"])

    audio_bytes = audio_up.getvalue() if audio_up is not None else None

    if ann_up is None:
        st.info("Upload an annotations file to begin.")
        st.stop()

    try:
        df_ann, extras = load_annotation_df(ann_up, int(top_k), labels)
        for k, v in extras.items():
            df_ann.attrs[k] = v
        default_shift = (df_ann.attrs.get("schema") == "android_wide_converted")
        apply_shift = st.checkbox("Apply time shift to this annotation file", value=default_shift)
        if apply_shift:
            df_ann = apply_time_shift(df_ann, shift_sec=float(rt_time_shift))
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
    _strip_attrs_inplace(df_ann)
    stepper_view(df_ann, f"Annotations: {ann_up.name}", audio_bytes=audio_bytes, audio_name=(audio_up.name if audio_up else ""))

elif mode == "Compare RT vs offline":
    st.write("Upload audio + real-time annotations, run offline classification, then compare side-by-side.")

    audio_up = st.file_uploader("Upload audio (WAV recommended)", type=["wav", "flac", "ogg", "aiff", "aif", "aifc", "mp3", "m4a", "mp4"])
    ann_up = st.file_uploader("Upload REAL-TIME output annotations (JSONL/JSON/CSV)", type=["jsonl", "json", "csv", "npz"])

    if audio_up is None or ann_up is None:
        st.info("Upload both an audio file and an annotations file to begin.")
        st.stop()

    audio_bytes = audio_up.getvalue()
    current_compare_key = hashlib.md5(audio_bytes + ann_up.getvalue()).hexdigest()
    if st.session_state.get("compare_input_hash") != current_compare_key:
        st.session_state.pop("compare_off_result", None)
        st.session_state.pop("compare_rt_df", None)
        st.session_state["compare_input_hash"] = current_compare_key

    # Parse real-time annotations
    try:
        df_rt, extras = load_annotation_df(ann_up, int(top_k), labels)
        for k, v in extras.items():
            df_rt.attrs[k] = v
        
        df_rt = apply_time_shift(df_rt, shift_sec=float(rt_time_shift))
    except Exception as e:
        st.exception(e)
        st.stop()

    if not labels:
        st.sidebar.error("Label map not loaded. Fix label map path in the sidebar.")
        st.stop()

    run_btn = st.button("Run OFFLINE and compare", type="primary")
    if run_btn:
        try:
            off_result = classify_audio_bytes(
                audio_bytes=audio_bytes,
                audio_name=audio_up.name,
                model_path=model_path,
                labels=labels,
                sample_hz=float(sample_hz),
                window_s=float(window_s),
                avg_frames=int(avg_frames),
                channel_mode=channel_mode,
            )
            _strip_attrs_inplace(df_rt)
            st.session_state["compare_off_result"] = off_result
            st.session_state["compare_rt_df"] = df_rt
        except Exception as e:
            st.exception(e)
            st.stop()

    off_result = st.session_state.get("compare_off_result", None)
    df_rt2 = st.session_state.get("compare_rt_df", pd.DataFrame())


    if off_result is not None and not df_rt2.empty:
        mode_used = off_result["effective_mode"]

        tab1, tab2 = st.tabs(["Compare", "Metrics"])

        def show_metrics_block(df_ref: pd.DataFrame, label: str):
            st.markdown(f"### {label}")
            k_compare = int(top_k)
            m = compute_compare_metrics(df_rt2, df_ref, k=k_compare)
            
            if m:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Frames compared", m["frames_compared"])
                c2.metric(f"Avg same in Top-{k_compare}", f'{m["avg_same_in_topk"]:.2f}')
                c3.metric("Top-1 match rate", f'{100*m["top1_match_rate"]:.1f}%')
                c4.metric("Top-2 match rate", f'{100*m["top2_match_rate"]:.1f}%')
                c5.metric("Avg |Δscore| (common)", f'{m["avg_abs_score_diff_on_common_labels"]:.3f}')
                st.caption(f'Avg Jaccard(Top-{k_compare}): {m["avg_jaccard_topk"]:.3f}')
            sd_ref = compute_stddev_of_category_means(df_ref, k=k_compare)
            sd_rt = compute_stddev_of_category_means(df_rt2, k=k_compare)
            st.caption(
                f"Std dev of per-label mean scores (Top-{k_compare} approx) — "
                f"RT: {sd_rt:.4f}, OFF: {sd_ref:.4f}"
            )
        
        if mode_used == "stereo_separate":
            df_left = result_output_to_df(off_result, "left", top_k=int(top_k))
            df_right = result_output_to_df(off_result, "right", top_k=int(top_k))
        else:
            primary_key = off_result["primary_key"]
            df_off = result_output_to_df(off_result, primary_key, top_k=int(top_k))

            label = "REAL-TIME vs OFFLINE"
            if primary_key == "avg":
                label += " (averaged stereo output)"
            elif primary_key == "mono":
                label += " (mono mixdown)"

        with tab1: #Compare
            if mode_used == "stereo_separate":
                inspect_channel = st.radio(
                    "Select offline channel to compare to real-time:",
                    options=["left", "right"],
                    horizontal=False,
                )
                df_inspect = df_left if inspect_channel == "left" else df_right

                side_by_side_view(
                    df_rt2,
                    df_inspect,
                    left_name=f"REAL-TIME ({ann_up.name})",
                    right_name=f"OFFLINE {inspect_channel.upper()}",
                    audio_bytes=audio_bytes,
                    audio_name=audio_up.name,
                )
            else:
                side_by_side_view(
                    df_rt2,
                    df_off,
                    left_name=f"REAL-TIME ({ann_up.name})",
                    right_name=f"OFFLINE {primary_key.upper()}",
                    audio_bytes=audio_bytes,
                    audio_name=audio_up.name,
                )

                if mode_used == "stereo_average":
                    with st.expander("Inspect LEFT vs RIGHT offline channels", expanded=False):
                        df_left = result_output_to_df(off_result, "left", top_k=int(top_k))
                        df_right = result_output_to_df(off_result, "right", top_k=int(top_k))
                        side_by_side_view(
                            df_left,
                            df_right,
                            left_name="OFFLINE LEFT",
                            right_name="OFFLINE RIGHT",
                            audio_bytes=None,
                            audio_name="",
                        )
        with tab2: #Metrics
            if mode_used == "stereo_separate":
                if "left" in off_result["outputs"] and "right" in off_result["outputs"]:
                    show_stereo_channel_metrics(
                        off_result,
                        top_k_compare=int(top_k),
                        heading="Stereo channel agreement inside OFFLINE output (LEFT vs RIGHT)",
                    )
                show_metrics_block(df_left, "REAL-TIME vs OFFLINE LEFT")
                show_metrics_block(df_right, "REAL-TIME vs OFFLINE RIGHT")
            else:
                show_metrics_block(df_off, label)
        