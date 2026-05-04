import argparse
import csv
import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.signal

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

Interpreter = None
_INTERPRETER_CLASS = None


DEFAULT_CONTEXT_RULES_PATH = Path(__file__).resolve().parent / "context_rules.json"


def get_interpreter_class():
    global Interpreter, _INTERPRETER_CLASS
    if _INTERPRETER_CLASS is not None:
        return _INTERPRETER_CLASS
    try:
        import tflite_runtime.interpreter as tflite

        _INTERPRETER_CLASS = tflite.Interpreter
    except ImportError:
        try:
            import tensorflow as tf

            _INTERPRETER_CLASS = tf.lite.Interpreter
        except ImportError:
            _INTERPRETER_CLASS = None
    Interpreter = _INTERPRETER_CLASS
    return _INTERPRETER_CLASS


def create_interpreter(model_path: str):
    interpreter_class = get_interpreter_class()
    if interpreter_class is None:
        raise RuntimeError("Install tensorflow, tflite_runtime, or ai_edge_litert to load the YAMNet model.")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tf\\.lite\\.Interpreter is deprecated.*")
        return interpreter_class(model_path=model_path)


def get_model_expected_len(model_path: str) -> int:
    interp = create_interpreter(model_path)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    return int(input_details[0]["shape"][-1])


def load_labels(labels_path: str) -> List[str]:
    if labels_path.lower().endswith(".csv"):
        labels = []
        with open(labels_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(row.get("display_name") or row.get("name") or row[list(row.keys())[-1]])
        return labels

    if labels_path.lower().endswith(".json"):
        with open(labels_path, "r") as f:
            data = json.load(f)

        candidates = [
            data.get("labels"),
            data.get("class_labels"),
            data.get("class_map"),
            data.get("classes"),
            (data.get("yamnet") or {}).get("labels"),
            (data.get("yamnet") or {}).get("class_map"),
        ]

        for candidate in candidates:
            if not candidate:
                continue

            if isinstance(candidate, list) and all(isinstance(x, str) for x in candidate):
                return candidate

            if isinstance(candidate, list) and all(isinstance(x, dict) for x in candidate):
                out = [x.get("display_name") or x.get("name") or x.get("label") for x in candidate]
                if all(out):
                    return out

            if isinstance(candidate, dict):
                keys_are_intlike = all(str(k).isdigit() for k in candidate.keys())
                if keys_are_intlike:
                    return [candidate[str(i)] if str(i) in candidate else candidate[i] for i in range(len(candidate))]

        raise ValueError(f"Could not locate labels inside {labels_path}")

    raise ValueError("Labels file must be .csv or .json")


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    if sf is not None:
        try:
            waveform, sr = sf.read(audio_path, always_2d=False)
            return np.asarray(waveform), int(sr)
        except RuntimeError:
            pass
        except sf.LibsndfileError:
            pass

    if AudioSegment is None:
        raise RuntimeError("Install soundfile or pydub from requirements.txt to read audio files.")

    segment = AudioSegment.from_file(audio_path)
    sr = int(segment.frame_rate)
    sample_width = int(segment.sample_width)
    channels = int(segment.channels)

    raw = np.array(segment.get_array_of_samples())
    if channels > 1:
        raw = raw.reshape(-1, channels)

    scale = float(1 << (8 * sample_width - 1))
    waveform = raw.astype(np.float32) / scale
    return waveform, sr


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
        chunk = x[i : i + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        frames.append(chunk)
        centers.append((i + win / 2) / sr)
        i += hop
    return np.stack(frames), np.array(centers, dtype=np.float32)


def run_tflite(interp: Any, frames: np.ndarray) -> np.ndarray:
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    scores = []
    expects_batched = input_details[0]["shape"].ndim == 2
    for frame in frames:
        x = frame.astype(np.float32)
        if expects_batched:
            x = x[None, :]
        interp.set_tensor(input_details[0]["index"], x)
        interp.invoke()
        y = interp.get_tensor(output_details[0]["index"])
        scores.append(np.squeeze(y))
    return np.stack(scores).astype(np.float32)


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


def classify_waveform(
    interp: Any,
    waveform: np.ndarray,
    sr: int,
    sample_hz: float,
    window_s: float,
    avg_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(waveform, dtype=np.float32)
    if np.max(np.abs(x)) > 1.5:
        x = x / 32768.0
    x = resample(x, sr, 16000)
    hop_s = 1.0 / sample_hz
    frames, t_centers = frame_waveform(x, 16000, window_s, hop_s)
    raw_scores = run_tflite(interp, frames)
    return t_centers, temporal_average(raw_scores, avg_frames)


def normalize_token(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("_", " ").split())


def validate_context_rules_config(config: Dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("Context rules config must be a JSON object.")

    label_pattern_sets = config.get("label_pattern_sets")
    if not isinstance(label_pattern_sets, dict) or not label_pattern_sets:
        raise ValueError("Context rules config requires a non-empty label_pattern_sets object.")

    for group_name, patterns in label_pattern_sets.items():
        if not isinstance(group_name, str) or not group_name.strip():
            raise ValueError("Every label pattern group must have a non-empty string name.")
        if not isinstance(patterns, list) or not all(isinstance(item, str) and item.strip() for item in patterns):
            raise ValueError(f"Label pattern group '{group_name}' must be a list of non-empty strings.")

    rules = config.get("context_rules")
    if not isinstance(rules, list):
        raise ValueError("Context rules config requires context_rules as a list.")

    known_groups = set(label_pattern_sets)
    for idx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"Rule #{idx + 1} must be an object.")
        name = rule.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Rule #{idx + 1} requires a non-empty name.")
        signals = rule.get("signals")
        if not isinstance(signals, list) or not all(isinstance(item, str) and item.strip() for item in signals):
            raise ValueError(f"Rule '{name}' requires signals as a list of non-empty strings.")

        for section in ("boost", "suppress"):
            values = rule.get(section, {})
            if values is None:
                continue
            if not isinstance(values, dict):
                raise ValueError(f"Rule '{name}' field '{section}' must be an object.")
            unknown_groups = sorted(set(values) - known_groups)
            if unknown_groups:
                raise ValueError(f"Rule '{name}' references unknown {section} group(s): {', '.join(unknown_groups)}.")
            for group_name, factor in values.items():
                try:
                    factor_value = float(factor)
                except (TypeError, ValueError):
                    raise ValueError(f"Rule '{name}' {section}.{group_name} must be numeric.") from None
                if factor_value <= 0:
                    raise ValueError(f"Rule '{name}' {section}.{group_name} must be greater than 0.")

        suppress_labels = rule.get("suppress_labels", {})
        if suppress_labels is None:
            continue
        if not isinstance(suppress_labels, dict):
            raise ValueError(f"Rule '{name}' field 'suppress_labels' must be an object.")
        for label_term, factor in suppress_labels.items():
            if not isinstance(label_term, str) or not label_term.strip():
                raise ValueError(f"Rule '{name}' suppress_labels keys must be non-empty strings.")
            try:
                factor_value = float(factor)
            except (TypeError, ValueError):
                raise ValueError(f"Rule '{name}' suppress_labels.{label_term} must be numeric.") from None
            if factor_value <= 0:
                raise ValueError(f"Rule '{name}' suppress_labels.{label_term} must be greater than 0.")


def load_context_rules_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONTEXT_RULES_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    validate_context_rules_config(config)
    return config


@dataclass
class ContextSignal:
    source: str
    value: str
    confidence: float = 1.0


@dataclass
class RuleImpact:
    rule: str
    source: str
    label: str
    weight: float
    matched_terms: List[str] = field(default_factory=list)


@dataclass
class Segment:
    label: str
    start_sec: float
    end_sec: float
    peak_score: float
    mean_score: float
    frame_count: int
    score_type: str


@dataclass
class SemanticSummary:
    scene_tags: List[str]
    dominant_labels: List[Dict[str, float]]
    dominant_non_silence_labels: List[Dict[str, float]]
    salient_segments: List[Dict[str, Any]]
    semantic_scene: str
    narrative: str
    evidence: Dict[str, Any]


_CONTEXT_RULES_CONFIG = load_context_rules_config()
LABEL_PATTERN_SETS: Dict[str, List[str]] = _CONTEXT_RULES_CONFIG["label_pattern_sets"]
CONTEXT_RULES: List[Dict[str, Any]] = _CONTEXT_RULES_CONFIG["context_rules"]


LOW_VALUE_SCENE_LABEL_TERMS = {
    "silence",
    "inside, small room",
    "inside, large room or hall",
    "reverberation",
    "snoring",
    "breathing",
    "hiccup",
    "sneeze",
    "cough",
    "burping, eructation",
    "chewing, mastication",
}


def load_context(context_path: Optional[str]) -> Dict[str, Any]:
    if not context_path:
        return {}
    with open(context_path, "r") as f:
        return json.load(f)


def extract_context_signals(context: Dict[str, Any]) -> List[ContextSignal]:
    signals: List[ContextSignal] = []

    def add_signal(source: str, value: Any, confidence: float = 1.0) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add_signal(source, item, confidence)
            return
        text = normalize_token(value)
        if text:
            signals.append(ContextSignal(source=source, value=text, confidence=float(confidence)))

    tags = context.get("tags") or context.get("scene_tags") or context.get("manual_tags") or []
    add_signal("tags", tags, 1.0)

    scene = context.get("scene") or {}
    for key in ("environment", "setting", "place_type", "indoor_outdoor", "land_use"):
        add_signal(f"scene.{key}", scene.get(key), 0.95)

    location = context.get("location") or {}
    for key in ("display_name", "name", "category", "type", "road", "neighbourhood", "suburb", "city"):
        add_signal(f"location.{key}", location.get(key), 0.8)

    weather = context.get("weather") or {}
    for key in ("summary", "description", "weathercode_label", "condition"):
        add_signal(f"weather.{key}", weather.get(key), 0.95)

    for numeric_key, token in (
        ("precipitation_mm", "precipitation"),
        ("rain_mm", "rain"),
        ("windspeed_10m", "wind"),
        ("windgusts_10m", "gust"),
        ("cloudcover", "cloudy"),
    ):
        value = weather.get(numeric_key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if numeric_key in ("precipitation_mm", "rain_mm") and numeric > 0:
            add_signal(f"weather.{numeric_key}", token, min(1.0, 0.5 + numeric / 10.0))
        if numeric_key in ("windspeed_10m", "windgusts_10m") and numeric >= 10:
            add_signal(f"weather.{numeric_key}", token, min(1.0, numeric / 40.0))
        if numeric_key == "cloudcover" and numeric >= 75:
            add_signal(f"weather.{numeric_key}", token, min(1.0, numeric / 100.0))

    recording = context.get("recording") or {}
    for key in ("device", "notes", "source"):
        add_signal(f"recording.{key}", recording.get(key), 0.6)

    return signals


def match_label_indices(labels: Sequence[str], terms: Iterable[str]) -> List[int]:
    norm_labels = [normalize_token(label) for label in labels]
    indices = []
    for idx, label in enumerate(norm_labels):
        if any(term in label for term in terms):
            indices.append(idx)
    return indices


def build_label_groups(labels: Sequence[str], label_pattern_sets: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[int]]:
    groups = {}
    for name, patterns in (label_pattern_sets or LABEL_PATTERN_SETS).items():
        groups[name] = match_label_indices(labels, [normalize_token(term) for term in patterns])
    return groups


def build_context_adjustment(
    labels: Sequence[str],
    context: Dict[str, Any],
    rules_config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[RuleImpact], List[ContextSignal]]:
    if rules_config is not None:
        validate_context_rules_config(rules_config)
    label_pattern_sets = (rules_config or _CONTEXT_RULES_CONFIG)["label_pattern_sets"]
    context_rules = (rules_config or _CONTEXT_RULES_CONFIG)["context_rules"]
    weights = np.ones(len(labels), dtype=np.float32)
    impacts: List[RuleImpact] = []
    signals = extract_context_signals(context)
    signal_values = [signal.value for signal in signals]
    label_groups = build_label_groups(labels, label_pattern_sets)
    norm_labels = [normalize_token(label) for label in labels]

    for rule in context_rules:
        matched = [signal for signal in signals if any(token in signal.value for token in rule["signals"])]
        if not matched:
            continue

        rule_conf = max(signal.confidence for signal in matched)
        matched_terms = sorted({signal.value for signal in matched})

        for group_name, factor in rule.get("boost", {}).items():
            for idx in label_groups.get(group_name, []):
                effective = 1.0 + (factor - 1.0) * rule_conf
                weights[idx] *= effective
                impacts.append(
                    RuleImpact(
                        rule=rule["name"],
                        source="boost",
                        label=labels[idx],
                        weight=float(effective),
                        matched_terms=matched_terms,
                    )
                )

        for group_name, factor in rule.get("suppress", {}).items():
            for idx in label_groups.get(group_name, []):
                effective = 1.0 - (1.0 - factor) * rule_conf
                weights[idx] *= effective
                impacts.append(
                    RuleImpact(
                        rule=rule["name"],
                        source="suppress",
                        label=labels[idx],
                        weight=float(effective),
                        matched_terms=matched_terms,
                        )
                )

        for label_term, factor in rule.get("suppress_labels", {}).items():
            norm_term = normalize_token(label_term)
            for idx, norm_label in enumerate(norm_labels):
                if norm_term in norm_label:
                    effective = 1.0 - (1.0 - factor) * rule_conf
                    weights[idx] *= effective
                    impacts.append(
                        RuleImpact(
                            rule=rule["name"],
                            source="suppress",
                            label=labels[idx],
                            weight=float(effective),
                            matched_terms=matched_terms,
                        )
                    )

    if not signal_values:
        return weights, impacts, signals

    return np.clip(weights, 0.3, 2.5), impacts, signals


def top_k_labels(scores: np.ndarray, labels: Sequence[str], top_k: int) -> List[Dict[str, float]]:
    idx = np.argsort(-scores)[:top_k]
    return [{"label": str(labels[i]), "score": float(scores[i])} for i in idx]


def is_scene_noise_label(label: str) -> bool:
    normalized = normalize_token(label)
    return normalized in LOW_VALUE_SCENE_LABEL_TERMS


def top_k_non_silence_labels(
    scores: np.ndarray,
    labels: Sequence[str],
    top_k: int,
    low_value_override_ratio: float = 2.25,
) -> List[Dict[str, float]]:
    ranked = np.argsort(-scores)
    out: List[Dict[str, float]] = []
    best_mean = float(scores[ranked[0]]) if len(ranked) else 0.0
    for idx in ranked:
        label = str(labels[idx])
        score = float(scores[idx])
        if is_scene_noise_label(label) and (best_mean <= 0.0 or score < best_mean * low_value_override_ratio):
            continue
        out.append({"label": label, "score": score})
        if len(out) >= top_k:
            break
    return out


def summarize_global_labels(scores: np.ndarray, labels: Sequence[str], top_k: int) -> List[Dict[str, float]]:
    mean_scores = np.mean(scores, axis=0)
    return top_k_labels(mean_scores, labels, top_k)


def summarize_non_silence_labels(scores: np.ndarray, labels: Sequence[str], top_k: int) -> List[Dict[str, float]]:
    mean_scores = np.mean(scores, axis=0)
    return top_k_non_silence_labels(mean_scores, labels, top_k)


def non_silence_confidence(adjusted_non_silence_top: Sequence[Dict[str, float]], audio_stats: Dict[str, Any]) -> float:
    if not adjusted_non_silence_top:
        return 0.0
    top_score = float(adjusted_non_silence_top[0]["score"])
    top3_mean = float(np.mean([item["score"] for item in adjusted_non_silence_top[:3]]))
    rms = float(audio_stats.get("rms", 0.0))
    silent_fraction = float(audio_stats.get("near_silent_fraction", 1.0))
    confidence = top_score * 18.0 + top3_mean * 10.0 + min(rms * 8.0, 0.25) - silent_fraction * 0.2
    return max(0.0, min(1.0, confidence))


def detect_segments(
    t_sec: np.ndarray,
    scores: np.ndarray,
    labels: Sequence[str],
    top_labels: Sequence[str],
    threshold: float,
    score_type: str,
) -> List[Segment]:
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    segments: List[Segment] = []
    step = float(np.median(np.diff(t_sec))) if len(t_sec) > 1 else 1.0

    for label in top_labels:
        idx = label_to_index.get(label)
        if idx is None:
            continue
        active = scores[:, idx] >= threshold
        start = None
        values: List[float] = []
        for frame_idx, is_active in enumerate(active):
            if is_active and start is None:
                start = frame_idx
                values = [float(scores[frame_idx, idx])]
            elif is_active:
                values.append(float(scores[frame_idx, idx]))
            elif start is not None:
                end_idx = frame_idx - 1
                segments.append(
                    Segment(
                        label=label,
                        start_sec=float(max(0.0, t_sec[start] - step / 2)),
                        end_sec=float(t_sec[end_idx] + step / 2),
                        peak_score=float(max(values)),
                        mean_score=float(sum(values) / len(values)),
                        frame_count=len(values),
                        score_type=score_type,
                    )
                )
                start = None
                values = []
        if start is not None:
            end_idx = len(active) - 1
            segments.append(
                Segment(
                    label=label,
                    start_sec=float(max(0.0, t_sec[start] - step / 2)),
                    end_sec=float(t_sec[end_idx] + step / 2),
                    peak_score=float(max(values)),
                    mean_score=float(sum(values) / len(values)),
                    frame_count=len(values),
                    score_type=score_type,
                )
            )

    return sorted(segments, key=lambda seg: (-seg.peak_score, seg.start_sec))


def infer_scene_tags(context: Dict[str, Any], adjusted_top: Sequence[Dict[str, float]]) -> List[str]:
    tags = set()
    for signal in extract_context_signals(context):
        for token in ("urban", "rural", "indoor", "outdoor", "industrial", "residential", "office", "rain", "wind"):
            if token in signal.value:
                tags.add(token)

    label_names = [normalize_token(item["label"]) for item in adjusted_top]
    if any(any(term in label for term in ("traffic", "car", "vehicle", "bus", "engine")) for label in label_names):
        tags.add("traffic-dominant")
    if any(any(term in label for term in ("bird", "wind", "rain", "insect", "frog")) for label in label_names):
        tags.add("nature-present")
    if any("speech" in label or "conversation" in label for label in label_names):
        tags.add("human-activity")
    return sorted(tags)


def choose_semantic_scene(
    scene_tags: Sequence[str],
    adjusted_top: Sequence[Dict[str, float]],
    confidence: float,
    audio_stats: Dict[str, Any],
) -> str:
    text = " ".join(scene_tags)
    labels = " ".join(normalize_token(item["label"]) for item in adjusted_top)
    silent_fraction = float(audio_stats.get("near_silent_fraction", 1.0))
    if confidence < 0.18 or silent_fraction > 0.92:
        return "low-confidence ambient recording"
    if "indoor" in text and "office" in text:
        return "indoor office or classroom soundscape"
    if "industrial" in text:
        return "industrial or machinery-heavy environment"
    if "urban" in text and "traffic-dominant" in text:
        return "urban roadside soundscape"
    if "rural" in text or "nature-present" in text:
        return "outdoor natural ambience"
    if "residential" in text:
        return "residential mixed-use environment"
    if "speech" in labels:
        return "human-centered acoustic scene"
    return "mixed acoustic environment"


def build_narrative(
    scene: str,
    raw_top: Sequence[Dict[str, float]],
    adjusted_top: Sequence[Dict[str, float]],
    adjusted_non_silence_top: Sequence[Dict[str, float]],
    scene_tags: Sequence[str],
    context_signals: Sequence[ContextSignal],
    confidence: float,
    audio_stats: Dict[str, Any],
) -> str:
    adjusted_labels = ", ".join(item["label"] for item in adjusted_top[:3]) or "no dominant labels"
    adjusted_non_silence = ", ".join(item["label"] for item in adjusted_non_silence_top[:3]) or "no non-silence labels"
    raw_labels = ", ".join(item["label"] for item in raw_top[:3]) or "no dominant labels"
    tags_text = ", ".join(scene_tags) if scene_tags else "no explicit scene tags"
    signal_text = ", ".join(sorted({signal.value for signal in context_signals})[:4]) if context_signals else "no external context"
    silent_fraction = float(audio_stats.get("near_silent_fraction", 1.0))
    rms = float(audio_stats.get("rms", 0.0))
    if scene == "low-confidence ambient recording":
        return (
            f"The recording is best treated as a low-confidence ambient recording. "
            f"Non-silence evidence is weak ({adjusted_non_silence}) and the audio appears quiet "
            f"(rms={rms:.4f}, near_silent_fraction={silent_fraction:.3f}). "
            f"Context cues suggest {tags_text}, but the acoustic evidence is not strong enough to make a confident scene claim."
        )
    return (
        f"The recording is interpreted as a {scene}. "
        f"Context-adjusted evidence emphasizes {adjusted_labels}; excluding silence-like labels, the strongest events are {adjusted_non_silence}. "
        f"The raw acoustic model emphasized {raw_labels}. Confidence={confidence:.2f}. "
        f"Scene tags: {tags_text}. Context cues used: {signal_text}."
    )


def build_semantic_summary(
    t_sec: np.ndarray,
    raw_scores: np.ndarray,
    adjusted_scores: np.ndarray,
    labels: Sequence[str],
    top_k: int,
    context: Dict[str, Any],
    impacts: Sequence[RuleImpact],
    context_signals: Sequence[ContextSignal],
    audio_stats: Dict[str, Any],
) -> SemanticSummary:
    raw_top = summarize_global_labels(raw_scores, labels, top_k)
    adjusted_top = summarize_global_labels(adjusted_scores, labels, top_k)
    adjusted_non_silence_top = summarize_non_silence_labels(adjusted_scores, labels, top_k)
    salient_labels = [item["label"] for item in adjusted_non_silence_top[: min(5, top_k)]]
    reference_score = float(adjusted_non_silence_top[0]["score"]) if adjusted_non_silence_top else (float(adjusted_top[0]["score"]) if adjusted_top else 0.15)
    threshold = max(0.01, reference_score * 0.45)
    segments = detect_segments(t_sec, adjusted_scores, labels, salient_labels, threshold, "adjusted")
    scene_tags = infer_scene_tags(context, adjusted_non_silence_top)
    confidence = non_silence_confidence(adjusted_non_silence_top, audio_stats)
    semantic_scene = choose_semantic_scene(scene_tags, adjusted_non_silence_top, confidence, audio_stats)
    narrative = build_narrative(
        semantic_scene,
        raw_top,
        adjusted_top,
        adjusted_non_silence_top,
        scene_tags,
        context_signals,
        confidence,
        audio_stats,
    )

    top_impacts = sorted(impacts, key=lambda item: abs(item.weight - 1.0), reverse=True)[:20]
    evidence = {
        "context_signals": [asdict(signal) for signal in context_signals],
        "top_rule_impacts": [asdict(impact) for impact in top_impacts],
        "raw_top_labels": raw_top,
        "adjusted_top_labels": adjusted_top,
        "adjusted_non_silence_labels": adjusted_non_silence_top,
        "scene_confidence": confidence,
    }
    return SemanticSummary(
        scene_tags=scene_tags,
        dominant_labels=adjusted_top,
        dominant_non_silence_labels=adjusted_non_silence_top,
        salient_segments=[asdict(segment) for segment in segments[:12]],
        semantic_scene=semantic_scene,
        narrative=narrative,
        evidence=evidence,
    )


def write_dual_jsonl(
    out_path: str,
    t_sec: np.ndarray,
    raw_scores: np.ndarray,
    adjusted_scores: np.ndarray,
    labels: Sequence[str],
    top_k: int,
) -> None:
    with open(out_path, "w") as f:
        for t, raw_row, adjusted_row in zip(t_sec, raw_scores, adjusted_scores):
            f.write(
                json.dumps(
                    {
                        "t_sec": float(t),
                        "raw_top": top_k_labels(raw_row, labels, top_k),
                        "adjusted_top": top_k_labels(adjusted_row, labels, top_k),
                    }
                )
                + "\n"
            )


def write_npz(
    out_path: str,
    t_sec: np.ndarray,
    raw_scores: np.ndarray,
    adjusted_scores: np.ndarray,
    labels: Sequence[str],
    adjustment_vector: np.ndarray,
    context_path: Optional[str],
    args: argparse.Namespace,
) -> None:
    np.savez_compressed(
        out_path,
        t_sec=t_sec.astype(np.float32),
        raw_scores=raw_scores.astype(np.float32),
        adjusted_scores=adjusted_scores.astype(np.float32),
        labels=np.array(labels, dtype=object),
        adjustment_vector=adjustment_vector.astype(np.float32),
        sample_hz=float(args.sample_hz),
        window_s=float(args.window_s),
        avg_frames=int(args.avg_frames),
        audio=str(args.audio),
        context_path=str(context_path or ""),
        model=str(args.model),
    )


def write_summary_json(
    out_path: str,
    summary: SemanticSummary,
    context: Dict[str, Any],
    adjustment_vector: np.ndarray,
    labels: Sequence[str],
    audio_stats: Dict[str, Any],
) -> None:
    top_adjustments = top_k_labels(adjustment_vector, labels, 15)
    payload = {
        "semantic_summary": asdict(summary),
        "context": context,
        "top_adjustment_weights": top_adjustments,
        "audio_stats": audio_stats,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def compute_audio_stats(waveform: np.ndarray, sr: int) -> Dict[str, Any]:
    mono = to_mono(np.asarray(waveform, dtype=np.float32)).astype(np.float32)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    duration_sec = float(len(mono) / sr) if sr > 0 else 0.0
    silent_fraction = float(np.mean(np.abs(mono) < 1e-3)) if mono.size else 1.0
    return {
        "sample_rate_hz": int(sr),
        "num_samples": int(len(mono)),
        "duration_sec": duration_sec,
        "peak_abs": peak,
        "rms": rms,
        "near_silent_fraction": silent_fraction,
    }


def default_output_paths(audio_path: str, output_dir: Optional[str]) -> Dict[str, str]:
    audio_stem = Path(audio_path).stem
    base_dir = Path(output_dir) if output_dir else Path(audio_path).resolve().parent
    base_dir.mkdir(parents=True, exist_ok=True)
    return {
        "jsonl": str(base_dir / f"{audio_stem}_semantic_predictions.jsonl"),
        "npz": str(base_dir / f"{audio_stem}_semantic_predictions.npz"),
        "summary": str(base_dir / f"{audio_stem}_semantic_summary.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Context-aware semantic audio pipeline. "
            "Runs YAMNet inference, applies context weighting, and emits scene-level summaries."
        )
    )
    parser.add_argument("--audio", required=True, help="Path to an audio file.")
    parser.add_argument("--context_json", help="Optional JSON file with location/weather/scene metadata.")
    parser.add_argument("--context_rules", default=str(DEFAULT_CONTEXT_RULES_PATH), help="JSON file containing context adjustment rules.")
    parser.add_argument("--model", default="yamnet.tflite")
    parser.add_argument("--labels", default="yamnet_class_map.csv")
    parser.add_argument("--sample_hz", type=float, default=1.0)
    parser.add_argument("--window_s", type=float, default=0.975)
    parser.add_argument("--avg_frames", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--auto_window", action="store_true")
    parser.add_argument("--out_jsonl", help="Output JSONL path for raw + adjusted top-k predictions.")
    parser.add_argument("--out_npz", help="Output NPZ path for raw and adjusted score matrices.")
    parser.add_argument("--out_summary", help="Output JSON path for scene-level semantic summary.")
    parser.add_argument("--output_dir", help="Optional output directory when explicit file paths are omitted.")
    args = parser.parse_args()

    if args.auto_window:
        expected_len = get_model_expected_len(args.model)
        args.window_s = expected_len / 16000.0

    outputs = default_output_paths(args.audio, args.output_dir)
    out_jsonl = args.out_jsonl or outputs["jsonl"]
    out_npz = args.out_npz or outputs["npz"]
    out_summary = args.out_summary or outputs["summary"]

    labels = load_labels(args.labels)
    rules_config = load_context_rules_config(args.context_rules)
    context = load_context(args.context_json)
    waveform, sr = load_audio(args.audio)
    waveform = to_mono(np.asarray(waveform)).astype(np.float32)
    audio_stats = compute_audio_stats(waveform, sr)

    interp = create_interpreter(args.model)
    interp.allocate_tensors()
    t_sec, raw_scores = classify_waveform(interp, waveform, sr, args.sample_hz, args.window_s, args.avg_frames)

    if raw_scores.shape[1] != len(labels):
        raise ValueError(f"Model outputs {raw_scores.shape[1]} classes but labels contain {len(labels)} entries.")

    adjustment_vector, impacts, context_signals = build_context_adjustment(labels, context, rules_config=rules_config)
    adjusted_scores = np.clip(raw_scores * adjustment_vector[None, :], 0.0, 1.0)

    summary = build_semantic_summary(
        t_sec=t_sec,
        raw_scores=raw_scores,
        adjusted_scores=adjusted_scores,
        labels=labels,
        top_k=args.top_k,
        context=context,
        impacts=impacts,
        context_signals=context_signals,
        audio_stats=audio_stats,
    )

    write_dual_jsonl(out_jsonl, t_sec, raw_scores, adjusted_scores, labels, args.top_k)
    write_npz(out_npz, t_sec, raw_scores, adjusted_scores, labels, adjustment_vector, args.context_json, args)
    write_summary_json(out_summary, summary, context, adjustment_vector, labels, audio_stats)

    print(f"[ok] wrote semantic predictions -> {out_jsonl}")
    print(f"[ok] wrote semantic matrix -> {out_npz}")
    print(f"[ok] wrote semantic summary -> {out_summary}")
    print(
        "[info] audio stats -> "
        f"duration={audio_stats['duration_sec']:.2f}s, "
        f"peak={audio_stats['peak_abs']:.4f}, "
        f"rms={audio_stats['rms']:.4f}, "
        f"near_silent_fraction={audio_stats['near_silent_fraction']:.3f}"
    )


if __name__ == "__main__":
    main()
