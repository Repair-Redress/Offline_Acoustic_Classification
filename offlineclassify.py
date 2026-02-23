import argparse
import csv
import json
import math

import numpy as np
import soundfile as sf
import scipy.signal
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

def get_model_expected_len(model_path: str) -> int:
    """Return expected waveform length from the model input tensor."""
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    # expects shape [15600] or [1,15600]
    return int(input_details[0]["shape"][-1])

def load_labels(labels_path: str):
    """
    Supports:
      - CSV like yamnet_class_map.csv with columns: index, mid, display_name
      - JSON metadata.json that contains a list/dict of labels somewhere
    """
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

        # Common patterns seen in metadata json files
        candidates = [
            data.get("labels"),
            data.get("class_labels"),
            data.get("class_map"),
            data.get("classes"),
            (data.get("yamnet") or {}).get("labels"),
            (data.get("yamnet") or {}).get("class_map"),
        ]

        for c in candidates:
            if not c:
                continue

            # Case A: list of strings
            if isinstance(c, list) and all(isinstance(x, str) for x in c):
                return c

            # Case B: list of dicts
            if isinstance(c, list) and all(isinstance(x, dict) for x in c):
                out = []
                for x in c:
                    out.append(x.get("display_name") or x.get("name") or x.get("label"))
                if all(out):
                    return out

            # Case C: dict mapping index->label OR name->index
            if isinstance(c, dict):
                # try index->label mapping
                try:
                    keys_are_intlike = all(str(k).isdigit() for k in c.keys())
                    if keys_are_intlike:
                        return [c[str(i)] if str(i) in c else c[i] for i in range(len(c))]
                except Exception:
                    pass

        raise ValueError(
            f"Couldn't find labels inside {labels_path}. "
            "If this metadata.json doesn't include the 521 labels, add a yamnet_class_map.csv and pass --labels to it."
        )

    raise ValueError("Labels file must be .csv or .json")

def to_mono(x: np.ndarray):
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)

def resample(x: np.ndarray, sr_in: int, sr_out: int = 16000):
    if sr_in == sr_out:
        return x
    # high quality polyphase resampling
    g = math.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return scipy.signal.resample_poly(x, up, down).astype(np.float32)

def frame_waveform(x, sr, window_s, hop_s):
    win = int(round(window_s * sr))
    hop = int(round(hop_s * sr))
    if win <= 0 or hop <= 0:
        raise ValueError("window_s and hop_s must be > 0")

    frames = []
    centers = []
    n = len(x)
    i = 0
    while i < n:
        chunk = x[i:i+win]
        if len(chunk) < win:
            # pad tail with zeros so last frame still gets a prediction
            chunk = np.pad(chunk, (0, win - len(chunk)))
        frames.append(chunk)
        centers.append((i + win/2) / sr)
        i += hop
    return np.stack(frames), np.array(centers)

def run_tflite(interp, frames: np.ndarray):
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    # Many YAMNet TFLite variants take [win] float32 waveform.
    # Some expect [1, win]. We'll handle both.
    scores = []
    for f in frames:
        x = f.astype(np.float32)
        if input_details[0]["shape"].ndim == 2:
            x = x[None, :]
        interp.set_tensor(input_details[0]["index"], x)
        interp.invoke()
        y = interp.get_tensor(output_details[0]["index"])
        y = np.squeeze(y)  # [521]
        scores.append(y)
    return np.stack(scores)  # [T, 521]

def temporal_average(scores: np.ndarray, avg_frames: int):
    if avg_frames <= 1:
        return scores
    # simple trailing window mean
    out = np.zeros_like(scores)
    csum = np.cumsum(scores, axis=0)
    for i in range(scores.shape[0]):
        j0 = max(0, i - avg_frames + 1)
        window_sum = csum[i] - (csum[j0 - 1] if j0 > 0 else 0)
        out[i] = window_sum / (i - j0 + 1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--model", default="yamnet.tflite")
    ap.add_argument("--labels", default="yamnet_class_map.csv")
    ap.add_argument("--sample_hz", type=float, default=1.0, help="predictions per second (hop = 1/sample_hz)")
    ap.add_argument("--window_s", type=float, default=0.975)
    ap.add_argument("--avg_frames", type=int, default=1)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--out_json", default="predictions.jsonl")
    ap.add_argument("--out_npz", default="predictions_full.npz", help="Optional: write full t_sec + 521-dim scores to .npz")
    ap.add_argument("--auto_window", action="store_true", help="Override window_s to match model input length")
    args = ap.parse_args()

    # Ensure window matches model input length if requested
    if args.auto_window:
        expected_len = get_model_expected_len(args.model)
        args.window_s = expected_len / 16000.0

    x, sr = sf.read(args.audio, always_2d=False)
    x = to_mono(np.asarray(x))
    x = x.astype(np.float32)

    # Normalize if audio was int-based
    if np.max(np.abs(x)) > 1.5:
        x = x / 32768.0

    x = resample(x, sr, 16000)

    hop_s = 1.0 / args.sample_hz
    frames, t_centers = frame_waveform(x, 16000, args.window_s, hop_s)

    interp = Interpreter(model_path=args.model)
    interp.allocate_tensors()

    raw_scores = run_tflite(interp, frames)
    scores = temporal_average(raw_scores, args.avg_frames)

    if args.out_npz:
        np.savez_compressed(
            args.out_npz,
            t_sec=t_centers.astype(np.float32),
            scores=scores.astype(np.float32),
            labels=np.array(labels, dtype=object),
            sample_hz=float(args.sample_hz),
            window_s=float(args.window_s),
            avg_frames=int(args.avg_frames),
            model=str(args.model),
        )
        print(f"[ok] wrote full scores npz -> {args.out_npz}")

    labels = load_labels(args.labels)
    if scores.shape[1] != len(labels):
        raise ValueError(
            f"Mismatch: model outputs {scores.shape[1]} classes, but label map has {len(labels)} labels.\n"
            f"Your categoriesFile must be the YAMNet 521-class map in the SAME index order."
        )

    with open(args.out_json, "w") as f:
        for t, s in zip(t_centers, scores):
            top_idx = np.argsort(-s)[:args.top_k]
            f.write(json.dumps({
                "t_sec": float(t),
                "top": [{"label": labels[i], "score": float(s[i])} for i in top_idx]
            }) + "\n")

if __name__ == "__main__":
    main()
