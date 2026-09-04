from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections import Counter
from pathlib import Path

import numpy as np

OFFICIAL_ANOMALY_KEYWORDS = [
    "trolley", "cart", "luggage", "bicycle", "scooter", "vehicles", "vans",
    "accident", "running", "jumping", "riding", "skateboarding", "scooting",
    "lying", "falling", "bending", "fighting", "pushing", "loitering",
    "climbing",
]


def natural_key(path: str | Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", Path(path).name)]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def append_jsonl(path: str | Path, value) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def atomic_json(path: str | Path, value) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def ewm_adjust(values, *, alpha: float | None = None, span: float | None = None):
    """Equivalent to pandas Series.ewm(..., adjust=True).mean()."""
    x = np.asarray(values, dtype=np.float64)
    if span is not None:
        alpha = 2.0 / (span + 1.0)
    assert alpha is not None and 0 < alpha <= 1
    out = np.empty_like(x)
    numerator = 0.0; denominator = 0.0
    decay = 1.0 - alpha
    for i, value in enumerate(x):
        numerator = value + decay * numerator
        denominator = 1.0 + decay * denominator
        out[i] = numerator / denominator
    return out


def ema_majority_smooth(ema_data, threshold, window_size=20, edge_region_size=None):
    # Faithful port of official majority_smooth.py.
    if window_size % 2 == 0:
        window_size += 1
    pad_size = window_size // 2
    if edge_region_size is None:
        edge_region_size = pad_size
    padded = np.pad(np.asarray(ema_data), pad_size, mode="edge")
    out = np.zeros(len(ema_data), dtype=np.int8)
    for i in range(len(ema_data)):
        if i < edge_region_size or i >= len(ema_data) - edge_region_size:
            start, end = max(0, i - pad_size), i + 1
        else:
            start, end = i, i + window_size
        window = padded[start:end]
        above = int(np.sum(window > threshold))
        out[i] = 1 if above > len(window) - above else 0
    return out


def cluster_keyword(lines, keywords=OFFICIAL_ANOMALY_KEYWORDS):
    preds, found = [], []
    for line in lines:
        hits = [word for word in keywords if word in line]
        found.extend(hits); preds.append(1 if hits else 0)
    return np.asarray(preds, dtype=np.int8), found


def _most_frequent_keyword(text, keywords):
    counts = Counter(word for word in re.findall(r"\b\w+\b", text) if word in keywords)
    return counts.most_common(1)[0][0] if counts else None


def _remove_sentences_with_keywords(text, keywords):
    parts = re.split(r",|\.", text)
    return ". ".join(s for s in parts if not any(k in s for k in keywords))


def official_perception_smoothing(descriptions, keywords=OFFICIAL_ANOMALY_KEYWORDS):
    preds, _ = cluster_keyword(descriptions, keywords)
    ema = ewm_adjust(preds, alpha=0.33)
    threshold = float(ema.mean())
    smooth = ema_majority_smooth(ema, threshold, window_size=1)
    modified = []
    for i, (pred, spred, text) in enumerate(zip(preds, smooth, descriptions)):
        if spred == 0 and pred == 1:
            text = _remove_sentences_with_keywords(text, keywords)
        elif spred == 1:
            lo, hi = max(0, i - 1), min(len(descriptions), i + 2)
            keyword = _most_frequent_keyword(" ".join(descriptions[lo:hi]), keywords)
            if keyword:
                addition = keyword if keyword.endswith("ing") else f"riding a {keyword}"
                text = re.sub(r"(the first person is)[^,]*", r"\1 " + addition, text)
        modified.append(text)
    # Official code protects only the second EMA alpha after smoothing.
    score_alpha = threshold if threshold != 0 else threshold + 1e-7
    return {
        "raw_keyword_predictions": preds,
        "ema_keyword_scores": ema,
        "smoothed_keyword_predictions": smooth,
        "smoothed_keyword_scores": ewm_adjust(smooth, alpha=score_alpha),
        "modified_descriptions": modified,
        "threshold": threshold,
    }


def official_final_scores(predictions):
    return ewm_adjust(predictions, span=10)


def parse_final_decision(text: str) -> int:
    # Backend-only replacement for the original BART MNLI post_process helper.
    # Qwen often uses a markdown heading followed by a bare bold conclusion,
    # e.g. ``Final Answer:\n**anomaly**``.  Prefer the final-answer suffix,
    # then fall back to the last explicit normal/anomaly token in the output.
    lowered=text.lower()
    marks=list(re.finditer(r"(?:final\s+(?:answer|determination)|overall\s+(?:answer|classification|determination))", lowered))
    scope=lowered[marks[-1].end():] if marks else lowered
    words=list(re.finditer(r"\b(normal|anomaly|anomalous)\b", scope))
    if not words and scope is not lowered:
        words=list(re.finditer(r"\b(normal|anomaly|anomalous)\b", lowered))
    if not words:
        raise ValueError("No final normal/anomaly decision in response")
    return int(words[-1].group(1) != "normal")


def pad_last(values, length):
    arr = np.asarray(values)
    if len(arr) == 0:
        raise ValueError("cannot align empty prediction")
    if len(arr) > length:
        raise ValueError(f"prediction longer than target: {len(arr)} > {length}")
    return np.pad(arr, (0, length-len(arr)), mode="edge")
