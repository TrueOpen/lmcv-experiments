from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sorted_token_ids(token_ids: Iterable[str]) -> List[str]:
    return sorted((str(token_id) for token_id in token_ids), key=lambda item: int(item))


def normalize_logprobs(logprobs: Sequence[float]) -> List[float]:
    if not logprobs:
        return []
    max_logprob = max(logprobs)
    if not math.isfinite(max_logprob):
        return [float("nan") for _ in logprobs]
    weights = [math.exp(value - max_logprob) for value in logprobs]
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        return [float("nan") for _ in logprobs]
    return [value / total for value in weights]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return float("nan")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan")
    return dot / (left_norm * right_norm)


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    if not p or not q or len(p) != len(q):
        return float("nan")
    total = 0.0
    for p_value, q_value in zip(p, q):
        if p_value <= 0.0:
            continue
        if q_value <= 0.0:
            return float("inf")
        total += p_value * math.log(p_value / q_value)
    return total


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    if not p or not q or len(p) != len(q):
        return float("nan")
    middle = [(p_value + q_value) / 2.0 for p_value, q_value in zip(p, q)]
    return 0.5 * kl_divergence(p, middle) + 0.5 * kl_divergence(q, middle)


def logprob_value(logprobs: Mapping[str, Mapping[str, object]],
                  token_id: str,
                  missing_logprob: float) -> float:
    if token_id not in logprobs:
        return missing_logprob
    return safe_float(logprobs[token_id].get("logprob"))


def raw_logit_value(logprobs: Mapping[str, Mapping[str, object]],
                    token_id: str,
                    missing_logit: float = float("nan")) -> float:
    if token_id not in logprobs:
        return missing_logit
    return safe_float(logprobs[token_id].get("raw_logit"))


def probability_metrics(left: Mapping[str, Mapping[str, object]],
                        right: Mapping[str, Mapping[str, object]],
                        token_ids: Sequence[str],
                        missing_logprob: float) -> Dict[str, float]:
    left_logprobs = [
        logprob_value(left, token_id, missing_logprob)
        for token_id in token_ids
    ]
    right_logprobs = [
        logprob_value(right, token_id, missing_logprob)
        for token_id in token_ids
    ]
    left_probs = normalize_logprobs(left_logprobs)
    right_probs = normalize_logprobs(right_logprobs)
    return {
        "prob_cosine": cosine(left_probs, right_probs),
        "kl_left_to_right": kl_divergence(left_probs, right_probs),
        "kl_right_to_left": kl_divergence(right_probs, left_probs),
        "js_divergence": js_divergence(left_probs, right_probs),
    }


def quantile(values: Sequence[float], q: float) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return clean[lower]
    frac = position - lower
    return clean[lower] * (1.0 - frac) + clean[upper] * frac


def depth_bucket(depth: int) -> str:
    depth = int(depth)
    if depth <= 16:
        return "0000-0016"
    if depth <= 64:
        return "0017-0064"
    if depth <= 256:
        return "0065-0256"
    if depth <= 1024:
        return "0257-1024"
    return "1025+"


def margin_bucket(margin: float) -> str:
    if not math.isfinite(margin):
        return "unknown"
    if margin < 0.05:
        return "tie_lt_0.05"
    if margin < 0.25:
        return "low_0.05_0.25"
    if margin < 1.0:
        return "mid_0.25_1"
    return "high_ge_1"
