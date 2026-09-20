#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def now_ms() -> int:
    return int(time.time() * 1000)


def normalize_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def completions_url(base_url: str) -> str:
    return f"{normalize_base_url(base_url)}/v1/completions"


def tokenize_url(base_url: str) -> str:
    return f"{normalize_base_url(base_url)}/tokenize"


def api_key_from_args(api_key: Optional[str]) -> Optional[str]:
    return api_key or os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY")


def post_json(url: str,
              payload: Dict[str, Any],
              *,
              api_key: Optional[str] = None,
              timeout: float = 600.0) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc
    return json.loads(body)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
            if isinstance(payload, str):
                yield {"sample_id": f"row_{line_number:06d}", "prompt": payload}
            elif isinstance(payload, dict):
                yield payload
            else:
                raise ValueError(f"{path}:{line_number}: row must be a JSON object or string")


def write_jsonl_row(handle: Any, payload: Dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    handle.write("\n")
    handle.flush()


def prompt_from_row(row: Dict[str, Any], row_index: int) -> Tuple[str, Any, Optional[List[int]]]:
    sample_id = str(row.get("sample_id") or row.get("id") or f"sample_{row_index:06d}")
    token_ids = row.get("prompt_token_ids") or row.get("input_token_ids") or row.get("input_ids")
    if token_ids is not None:
        return sample_id, [int(token_id) for token_id in token_ids], [int(token_id) for token_id in token_ids]
    for key in ("prompt", "input", "input_text", "text"):
        if key in row:
            return sample_id, str(row[key]), None
    raise ValueError(f"Row {sample_id} has no prompt/input/input_text/text or token ids")


def extract_choice(response: Dict[str, Any]) -> Dict[str, Any]:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"Response has no choices: {response}")
    return choices[0]


def first_int_list(value: Any) -> Optional[List[int]]:
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return [int(item) for item in value]
    return None


def extract_token_ids(payload: Dict[str, Any]) -> Optional[List[int]]:
    for key in (
        "token_ids",
        "prompt_token_ids",
        "output_token_ids",
        "input_ids",
        "tokens",
    ):
        token_ids = first_int_list(payload.get(key))
        if token_ids is not None:
            return token_ids
    logprobs = payload.get("logprobs")
    if isinstance(logprobs, dict):
        token_ids = first_int_list(logprobs.get("token_ids"))
        if token_ids is not None:
            return token_ids
    return None


def extract_prompt_token_ids(response: Dict[str, Any], choice: Dict[str, Any]) -> Optional[List[int]]:
    for source in (choice, response):
        token_ids = source.get("prompt_token_ids")
        if first_int_list(token_ids) is not None:
            return [int(token_id) for token_id in token_ids]
    return None


def tokenize_prompt(base_url: str,
                    model: str,
                    prompt: Any,
                    *,
                    api_key: Optional[str],
                    timeout: float) -> List[int]:
    if isinstance(prompt, list):
        return [int(token_id) for token_id in prompt]
    response = post_json(
        tokenize_url(base_url),
        {"model": model, "prompt": prompt},
        api_key=api_key,
        timeout=timeout,
    )
    token_ids = extract_token_ids(response)
    if token_ids is None:
        raise RuntimeError(f"Could not find token ids in /tokenize response: {response}")
    return token_ids


def route_field_from_choice(choice: Dict[str, Any]) -> Any:
    for key in ("routed_experts", "prompt_routed_experts"):
        if key in choice and choice[key] is not None:
            return choice[key]
    return None


def decode_routed_experts(value: Any) -> Optional[np.ndarray]:
    np_module = require_numpy()
    if value is None:
        return None
    if isinstance(value, str):
        raw = base64.b64decode(value)
        return np_module.load(io.BytesIO(raw), allow_pickle=False)
    if isinstance(value, list):
        return np_module.asarray(value)
    if isinstance(value, dict):
        for key in ("data", "array", "routed_experts", "prompt_routed_experts"):
            if key in value:
                return decode_routed_experts(value[key])
    raise TypeError(f"Unsupported routed_experts payload type: {type(value).__name__}")


def route_summary(value: Any) -> Dict[str, Any]:
    if value is None:
        return {"present": False}
    try:
        arr = decode_routed_experts(value)
    except RuntimeError as exc:
        return {"present": True, "decoded": False, "error": str(exc)}
    if arr is None:
        return {"present": False}
    return {
        "present": True,
        "shape": [int(dim) for dim in arr.shape],
        "dtype": str(arr.dtype),
        "num_entries": int(arr.size),
        "min": int(arr.min()) if arr.size else None,
        "max": int(arr.max()) if arr.size else None,
    }


def normalize_route_array(arr: np.ndarray) -> np.ndarray:
    np_module = require_numpy()
    arr = np_module.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape((arr.shape[0], 1, 1))
    elif arr.ndim == 2:
        arr = arr.reshape((arr.shape[0], arr.shape[1], 1))
    elif arr.ndim != 3:
        raise ValueError(f"Expected routed_experts rank 1/2/3, got shape {arr.shape}")
    return arr.astype(np_module.int64, copy=False)


def require_numpy() -> Any:
    if np is None:
        raise RuntimeError(
            "numpy is required to decode and compare routed_experts. "
            "Install it with: python3 -m pip install -r requirements.txt"
        )
    return np


def print_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
