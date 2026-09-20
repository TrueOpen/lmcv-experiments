from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .metrics import safe_float


def normalize_logprob_value(value: Any) -> Dict[str, Any]:
    if hasattr(value, "logprob"):
        return {
            "logprob": float(value.logprob),
            "rank": getattr(value, "rank", None),
            "decoded_token": getattr(value, "decoded_token", None),
        }
    if isinstance(value, dict):
        logprob = value.get("logprob", value.get("value"))
        return {
            "logprob": float(logprob),
            "rank": value.get("rank"),
            "decoded_token": value.get("decoded_token"),
        }
    return {"logprob": float(value), "rank": None, "decoded_token": None}


def normalize_logprob_dict(raw: Any) -> Dict[str, Dict[str, Any]]:
    if raw is None:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for token_id, value in dict(raw).items():
        out[str(int(token_id))] = normalize_logprob_value(value)
    return out


def sorted_top_logprobs(logprobs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(item: tuple) -> tuple:
        token_id, token_info = item
        rank = token_info.get("rank")
        if rank is None:
            rank_key = float("inf")
        else:
            rank_key = safe_float(rank, float("inf"))
        return rank_key, -safe_float(token_info.get("logprob")), int(token_id)

    rows = []
    for position, (token_id, token_info) in enumerate(
        sorted(logprobs.items(), key=sort_key),
        start=1,
    ):
        rows.append({
            "position": position,
            "token_id": int(token_id),
            "logprob": safe_float(token_info.get("logprob")),
            "rank": token_info.get("rank"),
            "decoded_token": token_info.get("decoded_token"),
        })
    return rows


def logprob_for_token(logprobs: Dict[str, Dict[str, Any]], token_id: int) -> Optional[float]:
    item = logprobs.get(str(int(token_id)))
    if item is None:
        return None
    return safe_float(item.get("logprob"))


def rank_for_token(logprobs: Dict[str, Dict[str, Any]], token_id: int) -> Optional[int]:
    item = logprobs.get(str(int(token_id)))
    if item is None:
        return None
    rank = item.get("rank")
    if rank is None:
        return None
    return int(rank)


def make_tokens_prompt(token_ids: Sequence[int]) -> Any:
    ids = [int(token_id) for token_id in token_ids]
    try:
        from vllm.inputs import TokensPrompt

        return TokensPrompt(prompt_token_ids=ids)
    except Exception:
        return {"prompt_token_ids": ids}


def build_llm_kwargs(args: Any) -> Dict[str, Any]:
    kwargs = {
        "model": args.model,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": getattr(args, "max_num_seqs", None),
        "max_num_batched_tokens": getattr(args, "max_num_batched_tokens", None),
        "max_logprobs": args.max_logprobs,
        "quantization": None if args.quantization == "bf16" else args.quantization,
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def build_sampling_params(args: Any,
                          *,
                          max_tokens: int,
                          logprobs: Optional[int] = None,
                          prompt_logprobs: Optional[int] = None,
                          detokenize: bool = False) -> Any:
    from vllm import SamplingParams

    kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k if args.top_k > 0 else -1,
        "max_tokens": max_tokens,
        "logprobs": logprobs,
        "prompt_logprobs": prompt_logprobs,
        "seed": args.seed,
        "detokenize": detokenize,
    }
    supported = set()
    try:
        supported = set(inspect.signature(SamplingParams).parameters)
    except (TypeError, ValueError):
        pass
    if supported:
        kwargs = {key: value for key, value in kwargs.items() if key in supported}
    return SamplingParams(**kwargs)


def write_jsonl_row(handle: Any, payload: Dict[str, Any], flush: bool = True) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    handle.write("\n")
    if flush:
        handle.flush()
