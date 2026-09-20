#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from worker_logits_verify.schema import (  # noqa: E402
    DecodeConfig,
    VariantInfo,
    dataclass_to_dict,
    parse_worker_sample,
    read_jsonl,
    runtime_metadata,
    write_json,
)
from worker_logits_verify.vllm_utils import sorted_top_logprobs, write_jsonl_row  # noqa: E402


JSONDict = Dict[str, Any]


def log(message: str) -> None:
    print(message, flush=True)


def load_extra_body(raw: str) -> JSONDict:
    if not raw:
        return {}
    path = Path(raw)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(raw)


def compact_dict(payload: JSONDict) -> JSONDict:
    return {key: value for key, value in payload.items() if value is not None}


def endpoint_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


class VLLMOpenAIClient:
    def __init__(self,
                 base_url: str,
                 api_key: str,
                 timeout: float,
                 retries: int,
                 retry_sleep: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.retry_sleep = retry_sleep

    def post_json(self, path: str, payload: JSONDict) -> JSONDict:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint_url(self.base_url, path),
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code} from {path}: {body}")
            except urllib.error.URLError as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(self.retry_sleep)
        assert last_error is not None
        raise last_error

    def tokenize(self, model: str, text: str) -> List[int]:
        response = self.post_json("/tokenize", {
            "model": model,
            "prompt": text,
            "add_special_tokens": False,
        })
        token_ids = (
            response.get("token_ids")
            or response.get("tokens")
            or response.get("input_ids")
        )
        if not isinstance(token_ids, list):
            raise ValueError(f"/tokenize response did not include token ids: {response}")
        return [int(token_id) for token_id in token_ids]


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_token_id(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.startswith("token_id:"):
        text = text.split(":", 1)[1].strip()
    if text.startswith("<token_id:") and text.endswith(">"):
        text = text[len("<token_id:"):-1].strip()
    if text.isdigit():
        return int(text)
    return None


def decoded_token_from_key(value: Any, token_id: Optional[int]) -> str:
    if token_id is None and isinstance(value, str):
        return value
    if isinstance(value, str) and not value.strip().startswith("token_id:"):
        return value
    return ""


def normalize_logprob_mapping(raw: Any,
                              *,
                              strict_token_ids: bool,
                              context: str) -> Dict[str, Dict[str, Any]]:
    if raw is None:
        return {}

    rows: List[Tuple[int, Dict[str, Any]]] = []
    bad_keys: List[str] = []

    def add_item(key: Any, value: Any, position: int) -> None:
        token_id = parse_token_id(key)
        decoded_token = decoded_token_from_key(key, token_id)
        rank: Optional[int] = None
        logprob: Optional[float] = None

        if isinstance(value, dict):
            explicit_token_id = value.get("token_id")
            if explicit_token_id is not None:
                token_id = parse_token_id(explicit_token_id)
            elif token_id is None:
                token_id = parse_token_id(value.get("token"))
            decoded_token = str(
                value.get("decoded_token")
                or value.get("token")
                or decoded_token
                or ""
            )
            rank_value = value.get("rank")
            rank = int(rank_value) if rank_value is not None else None
            logprob = safe_float(value.get("logprob", value.get("value")))
        else:
            logprob = safe_float(value)

        if token_id is None:
            bad_keys.append(str(key))
            return
        rows.append((
            int(token_id),
            {
                "logprob": logprob,
                "rank": rank if rank is not None else position,
                "decoded_token": decoded_token,
            },
        ))

    if isinstance(raw, dict):
        for position, (key, value) in enumerate(raw.items(), start=1):
            add_item(key, value, position)
    elif isinstance(raw, list):
        for position, item in enumerate(raw, start=1):
            if isinstance(item, dict):
                key = item.get("token_id", item.get("token"))
                add_item(key, item, position)
            else:
                add_item(position, item, position)
    else:
        raise TypeError(f"{context}: unsupported logprob container {type(raw).__name__}")

    if bad_keys and strict_token_ids:
        preview = ", ".join(repr(key) for key in bad_keys[:5])
        raise ValueError(
            f"{context}: top logprob keys do not expose token ids ({preview}). "
            "Start vLLM OpenAI server with token-id support and keep "
            "`return_token_ids` / `return_tokens_as_token_ids` enabled, "
            "or rerun with --allow-text-top-logprobs to ignore unmapped top-k entries."
        )

    return {str(token_id): info for token_id, info in rows}


def extract_completion_token_ids(choice: JSONDict) -> List[int]:
    logprobs = choice.get("logprobs") or {}
    candidates = [
        choice.get("token_ids"),
        choice.get("output_token_ids"),
        logprobs.get("token_ids") if isinstance(logprobs, dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            return [int(token_id) for token_id in candidate]

    tokens = logprobs.get("tokens") if isinstance(logprobs, dict) else None
    if isinstance(tokens, list):
        parsed = [parse_token_id(token) for token in tokens]
        if all(token_id is not None for token_id in parsed):
            return [int(token_id) for token_id in parsed if token_id is not None]
    return []


def _nested_prompt_ids(value: Any, choice_index: int) -> Optional[List[int]]:
    if not isinstance(value, list):
        return None
    if not value:
        return []
    if all(isinstance(item, int) for item in value):
        return [int(item) for item in value]
    if choice_index < len(value) and isinstance(value[choice_index], list):
        return [int(token_id) for token_id in value[choice_index]]
    return None


def extract_prompt_token_ids(response: JSONDict,
                             choice: JSONDict,
                             choice_index: int) -> List[int]:
    logprobs = choice.get("logprobs") or {}
    candidates = [
        choice.get("prompt_token_ids"),
        logprobs.get("prompt_token_ids") if isinstance(logprobs, dict) else None,
        response.get("prompt_token_ids"),
    ]
    for candidate in candidates:
        token_ids = _nested_prompt_ids(candidate, choice_index)
        if token_ids is not None:
            return token_ids
    return []


def selected_rank(top_logprobs: Sequence[JSONDict], token_id: int) -> Optional[int]:
    for item in top_logprobs:
        if int(item["token_id"]) == int(token_id):
            rank = item.get("rank", item.get("position"))
            return int(rank) if rank is not None else None
    return None


def trace_from_completion_choice(choice: JSONDict,
                                 output_ids: Sequence[int],
                                 *,
                                 strict_token_ids: bool) -> List[JSONDict]:
    logprobs = choice.get("logprobs") or {}
    tokens = logprobs.get("tokens", []) if isinstance(logprobs, dict) else []
    token_logprobs = logprobs.get("token_logprobs", []) if isinstance(logprobs, dict) else []
    top_logprobs = logprobs.get("top_logprobs", []) if isinstance(logprobs, dict) else []
    rows: List[JSONDict] = []

    for depth, token_id in enumerate(output_ids):
        selected_logprob = (
            safe_float(token_logprobs[depth])
            if depth < len(token_logprobs) else None
        )
        target_token = tokens[depth] if depth < len(tokens) else ""
        raw_top = top_logprobs[depth] if depth < len(top_logprobs) else {}
        top_map = normalize_logprob_mapping(
            raw_top,
            strict_token_ids=strict_token_ids,
            context=f"completion depth={depth}",
        )
        top_rows = sorted_top_logprobs(top_map)
        rows.append({
            "depth": depth,
            "target_token_id": int(token_id),
            "target_decoded_token": decoded_token_from_key(target_token, parse_token_id(target_token)),
            "selected_logprob": selected_logprob,
            "selected_rank": selected_rank(top_rows, int(token_id)),
            "top_logprobs": top_rows,
        })
    return rows


def make_completion_body(args: argparse.Namespace,
                         prompts: Sequence[Any],
                         *,
                         prompt_logprobs: Optional[int] = None) -> JSONDict:
    body: JSONDict = {
        "model": args.model,
        "prompt": list(prompts),
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "logprobs": args.logprobs if prompt_logprobs is None else None,
        "prompt_logprobs": prompt_logprobs,
        "echo": False,
        "seed": args.seed,
        "stream": False,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
        "skip_special_tokens": False,
    }
    if args.top_k > 0:
        body["top_k"] = args.top_k
    body.update(load_extra_body(args.extra_body_json))
    return compact_dict(body)


def choices_by_index(response: JSONDict, expected_count: int) -> List[JSONDict]:
    choices = response.get("choices")
    if not isinstance(choices, list):
        raise ValueError(f"Completion response missing choices: {response}")
    ordered: List[Optional[JSONDict]] = [None] * expected_count
    extras: List[JSONDict] = []
    for position, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        index = choice.get("index", position)
        if isinstance(index, int) and 0 <= index < expected_count and ordered[index] is None:
            ordered[index] = choice
        else:
            extras.append(choice)
    for index in range(expected_count):
        if ordered[index] is None and extras:
            ordered[index] = extras.pop(0)
    if any(choice is None for choice in ordered):
        raise RuntimeError(f"vLLM returned {len(choices)} choices for {expected_count} prompts")
    return [choice for choice in ordered if choice is not None]


def make_variant(args: argparse.Namespace) -> VariantInfo:
    return VariantInfo(
        variant_id=args.variant_id,
        model=args.model,
        tokenizer_hash=args.tokenizer_hash,
        tokenizer_name=args.tokenizer_name or args.model,
        quantization=args.quantization,
        gpu=args.gpu,
        backend="vllm-openai-api",
        backend_version=args.backend_version,
        dtype=args.dtype,
        extra={
            "base_url": args.base_url,
            "logprobs": args.logprobs,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run worker inference through the vLLM OpenAI-compatible Completions API and store token-id logprob evidence."
    )
    parser.add_argument("--input", default="data/prompts.jsonl")
    parser.add_argument("--output", default="data/worker_evidence.openai.jsonl")
    parser.add_argument("--base-url", default=os.environ.get("VLLM_OPENAI_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--quantization", default="bf16")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--backend-version", default="")
    parser.add_argument("--tokenizer-name", default="")
    parser.add_argument("--tokenizer-hash", default="")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--logprobs", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--extra-body-json", default="",
                        help="Extra JSON object or JSON file merged into each /v1/completions request.")
    parser.add_argument("--allow-text-top-logprobs", action="store_true",
                        help="Keep running if top_logprobs keys are text instead of token ids; unmapped top-k entries are dropped.")
    parser.add_argument("--disable-tokenize-fallback", action="store_true",
                        help="Do not call vLLM /tokenize if prompt_token_ids are absent from completion responses.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    client = VLLMOpenAIClient(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )
    variant = make_variant(args)
    decode_config = DecodeConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        do_sample=args.temperature > 0.0,
    )

    samples = []
    for index, payload in enumerate(read_jsonl(Path(args.input)), start=1):
        if args.limit and index > args.limit:
            break
        samples.append((index, parse_worker_sample(payload)))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strict_token_ids = not args.allow_text_top_logprobs
    total_start = time.time()
    written = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for request_index, (input_index, sample) in enumerate(samples, start=1):
            prompts = [sample.input_text]
            body = make_completion_body(args, prompts)
            log(f"Worker OpenAI request {request_index}/{len(samples)}")
            response = client.post_json("/completions", body)
            choice = choices_by_index(response, 1)[0]
            input_ids = extract_prompt_token_ids(response, choice, 0)
            if not input_ids and not args.disable_tokenize_fallback:
                input_ids = client.tokenize(args.model, sample.input_text)
            if not input_ids:
                raise ValueError(
                    "Could not recover input_ids. Keep return_token_ids enabled "
                    "or allow the /tokenize fallback."
                )

            output_ids = extract_completion_token_ids(choice)
            if not output_ids:
                raise ValueError(
                    "Could not recover output_ids from completion response. "
                    "Ensure vLLM supports return_token_ids on /v1/completions."
                )

            row = {
                "sample_id": sample.sample_id,
                "input_text": sample.input_text,
                "input_ids": input_ids,
                "output_text": choice.get("text", ""),
                "output_ids": output_ids,
                "category": sample.category,
                "target_input_tokens": sample.target_input_tokens,
                "worker_variant": dataclass_to_dict(variant),
                "decode_config": dataclass_to_dict(decode_config),
                "worker_trace_mode": "vllm_openai_completions_decode",
                "worker_trace": trace_from_completion_choice(
                    choice,
                    output_ids,
                    strict_token_ids=strict_token_ids,
                ),
            }
            write_jsonl_row(handle, row)
            written += 1
            log(
                f"[{input_index}] sample={sample.sample_id} "
                f"input_tokens={len(input_ids)} output_tokens={len(output_ids)}"
            )

    write_json(output_path.with_suffix(output_path.suffix + ".metadata.json"), {
        "metadata": runtime_metadata(sys.argv),
        "args": vars(args),
        "worker_variant": dataclass_to_dict(variant),
        "sample_count": written,
        "elapsed_seconds": time.time() - total_start,
    })
    log(f"Wrote {written} worker evidence rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
