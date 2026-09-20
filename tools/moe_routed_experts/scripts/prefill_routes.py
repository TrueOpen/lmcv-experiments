#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from routed_common import (
    api_key_from_args,
    completions_url,
    extract_choice,
    now_ms,
    post_json,
    print_progress,
    read_jsonl,
    route_field_from_choice,
    route_summary,
    write_jsonl_row,
)


def load_extra_json(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--extra-json must decode to a JSON object")
    return payload


def model_for_row(args: argparse.Namespace, row: Dict[str, Any]) -> str:
    if args.model:
        return args.model
    request = row.get("request") or {}
    model = request.get("model")
    if not model:
        raise RuntimeError(
            f"Row {row.get('sample_id')} has no request.model; pass --model explicitly."
        )
    return str(model)


def build_prefill_request(args: argparse.Namespace,
                          model: str,
                          input_ids: list[int],
                          output_ids: list[int]) -> Dict[str, Any]:
    route_prompt_start = len(input_ids) if output_ids else max(0, len(input_ids) - 1)
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": input_ids + output_ids,
        "max_tokens": 0,
        "echo": True,
        "temperature": 0.0,
        "stream": False,
        "n": 1,
        "return_token_ids": True,
        "routed_experts_prompt_start": route_prompt_start,
    }
    if args.logprobs is not None:
        payload["logprobs"] = args.logprobs
    if args.prompt_logprobs is not None:
        payload["prompt_logprobs"] = args.prompt_logprobs
    payload.update(load_extra_json(args.extra_json))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay generated output as prompt/prefill and save routed_experts."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None, help="Verifier model. Defaults to each row's request.model.")
    parser.add_argument("--input", default="data/generated_routes.jsonl")
    parser.add_argument("--output", default="data/prefill_routes.jsonl")
    parser.add_argument("--logprobs", type=int, default=None)
    parser.add_argument("--prompt-logprobs", type=int, default=None)
    parser.add_argument("--extra-json", default=None, help="Extra JSON object merged into the request.")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    api_key = api_key_from_args(args.api_key)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(Path(args.input)))
    count = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            sample_id = str(row.get("sample_id"))
            input_ids = [int(token_id) for token_id in row["input_token_ids"]]
            output_ids = [int(token_id) for token_id in row["output_token_ids"]]
            model = model_for_row(args, row)
            request_payload = build_prefill_request(args, model, input_ids, output_ids)
            start_ms = now_ms()
            response = post_json(
                completions_url(args.base_url),
                request_payload,
                api_key=api_key,
                timeout=args.timeout,
            )
            elapsed_ms = now_ms() - start_ms
            choice = extract_choice(response)
            routed_experts = route_field_from_choice(choice)
            if routed_experts is None:
                raise RuntimeError(
                    "Prefill response did not contain routed_experts. "
                    "Start vLLM with --enable-return-routed-experts."
                )
            record = {
                "sample_id": sample_id,
                "source_model": (row.get("request") or {}).get("model"),
                "verifier_model": model,
                "category": row.get("category", ""),
                "target_input_tokens": row.get("target_input_tokens"),
                "anchor_variant": row.get("anchor_variant", ""),
                "source_prompt_metadata": row.get("source_prompt_metadata", {}),
                "input_token_ids": input_ids,
                "output_token_ids": output_ids,
                "prefill_prompt_token_ids": input_ids + output_ids,
                "routed_experts": routed_experts,
                "routed_experts_summary": route_summary(routed_experts),
                "route_scope": "output_prompt_suffix_requested",
                "route_prompt_start": request_payload.get("routed_experts_prompt_start", 0),
                "route_alignment_note": (
                    "Uses input_len because prefill prompt is input_ids + output_ids; "
                    "therefore input_len is inside the full prompt when output_ids is non-empty."
                ),
                "request": {
                    "base_url": args.base_url,
                    "model": model,
                    "max_tokens": 0,
                    "echo": True,
                    "temperature": 0.0,
                    "routed_experts_prompt_start": request_payload.get("routed_experts_prompt_start"),
                    "logprobs": args.logprobs,
                    "prompt_logprobs": args.prompt_logprobs,
                    "extra_json": load_extra_json(args.extra_json),
                },
                "usage": response.get("usage"),
                "response_id": response.get("id"),
                "created": response.get("created"),
                "elapsed_ms": elapsed_ms,
            }
            write_jsonl_row(handle, record)
            count += 1
            print_progress(
                f"[prefill] {sample_id}: input_tokens={len(input_ids)} "
                f"output_tokens={len(output_ids)} "
                f"route_shape={record['routed_experts_summary'].get('shape')}"
            )

    print_progress(f"Wrote {count} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
