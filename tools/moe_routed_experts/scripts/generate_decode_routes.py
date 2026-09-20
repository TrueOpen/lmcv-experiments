#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from routed_common import (
    api_key_from_args,
    completions_url,
    extract_choice,
    extract_prompt_token_ids,
    extract_token_ids,
    now_ms,
    post_json,
    print_progress,
    prompt_from_row,
    read_jsonl,
    route_field_from_choice,
    route_summary,
    tokenize_prompt,
    write_jsonl_row,
)


def iter_input_rows(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    if args.prompt is not None:
        yield {"sample_id": "prompt_000001", "prompt": args.prompt}
        return
    if args.prompts_file is None:
        raise SystemExit("Pass --prompt or --prompts-file")
    for index, row in enumerate(read_jsonl(Path(args.prompts_file)), start=1):
        if args.limit and index > args.limit:
            break
        yield row


def load_extra_json(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--extra-json must decode to a JSON object")
    return payload


def build_request(args: argparse.Namespace,
                  prompt: Any,
                  prompt_token_ids: Optional[list[int]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": False,
        "n": 1,
        "return_token_ids": True,
    }
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.logprobs is not None:
        payload["logprobs"] = args.logprobs
    if args.routed_experts_prompt_start is not None:
        payload["routed_experts_prompt_start"] = args.routed_experts_prompt_start
    payload.update(load_extra_json(args.extra_json))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run normal vLLM generation and save output token ids plus routed_experts."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--output", default="data/generated_routes.jsonl")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logprobs", type=int, default=None)
    parser.add_argument("--extra-json", default=None, help="Extra JSON object merged into the request.")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only run the first N prompts from --prompts-file. 0 means all prompts.",
    )
    parser.add_argument(
        "--routed-experts-prompt-start",
        type=int,
        default=None,
        help=(
            "Advanced: pass routed_experts_prompt_start to vLLM. "
            "By default the generation request omits it to avoid vLLM versions "
            "that assert when the value reaches the prompt length."
        ),
    )
    args = parser.parse_args()

    api_key = api_key_from_args(args.api_key)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(iter_input_rows(args), start=1):
            sample_id, prompt, prompt_token_ids = prompt_from_row(row, row_index)
            if prompt_token_ids is None:
                prompt_token_ids = tokenize_prompt(
                    args.base_url,
                    args.model,
                    prompt,
                    api_key=api_key,
                    timeout=args.timeout,
                )
            request_payload = build_request(args, prompt, prompt_token_ids)
            start_ms = now_ms()
            response = post_json(
                completions_url(args.base_url),
                request_payload,
                api_key=api_key,
                timeout=args.timeout,
            )
            elapsed_ms = now_ms() - start_ms
            choice = extract_choice(response)
            output_token_ids = extract_token_ids(choice)
            if output_token_ids is None:
                raise RuntimeError(
                    "Could not find generated token ids in the response. "
                    "Check that your vLLM version supports return_token_ids."
                )
            response_prompt_ids = extract_prompt_token_ids(response, choice)
            routed_experts = route_field_from_choice(choice)
            if routed_experts is None:
                raise RuntimeError(
                    "Response did not contain routed_experts. "
                    "Start vLLM with --enable-return-routed-experts."
                )
            record = {
                "sample_id": sample_id,
                "input_text": prompt if isinstance(prompt, str) else row.get("input_text"),
                "input_token_ids": response_prompt_ids or prompt_token_ids,
                "output_text": choice.get("text", ""),
                "output_token_ids": output_token_ids,
                "category": row.get("category", ""),
                "target_input_tokens": row.get("target_input_tokens"),
                "anchor_variant": row.get("anchor_variant", ""),
                "source_prompt_metadata": row.get("metadata", {}),
                "source_prompt_decode_config": row.get("decode_config"),
                "source_prompt_tokenizations": row.get("tokenizations", {}),
                "routed_experts": routed_experts,
                "routed_experts_summary": route_summary(routed_experts),
                "route_scope": (
                    "custom_prompt_start_requested"
                    if "routed_experts_prompt_start" in request_payload
                    else "default_full_route_window"
                ),
                "route_prompt_start": request_payload.get("routed_experts_prompt_start", 0),
                "route_alignment_note": (
                    "Custom routed_experts_prompt_start was passed through to vLLM."
                    if "routed_experts_prompt_start" in request_payload
                    else "Generation omits routed_experts_prompt_start; vLLM default start is 0."
                ),
                "request": {
                    "base_url": args.base_url,
                    "model": args.model,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "seed": args.seed,
                    "logprobs": args.logprobs,
                    "routed_experts_prompt_start": request_payload.get("routed_experts_prompt_start"),
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
                f"[generate] {sample_id}: input_tokens={len(record['input_token_ids'])} "
                f"output_tokens={len(output_token_ids)} "
                f"route_shape={record['routed_experts_summary'].get('shape')}"
            )

    print_progress(f"Wrote {count} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
