#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if path.exists():
        sys.path.insert(0, str(path))

from worker_inference import (  # noqa: E402
    VLLMOpenAIClient,
    choices_by_index,
    compact_dict,
    load_extra_body,
    log,
    normalize_logprob_mapping,
)
from worker_logits_verify.schema import (  # noqa: E402
    VariantInfo,
    dataclass_to_dict,
    read_jsonl,
    runtime_metadata,
    write_json,
)
from worker_logits_verify.verify_vllm_worker_evidence import (  # noqa: E402
    compare_depth,
    write_csv,
    write_summary,
)


JSONDict = Dict[str, Any]


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
            "prompt_logprobs": args.prompt_logprobs,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
        },
    )


def make_verifier_body(args: argparse.Namespace,
                       prompts: Sequence[Sequence[int]]) -> JSONDict:
    body: JSONDict = {
        "model": args.model,
        "prompt": [[int(token_id) for token_id in prompt] for prompt in prompts],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "prompt_logprobs": args.prompt_logprobs,
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


def extract_prompt_logprobs(choice: JSONDict,
                            *,
                            strict_token_ids: bool,
                            sample_id: str) -> List[Dict[str, Dict[str, Any]]]:
    raw = choice.get("prompt_logprobs")
    if raw is None:
        logprobs = choice.get("logprobs") or {}
        if isinstance(logprobs, dict):
            raw = logprobs.get("prompt_logprobs")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError(
            f"sample={sample_id}: prompt_logprobs should be a list, got {type(raw).__name__}"
        )
    return [
        normalize_logprob_mapping(
            item,
            strict_token_ids=strict_token_ids,
            context=f"sample={sample_id} prompt_pos={position}",
        )
        for position, item in enumerate(raw)
    ]


def write_threshold_summary(path: Path,
                            metric_rows: Sequence[JSONDict],
                            *,
                            max_abs_logprob_diff: float,
                            max_missing_selected_rate: float) -> str:
    total = len(metric_rows)
    finite_abs = []
    missing_selected = 0
    over_abs = 0
    for row in metric_rows:
        verifier_logprob = row.get("verifier_selected_logprob")
        if verifier_logprob in {"", None}:
            missing_selected += 1
        try:
            abs_diff = float(row.get("abs_logprob_diff"))
        except (TypeError, ValueError):
            abs_diff = float("nan")
        if math.isfinite(abs_diff):
            finite_abs.append(abs_diff)
            if abs_diff > max_abs_logprob_diff:
                over_abs += 1
    missing_rate = missing_selected / total if total else 0.0
    verdict = (
        "PASS"
        if over_abs == 0 and missing_rate <= max_missing_selected_rate
        else "FAIL"
    )
    lines = [
        "# threshold verdict",
        "",
        f"- verdict: {verdict}",
        f"- depth rows: {total}",
        f"- max_abs_logprob_diff_threshold: {max_abs_logprob_diff:.8g}",
        f"- over_abs_threshold_count: {over_abs}",
        f"- missing_selected_rate: {missing_rate:.8g}",
        f"- max_missing_selected_rate: {max_missing_selected_rate:.8g}",
        f"- observed_max_abs_logprob_diff: {max(finite_abs, default=float('nan')):.8g}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return verdict


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify worker_inference evidence by recomputing prompt logprobs through the vLLM OpenAI-compatible Completions API."
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output-dir", default="results/vllm_openai_worker_verify")
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
    parser.add_argument("--prompt-logprobs", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1,
                        help="Completion tokens to request while collecting prompt_logprobs; generated token is ignored.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--missing-logprob", type=float, default=-100.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-total-tokens", type=int, default=0,
                        help="Skip rows whose input_ids + output_ids exceeds this length.")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--extra-body-json", default="",
                        help="Extra JSON object or JSON file merged into each /v1/completions request.")
    parser.add_argument("--allow-text-top-logprobs", action="store_true",
                        help="Keep running if prompt_logprobs keys are text instead of token ids; unmapped entries are dropped.")
    parser.add_argument("--max-abs-logprob-diff", type=float, default=None,
                        help="Optional hard threshold for PASS/FAIL summary.")
    parser.add_argument("--max-missing-selected-rate", type=float, default=0.0,
                        help="Allowed selected-token missing rate when --max-abs-logprob-diff is set.")
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
    strict_token_ids = not args.allow_text_top_logprobs

    evidence_rows: List[JSONDict] = []
    skipped_too_long = 0
    for index, row in enumerate(read_jsonl(Path(args.evidence)), start=1):
        if args.limit and index > args.limit:
            break
        total_tokens = len(row.get("input_ids", [])) + len(row.get("output_ids", []))
        if args.max_total_tokens and total_tokens > args.max_total_tokens:
            skipped_too_long += 1
            continue
        evidence_rows.append(row)

    log(
        f"Loaded evidence rows: {len(evidence_rows)} "
        f"(skipped_too_long={skipped_too_long})"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: List[JSONDict] = []
    total_start = time.time()

    for request_index, evidence in enumerate(evidence_rows, start=1):
        full_prompt = [
            int(token_id)
            for token_id in evidence["input_ids"] + evidence["output_ids"]
        ]
        log(
            f"Verifier OpenAI prompt_logprobs request {request_index}/{len(evidence_rows)}, "
            f"tokens={len(full_prompt)}"
        )
        response = client.post_json("/completions", make_verifier_body(args, [full_prompt]))
        choice = choices_by_index(response, 1)[0]

        sample_id = str(evidence.get("sample_id", ""))
        input_len = len(evidence.get("input_ids", []))
        prompt_logprobs = extract_prompt_logprobs(
            choice,
            strict_token_ids=strict_token_ids,
            sample_id=sample_id,
        )
        if not prompt_logprobs:
            raise ValueError(
                "Verifier response did not include prompt_logprobs. "
                "Ensure the vLLM OpenAI server supports the prompt_logprobs request field."
            )
        for worker_depth in evidence.get("worker_trace", []):
            depth = int(worker_depth["depth"])
            prompt_pos = input_len + depth
            verifier_logprobs = (
                prompt_logprobs[prompt_pos]
                if prompt_pos < len(prompt_logprobs) else {}
            )
            row = compare_depth(worker_depth, verifier_logprobs, args.missing_logprob)
            row.update({
                "sample_id": sample_id,
                "input_token_count": input_len,
                "output_token_count": len(evidence.get("output_ids", [])),
                "worker_trace_mode": evidence.get("worker_trace_mode"),
                "verifier_trace_mode": "vllm_openai_prompt_logprobs_full_prefill",
            })
            metric_rows.append(row)

    fields = [
        "sample_id",
        "depth",
        "input_token_count",
        "output_token_count",
        "target_token_id",
        "target_decoded_token",
        "worker_trace_mode",
        "verifier_trace_mode",
        "worker_selected_logprob",
        "verifier_selected_logprob",
        "logprob_diff_verifier_minus_worker",
        "abs_logprob_diff",
        "worker_selected_rank",
        "verifier_selected_rank",
        "rank_delta_verifier_minus_worker",
        "common_top_count",
        "union_top_count",
        "topk_jaccard",
        "missing_from_verifier_count",
        "extra_in_verifier_count",
        "common_js_divergence",
        "union_js_divergence",
        "common_prob_cosine",
        "union_prob_cosine",
    ]
    metrics_path = output_dir / "worker_vs_verifier_depth_metrics.csv"
    write_csv(metrics_path, metric_rows, fields)
    write_summary(output_dir / "summary.md", metric_rows)

    verdict: Optional[str] = None
    if args.max_abs_logprob_diff is not None:
        verdict = write_threshold_summary(
            output_dir / "threshold_verdict.md",
            metric_rows,
            max_abs_logprob_diff=args.max_abs_logprob_diff,
            max_missing_selected_rate=args.max_missing_selected_rate,
        )

    write_json(output_dir / "metadata.json", {
        "metadata": runtime_metadata(sys.argv),
        "args": vars(args),
        "verifier_variant": dataclass_to_dict(variant),
        "evidence_count": len(evidence_rows),
        "skipped_too_long": skipped_too_long,
        "depth_row_count": len(metric_rows),
        "elapsed_seconds": time.time() - total_start,
        "verdict": verdict,
    })
    log(f"Wrote {metrics_path}")
    log(f"Wrote {output_dir / 'summary.md'}")
    if verdict is not None:
        log(f"Threshold verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
