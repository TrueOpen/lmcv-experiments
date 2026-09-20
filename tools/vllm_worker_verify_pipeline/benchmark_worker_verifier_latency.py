#!/usr/bin/env python3
"""Benchmark worker vs verifier latency on vLLM across input/output length grid.

Two flows share one vLLM instance (same model, same GPU state) so the comparison
is apples-to-apples:

* worker   -- normal generation: prefill(input) + decode(output_len tokens),
              SamplingParams(max_tokens=output_len, logprobs=K).
* verifier -- single prefill over worker's input_ids + output_ids with
              SamplingParams(max_tokens=1, prompt_logprobs=K) to recover top-k
              logprobs at every position.

For each (input_len, output_len) cell we run `warmup` unmeasured repeats then
`repeats` measured repeats, and report median / mean / p10 / p90 wall times plus
a worker->verifier ratio. Output length is pinned exactly via ignore_eos +
min_tokens so the sweep axis is clean.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pipeline_common import (
    build_llm_kwargs,
    make_tokens_prompt,
    runtime_metadata,
    write_json,
)


def log(message: str) -> None:
    print(message, flush=True)


def vllm_version() -> str:
    try:
        import vllm

        return vllm.__version__
    except Exception:
        return ""


def parse_int_list(text: str) -> List[int]:
    return [int(chunk) for chunk in text.replace(",", " ").split() if chunk.strip()]


def percentile(values: Sequence[float], q: float) -> float:
    clean = sorted(values)
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(clean) - 1)
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "median": float("nan"),
                "p10": float("nan"), "p90": float("nan"),
                "min": float("nan"), "max": float("nan"), "std": float("nan")}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


@dataclass
class RunTiming:
    total_s: float
    ttft_s: float          # prefill proxy (arrival -> first token); nan if unavailable
    decode_s: float        # total - ttft; nan if ttft unavailable
    gen_tokens: int


def make_random_input_ids(rng: random.Random, length: int,
                          vocab_size: int, reserved: int) -> List[int]:
    # Sample from a "safe" middle band of the vocab to avoid special / reserved ids.
    lo = reserved
    hi = max(reserved + 1, vocab_size - reserved)
    return [rng.randrange(lo, hi) for _ in range(length)]


def request_metrics_timing(request_output: object, total_s: float,
                           gen_tokens: int) -> RunTiming:
    """Break total time into prefill(TTFT) vs decode using vLLM request metrics.

    metrics fields vary across vLLM versions; we defensively pull common ones and
    fall back to nan when unavailable (then only `total_s` is meaningful).
    """
    metrics = getattr(request_output, "metrics", None)
    ttft = float("nan")
    if metrics is not None:
        arrival = getattr(metrics, "arrival_time", None)
        first_scheduled = getattr(metrics, "first_scheduled_time", None)
        first_token = getattr(metrics, "first_token_time", None)
        start = arrival if arrival is not None else first_scheduled
        if start is not None and first_token is not None:
            ttft = max(0.0, float(first_token) - float(start))
    decode = total_s - ttft if ttft == ttft else float("nan")  # nan-safe
    return RunTiming(total_s=total_s, ttft_s=ttft, decode_s=decode,
                     gen_tokens=gen_tokens)


def time_worker(llm: object, input_ids: Sequence[int], output_len: int,
                sampling_cls, top_k: int, seed: int) -> RunTiming:
    from vllm import SamplingParams  # noqa: F401  (sampling_cls already SamplingParams)

    params = sampling_cls(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=output_len,
        min_tokens=output_len,   # pin exact output length ...
        ignore_eos=True,         # ... by never stopping at EOS
        logprobs=top_k,
        prompt_logprobs=None,
        seed=seed,
        detokenize=False,
    )
    prompt = make_tokens_prompt(input_ids)
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], params)
    total_s = time.perf_counter() - t0
    completion = outputs[0].outputs[0]
    gen_tokens = len(list(getattr(completion, "token_ids", [])))
    timing = request_metrics_timing(outputs[0], total_s, gen_tokens)
    output_ids = [int(t) for t in getattr(completion, "token_ids", [])]
    return timing, output_ids


def time_verifier(llm: object, input_ids: Sequence[int],
                  output_ids: Sequence[int], sampling_cls,
                  top_k: int, seed: int) -> RunTiming:
    params = sampling_cls(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        prompt_logprobs=top_k,
        logprobs=None,
        seed=seed,
        detokenize=False,
    )
    prompt = make_tokens_prompt(list(input_ids) + list(output_ids))
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], params)
    total_s = time.perf_counter() - t0
    # verifier is a single forward: whole thing is "prefill".
    return request_metrics_timing(outputs[0], total_s, gen_tokens=0)


def build_sampling_cls():
    from vllm import SamplingParams
    return SamplingParams


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark worker vs verifier latency across input/output length grid."
    )
    # model / engine
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantization", default="bf16")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-logprobs", type=int, default=None,
                        help="Must be >= top-k; defaults to top-k when unset.")
    # sweep grid
    parser.add_argument("--input-lengths", default="128 512 1024 2048 4096",
                        help="Comma/space separated prefill input token counts.")
    parser.add_argument("--output-lengths", default="1 16 64 256 1024",
                        help="Comma/space separated worker generation lengths.")
    parser.add_argument("--top-k", type=int, default=20,
                        help="logprobs / prompt_logprobs top-k for both flows.")
    # timing
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-verifier", action="store_true")
    parser.add_argument("--skip-worker", action="store_true",
                        help="Verifier-only mode still needs output_ids; uses random ids.")
    parser.add_argument("--max-total-tokens", type=int, default=0,
                        help="Skip a grid cell if input+output exceeds this (0=off).")
    # output
    parser.add_argument("--output-dir", default="results/worker_verifier_latency")
    parser.add_argument("--gpu", default="", help="Free-form GPU label recorded in metadata.")
    args = parser.parse_args(argv)

    input_lengths = parse_int_list(args.input_lengths)
    output_lengths = parse_int_list(args.output_lengths)
    top_k = args.top_k
    if args.max_logprobs is None:
        args.max_logprobs = top_k

    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 32000)

    llm_kwargs = build_llm_kwargs(args)
    log(f"Loading vLLM with kwargs: {llm_kwargs}")
    llm = LLM(**llm_kwargs)
    sampling_cls = build_sampling_cls()

    rng = random.Random(args.seed)
    reserved = min(256, max(1, vocab_size // 100))

    raw_rows: List[Dict[str, object]] = []
    agg_rows: List[Dict[str, object]] = []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "latency_raw.jsonl"
    raw_handle = raw_path.open("w", encoding="utf-8")

    grid: List[Tuple[int, int]] = [
        (i, o) for i in input_lengths for o in output_lengths
    ]
    total_start = time.time()

    for cell_idx, (in_len, out_len) in enumerate(grid, start=1):
        total_tokens = in_len + out_len
        if args.max_total_tokens and total_tokens > args.max_total_tokens:
            log(f"[{cell_idx}/{len(grid)}] skip in={in_len} out={out_len} "
                f"(total {total_tokens} > {args.max_total_tokens})")
            continue

        log(f"[{cell_idx}/{len(grid)}] input_len={in_len} output_len={out_len} "
            f"warmup={args.warmup} repeats={args.repeats}")

        worker_total: List[float] = []
        worker_ttft: List[float] = []
        worker_decode: List[float] = []
        verifier_total: List[float] = []
        worker_tok_s: List[float] = []
        verifier_tok_s: List[float] = []
        gen_tokens_seen = 0

        n_iters = args.warmup + args.repeats
        for it in range(n_iters):
            measured = it >= args.warmup
            iter_seed = args.seed + it
            input_ids = make_random_input_ids(rng, in_len, vocab_size, reserved)

            output_ids: List[int]
            if not args.skip_worker:
                w_timing, output_ids = time_worker(
                    llm, input_ids, out_len, sampling_cls, top_k, iter_seed
                )
                if measured:
                    worker_total.append(w_timing.total_s)
                    if w_timing.ttft_s == w_timing.ttft_s:
                        worker_ttft.append(w_timing.ttft_s)
                        worker_decode.append(w_timing.decode_s)
                    if w_timing.gen_tokens > 0:
                        worker_tok_s.append(w_timing.gen_tokens / w_timing.total_s)
                    gen_tokens_seen = w_timing.gen_tokens
            else:
                output_ids = make_random_input_ids(rng, out_len, vocab_size, reserved)
                gen_tokens_seen = out_len

            if not args.skip_verifier:
                v_timing = time_verifier(
                    llm, input_ids, output_ids, sampling_cls, top_k, iter_seed
                )
                if measured:
                    verifier_total.append(v_timing.total_s)
                    verifier_tok_s.append(total_tokens / v_timing.total_s)

            if measured:
                raw_row = {
                    "input_len": in_len,
                    "output_len": out_len,
                    "total_tokens": total_tokens,
                    "iter": it - args.warmup,
                    "worker_total_s": (worker_total[-1] if worker_total and not args.skip_worker else None),
                    "verifier_total_s": (verifier_total[-1] if verifier_total and not args.skip_verifier else None),
                }
                raw_handle.write(json.dumps(raw_row, ensure_ascii=False) + "\n")
                raw_handle.flush()

        w_sum = summarize(worker_total)
        v_sum = summarize(verifier_total)
        ttft_sum = summarize(worker_ttft)
        decode_sum = summarize(worker_decode)
        ratio_median = (
            w_sum["median"] / v_sum["median"]
            if v_sum["median"] == v_sum["median"] and v_sum["median"] > 0
            and w_sum["median"] == w_sum["median"]
            else float("nan")
        )

        agg = {
            "input_len": in_len,
            "output_len": out_len,
            "total_tokens": total_tokens,
            "gen_tokens": gen_tokens_seen,
            "worker_total_median_s": w_sum["median"],
            "worker_total_mean_s": w_sum["mean"],
            "worker_total_p10_s": w_sum["p10"],
            "worker_total_p90_s": w_sum["p90"],
            "worker_ttft_median_s": ttft_sum["median"],
            "worker_decode_median_s": decode_sum["median"],
            "worker_tok_per_s_median": summarize(worker_tok_s)["median"],
            "verifier_total_median_s": v_sum["median"],
            "verifier_total_mean_s": v_sum["mean"],
            "verifier_total_p10_s": v_sum["p10"],
            "verifier_total_p90_s": v_sum["p90"],
            "verifier_tok_per_s_median": summarize(verifier_tok_s)["median"],
            "worker_over_verifier_median": ratio_median,
        }
        agg_rows.append(agg)
        log(
            f"    worker median={w_sum['median']:.4f}s "
            f"(ttft={ttft_sum['median']:.4f}s decode={decode_sum['median']:.4f}s) | "
            f"verifier median={v_sum['median']:.4f}s | "
            f"worker/verifier={ratio_median:.3f}"
        )

    raw_handle.close()

    # aggregated CSV
    fields = [
        "input_len", "output_len", "total_tokens", "gen_tokens",
        "worker_total_median_s", "worker_total_mean_s",
        "worker_total_p10_s", "worker_total_p90_s",
        "worker_ttft_median_s", "worker_decode_median_s",
        "worker_tok_per_s_median",
        "verifier_total_median_s", "verifier_total_mean_s",
        "verifier_total_p10_s", "verifier_total_p90_s",
        "verifier_tok_per_s_median",
        "worker_over_verifier_median",
    ]
    csv_path = output_dir / "latency_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in agg_rows:
            writer.writerow({f: row.get(f, "") for f in fields})

    write_markdown_summary(output_dir / "summary.md", agg_rows, args, top_k)

    write_json(output_dir / "metadata.json", {
        "metadata": runtime_metadata(sys.argv),
        "args": vars(args),
        "vllm_version": vllm_version(),
        "vocab_size": vocab_size,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "input_lengths": input_lengths,
        "output_lengths": output_lengths,
        "elapsed_seconds": time.time() - total_start,
        "cell_count": len(agg_rows),
    })

    log(f"Wrote {csv_path}")
    log(f"Wrote {raw_path}")
    log(f"Wrote {output_dir / 'summary.md'}")
    log(f"Done in {time.time() - total_start:.1f}s")
    return 0


def write_markdown_summary(path: Path, rows: Sequence[Dict[str, object]],
                           args: argparse.Namespace, top_k: int) -> None:
    def fmt(x: object) -> str:
        try:
            v = float(x)  # type: ignore[arg-type]
            if v != v:
                return "-"
            return f"{v:.4f}"
        except (TypeError, ValueError):
            return "-"

    lines = [
        "# Worker vs Verifier latency benchmark",
        "",
        f"- model: `{args.model}` (quant={args.quantization}, dtype={args.dtype})",
        f"- top_k logprobs: {top_k}",
        f"- warmup={args.warmup}, repeats={args.repeats}, tp={args.tensor_parallel_size}",
        f"- gpu: {args.gpu or '(unset)'}",
        "",
        "Worker = prefill(input) + decode(output_len). "
        "Verifier = single prefill over input+output for top-k prompt logprobs.",
        "",
        "| input | output | total | worker med (s) | w ttft | w decode | "
        "verifier med (s) | worker/verifier |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {inl} | {outl} | {tot} | {wt} | {ttft} | {dec} | {vt} | {ratio} |".format(
                inl=r["input_len"], outl=r["output_len"], tot=r["total_tokens"],
                wt=fmt(r["worker_total_median_s"]),
                ttft=fmt(r["worker_ttft_median_s"]),
                dec=fmt(r["worker_decode_median_s"]),
                vt=fmt(r["verifier_total_median_s"]),
                ratio=fmt(r["worker_over_verifier_median"]),
            )
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- `worker/verifier` > 1 means the worker (generation) is slower "
                 "than the verifier (single prefill) for that cell.")
    lines.append("- ttft/decode split comes from vLLM request metrics; shown as `-` "
                 "when the running vLLM version does not expose them.")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
