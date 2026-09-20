from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import (
    DepthTrace,
    Tokenization,
    TopLogprob,
    VariantInfo,
    dataclass_to_dict,
    depth_trace_to_dict,
    parse_worker_sample,
    read_jsonl,
    runtime_metadata,
    write_json,
    write_jsonl,
)
from .tokenizer_utils import (
    decode_token,
    encode_text,
    parse_probe_token_ids,
    probe_ids_from_texts,
    tokenizer_hash,
    tokenizer_name,
    unique_ints,
)


DEFAULT_PROBE_TEXTS = [
    " yes",
    " no",
    " true",
    " false",
    " accept",
    " reject",
    " valid",
    " invalid",
    "0",
    "1",
    ".",
    ",",
    ":",
    "{",
    "}",
    "\n",
    " the",
    " and",
]


def parse_torch_dtype(torch: object, dtype: str) -> object:
    if dtype == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype for transformers loader: {dtype}")
    return mapping[dtype]


def load_transformers(model_name: str,
                      dtype: str,
                      device_map: str,
                      trust_remote_code: bool,
                      attn_implementation: Optional[str]) -> tuple:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    kwargs = {
        "torch_dtype": parse_torch_dtype(torch, dtype),
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model, torch


def transformers_version() -> str:
    try:
        import transformers

        return transformers.__version__
    except Exception:
        return ""


def make_variant(args: argparse.Namespace,
                 tokenizer: object,
                 backend_version: str) -> VariantInfo:
    tok_hash = tokenizer_hash(tokenizer)
    variant_id = args.variant_id or "-".join(
        item for item in [
            Path(str(args.model)).name,
            args.quantization,
            args.gpu,
            tok_hash,
        ] if item
    )
    return VariantInfo(
        variant_id=variant_id,
        model=args.model,
        tokenizer_hash=tok_hash,
        tokenizer_name=tokenizer_name(tokenizer),
        quantization=args.quantization,
        gpu=args.gpu,
        backend="transformers",
        backend_version=backend_version,
        dtype=args.dtype,
        extra={
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_map": args.device_map,
            "attn_implementation": args.attn_implementation,
        },
    )


def tokenization_for_sample(sample: object,
                            tokenizer: object,
                            tok_hash: str,
                            allow_retokenize: bool) -> Tokenization:
    existing = sample.tokenizations.get(tok_hash)
    if existing is not None:
        return existing
    if not allow_retokenize:
        known = ", ".join(sample.tokenizations.keys())
        raise RuntimeError(
            f"Sample {sample.sample_id} has no tokenization for {tok_hash}. "
            f"Known tokenizers: {known}. Pass --allow-retokenize for cross-tokenizer models."
        )
    input_ids = encode_text(tokenizer, sample.input_text)
    output_text = sample.canonical_output_text
    if not output_text:
        raise RuntimeError(
            f"Sample {sample.sample_id} has no canonical_output_text to retokenize"
        )
    output_ids = encode_text(tokenizer, output_text)
    return Tokenization(
        tokenizer_hash=tok_hash,
        tokenizer_name=tokenizer_name(tokenizer),
        input_ids=input_ids,
        output_ids=output_ids,
    )


def build_probe_ids(tokenizer: object, raw_ids: str, use_default: bool) -> List[int]:
    probe_ids: List[int] = []
    if use_default:
        probe_ids.extend(probe_ids_from_texts(tokenizer, DEFAULT_PROBE_TEXTS))
    probe_ids.extend(parse_probe_token_ids(raw_ids))
    return unique_ints(probe_ids)


def rank_for_token(torch: object, logits: object, token_id: int) -> int:
    target_value = logits[int(token_id)]
    # Rank is 1 + number of logits strictly greater than the target.
    return int(torch.sum(logits > target_value).item()) + 1


def topk_for_logits(torch: object,
                    tokenizer: object,
                    logits: object,
                    logprobs: object,
                    top_k: int) -> List[TopLogprob]:
    values, indices = torch.topk(logprobs, k=min(top_k, int(logprobs.shape[-1])))
    rows = []
    for rank, (logprob, token_id) in enumerate(
        zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()),
        start=1,
    ):
        token_id = int(token_id)
        rows.append(TopLogprob(
            token_id=token_id,
            logprob=float(logprob),
            raw_logit=float(logits[token_id].detach().cpu().item()),
            rank=rank,
            decoded_token=decode_token(tokenizer, token_id),
        ))
    return rows


def top_k_mass(rows: Sequence[TopLogprob]) -> float:
    return sum(math.exp(row.logprob) for row in rows if math.isfinite(row.logprob))


def trace_sample(tokenizer: object,
                 model: object,
                 torch: object,
                 sample_id: str,
                 variant: VariantInfo,
                 input_ids: Sequence[int],
                 output_ids: Sequence[int],
                 top_k: int,
                 probe_ids: Sequence[int],
                 stride: int,
                 max_depth: int,
                 include_raw_logits: bool) -> List[DepthTrace]:
    if not output_ids:
        return []
    device = next(model.parameters()).device
    rows: List[DepthTrace] = []

    input_tensor = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    with torch.inference_mode():
        prefill = model(input_ids=input_tensor, use_cache=True)
    logits = prefill.logits[0, -1, :].detach().float()
    past_key_values = prefill.past_key_values

    limit = len(output_ids)
    if max_depth > 0:
        limit = min(limit, max_depth)

    for depth in range(limit):
        target_token_id = int(output_ids[depth])
        if stride > 1 and depth % stride != 0 and depth != limit - 1:
            next_input = torch.tensor([[target_token_id]], dtype=torch.long, device=device)
            with torch.inference_mode():
                step = model(
                    input_ids=next_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            logits = step.logits[0, -1, :].detach().float()
            past_key_values = step.past_key_values
            continue

        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        top_rows = topk_for_logits(torch, tokenizer, logits, logprobs, top_k)
        selected_rank = rank_for_token(torch, logits, target_token_id)
        selected_logprob = float(logprobs[target_token_id].detach().cpu().item())
        selected_raw_logit = float(logits[target_token_id].detach().cpu().item())

        top1 = top_rows[0] if top_rows else None
        top2 = top_rows[1] if len(top_rows) > 1 else None
        margin = None
        if top1 is not None and top2 is not None:
            margin = float(top1.logprob - top2.logprob)

        probe_logprobs: Dict[str, float] = {}
        probe_raw_logits: Dict[str, float] = {}
        for probe_id in probe_ids:
            if 0 <= int(probe_id) < int(logprobs.shape[-1]):
                key = str(int(probe_id))
                probe_logprobs[key] = float(logprobs[int(probe_id)].detach().cpu().item())
                if include_raw_logits:
                    probe_raw_logits[key] = float(logits[int(probe_id)].detach().cpu().item())

        rows.append(DepthTrace(
            sample_id=sample_id,
            variant_id=variant.variant_id,
            depth=depth,
            input_token_count=len(input_ids),
            output_token_count=len(output_ids),
            target_token_id=target_token_id,
            target_decoded_token=decode_token(tokenizer, target_token_id),
            selected_logprob=selected_logprob,
            selected_raw_logit=selected_raw_logit if include_raw_logits else float("nan"),
            selected_rank=selected_rank,
            top1_token_id=top1.token_id if top1 else -1,
            top1_logprob=top1.logprob if top1 else float("nan"),
            top1_raw_logit=top1.raw_logit if top1 and include_raw_logits else float("nan"),
            top2_token_id=top2.token_id if top2 else None,
            top2_logprob=top2.logprob if top2 else None,
            top2_raw_logit=top2.raw_logit if top2 and include_raw_logits else None,
            top1_top2_margin_logprob=margin,
            top_k_mass=top_k_mass(top_rows),
            top_logprobs=top_rows,
            probe_logprobs=probe_logprobs,
            probe_raw_logits=probe_raw_logits,
            metadata={
                "tokenizer_hash": variant.tokenizer_hash,
            },
        ))

        next_input = torch.tensor([[target_token_id]], dtype=torch.long, device=device)
        with torch.inference_mode():
            step = model(
                input_ids=next_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = step.logits[0, -1, :].detach().float()
        past_key_values = step.past_key_values

    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect depth-aware teacher-forced sparse logits/logprob traces."
    )
    parser.add_argument("--samples", default="data/anchor_samples.jsonl")
    parser.add_argument("--output-dir", default="results/trace")
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant-id", default="")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--quantization", default="bf16")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--probe-token-ids", default="")
    parser.add_argument("--no-default-probes", action="store_true")
    parser.add_argument("--allow-retokenize", action="store_true",
                        help="Allow text-level replay when tokenizer hash differs.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Record every Nth depth. Context is still replayed at every token.")
    parser.add_argument("--max-depth", type=int, default=0,
                        help="Maximum output depth to replay per sample. 0 means all output tokens.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-raw-logits", action="store_true",
                        help="Skip storing raw logits values in the trace rows.")
    args = parser.parse_args(argv)

    tokenizer, model, torch = load_transformers(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    variant = make_variant(args, tokenizer, transformers_version())
    probe_ids = build_probe_ids(
        tokenizer,
        raw_ids=args.probe_token_ids,
        use_default=not args.no_default_probes,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "trace.jsonl"
    metadata_path = output_dir / "metadata.json"

    sample_count = 0
    trace_row_count = 0
    with rows_path.open("w", encoding="utf-8") as handle:
        for index, payload in enumerate(read_jsonl(Path(args.samples)), start=1):
            if args.limit and index > args.limit:
                break
            sample = parse_worker_sample(payload)
            tok = tokenization_for_sample(
                sample,
                tokenizer=tokenizer,
                tok_hash=variant.tokenizer_hash,
                allow_retokenize=args.allow_retokenize,
            )
            sample_rows = trace_sample(
                tokenizer=tokenizer,
                model=model,
                torch=torch,
                sample_id=sample.sample_id,
                variant=variant,
                input_ids=tok.input_ids,
                output_ids=tok.output_ids,
                top_k=args.top_k,
                probe_ids=probe_ids,
                stride=max(1, args.stride),
                max_depth=args.max_depth,
                include_raw_logits=not args.no_raw_logits,
            )
            for row in sample_rows:
                handle.write(json.dumps(
                    depth_trace_to_dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                ))
                handle.write("\n")
                trace_row_count += 1
            sample_count += 1
            print(
                f"[{index}] sample={sample.sample_id} "
                f"input_tokens={len(tok.input_ids)} output_tokens={len(tok.output_ids)} "
                f"trace_rows={len(sample_rows)}"
            )
    write_json(metadata_path, {
        "metadata": runtime_metadata(sys.argv),
        "args": vars(args),
        "variant": dataclass_to_dict(variant),
        "sample_count": sample_count,
        "trace_row_count": trace_row_count,
        "probe_token_ids": probe_ids,
    })
    print(f"Wrote {trace_row_count} trace rows to {rows_path}")
    print(f"Wrote metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
