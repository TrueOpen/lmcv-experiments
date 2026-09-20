from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .schema import (
    DecodeConfig,
    Tokenization,
    VariantInfo,
    dataclass_to_dict,
    parse_worker_sample,
    read_jsonl,
    runtime_metadata,
    worker_sample_to_dict,
    write_json,
    write_jsonl,
)
from .tokenizer_utils import decode_tokens, encode_text, tokenizer_hash, tokenizer_name


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
    torch_dtype = parse_torch_dtype(torch, dtype)
    kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model, torch


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


def set_seed(torch: object, seed: int) -> None:
    random.seed(seed)
    try:
        torch.manual_seed(seed)
        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def generate_one(tokenizer: object,
                 model: object,
                 torch: object,
                 input_text: str,
                 config: DecodeConfig) -> List[int]:
    encoded = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generation_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "temperature": config.temperature if config.do_sample else None,
        "top_p": config.top_p,
        "top_k": config.top_k if config.top_k > 0 else None,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    generation_kwargs = {
        key: value for key, value in generation_kwargs.items()
        if value is not None
    }
    with torch.inference_mode():
        output = model.generate(**encoded, **generation_kwargs)
    prompt_len = int(encoded["input_ids"].shape[-1])
    return [int(token_id) for token_id in output[0, prompt_len:].detach().cpu().tolist()]


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


def transformers_version() -> str:
    try:
        import transformers

        return transformers.__version__
    except Exception:
        return ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate canonical anchor outputs for a prompt dataset."
    )
    parser.add_argument("--input", default="data/prompts.jsonl")
    parser.add_argument("--output", default="data/anchor_samples.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant-id", default="")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--quantization", default="bf16")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    tokenizer, model, torch = load_transformers(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    set_seed(torch, args.seed)
    variant = make_variant(args, tokenizer, transformers_version())
    config = DecodeConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        do_sample=args.temperature > 0.0,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for index, payload in enumerate(read_jsonl(Path(args.input)), start=1):
            if args.limit and index > args.limit:
                break
            sample = parse_worker_sample(payload)
            output_ids = generate_one(tokenizer, model, torch, sample.input_text, config)
            input_ids = encode_text(tokenizer, sample.input_text)
            output_text = decode_tokens(tokenizer, output_ids)
            tok = Tokenization(
                tokenizer_hash=variant.tokenizer_hash,
                tokenizer_name=variant.tokenizer_name,
                input_ids=input_ids,
                output_ids=output_ids,
            )
            sample.canonical_output_text = output_text
            sample.anchor_variant = variant.variant_id
            sample.decode_config = config
            sample.tokenizations[variant.tokenizer_hash] = tok
            handle.write(json.dumps(
                worker_sample_to_dict(sample),
                ensure_ascii=False,
                sort_keys=True,
            ))
            handle.write("\n")
            sample_count += 1
            print(
                f"[{index}] sample={sample.sample_id} "
                f"input_tokens={len(input_ids)} output_tokens={len(output_ids)}"
            )

    write_json(output_path.with_suffix(output_path.suffix + ".metadata.json"), {
        "metadata": runtime_metadata(sys.argv),
        "args": vars(args),
        "variant": dataclass_to_dict(variant),
        "sample_count": sample_count,
    })
    print(f"Wrote {sample_count} anchor samples to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
