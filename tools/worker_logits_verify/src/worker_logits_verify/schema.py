from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


JSONDict = Dict[str, Any]


@dataclass
class DecodeConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 0
    do_sample: bool = False


@dataclass
class VariantInfo:
    variant_id: str
    model: str
    tokenizer_hash: str
    tokenizer_name: str = ""
    quantization: str = "bf16"
    gpu: str = ""
    backend: str = "transformers"
    backend_version: str = ""
    dtype: str = ""
    extra: JSONDict = field(default_factory=dict)


@dataclass
class Tokenization:
    tokenizer_hash: str
    tokenizer_name: str
    input_ids: List[int]
    output_ids: List[int]


@dataclass
class WorkerSample:
    sample_id: str
    input_text: str
    canonical_output_text: str = ""
    category: str = ""
    target_input_tokens: Optional[int] = None
    anchor_variant: str = ""
    decode_config: Optional[DecodeConfig] = None
    tokenizations: Dict[str, Tokenization] = field(default_factory=dict)
    metadata: JSONDict = field(default_factory=dict)


@dataclass
class TopLogprob:
    token_id: int
    logprob: float
    raw_logit: float
    rank: int
    decoded_token: str = ""


@dataclass
class DepthTrace:
    sample_id: str
    variant_id: str
    depth: int
    input_token_count: int
    output_token_count: int
    target_token_id: int
    target_decoded_token: str
    selected_logprob: float
    selected_raw_logit: float
    selected_rank: int
    top1_token_id: int
    top1_logprob: float
    top1_raw_logit: float
    top2_token_id: Optional[int]
    top2_logprob: Optional[float]
    top2_raw_logit: Optional[float]
    top1_top2_margin_logprob: Optional[float]
    top_k_mass: float
    top_logprobs: List[TopLogprob]
    probe_logprobs: Dict[str, float] = field(default_factory=dict)
    probe_raw_logits: Dict[str, float] = field(default_factory=dict)
    metadata: JSONDict = field(default_factory=dict)


def now_unix() -> float:
    return time.time()


def runtime_metadata(argv: Sequence[str]) -> JSONDict:
    return {
        "created_at_unix": now_unix(),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": list(argv),
    }


def stable_json_hash(payload: Any, length: int = 16) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def dataclass_to_dict(value: Any) -> JSONDict:
    return asdict(value)


def read_jsonl(path: Path) -> Iterator[JSONDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc


def write_jsonl(path: Path, rows: Iterable[JSONDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def read_json(path: Path) -> JSONDict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: JSONDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def parse_decode_config(payload: Optional[JSONDict]) -> Optional[DecodeConfig]:
    if payload is None:
        return None
    return DecodeConfig(**payload)


def parse_tokenizations(payload: Optional[JSONDict]) -> Dict[str, Tokenization]:
    if not payload:
        return {}
    return {
        str(key): Tokenization(**value)
        for key, value in payload.items()
    }


def parse_worker_sample(payload: JSONDict) -> WorkerSample:
    data = dict(payload)
    data["decode_config"] = parse_decode_config(data.get("decode_config"))
    data["tokenizations"] = parse_tokenizations(data.get("tokenizations"))
    return WorkerSample(**data)


def worker_sample_to_dict(sample: WorkerSample) -> JSONDict:
    return dataclass_to_dict(sample)


def top_logprob_to_dict(value: TopLogprob) -> JSONDict:
    return dataclass_to_dict(value)


def depth_trace_to_dict(value: DepthTrace) -> JSONDict:
    return dataclass_to_dict(value)
