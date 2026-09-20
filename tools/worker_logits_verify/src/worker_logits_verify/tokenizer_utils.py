from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, List, Sequence


def tokenizer_hash(tokenizer: Any, length: int = 16) -> str:
    """Build a stable-ish tokenizer fingerprint from local tokenizer state."""
    parts = {
        "class": tokenizer.__class__.__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        "chat_template": getattr(tokenizer, "chat_template", None),
    }
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        parts["backend_sha256"] = hashlib.sha256(
            backend.to_str().encode("utf-8")
        ).hexdigest()
    else:
        try:
            vocab_items = sorted(tokenizer.get_vocab().items())
        except Exception:
            vocab_items = []
        parts["vocab_sha256"] = hashlib.sha256(
            json.dumps(vocab_items, sort_keys=True).encode("utf-8")
        ).hexdigest()
    encoded = json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def tokenizer_name(tokenizer: Any) -> str:
    return str(getattr(tokenizer, "name_or_path", "") or tokenizer.__class__.__name__)


def encode_text(tokenizer: Any, text: str) -> List[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return ""


def decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(list(token_ids), skip_special_tokens=False)


def parse_probe_token_ids(raw: str) -> List[int]:
    out: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return unique_ints(out)


def probe_ids_from_texts(tokenizer: Any, texts: Iterable[str]) -> List[int]:
    out: List[int] = []
    for text in texts:
        out.extend(encode_text(tokenizer, text))
    return unique_ints(out)


def unique_ints(values: Iterable[int]) -> List[int]:
    seen = set()
    out = []
    for value in values:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
