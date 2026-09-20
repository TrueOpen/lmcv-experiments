#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union


JSONDict = Dict[str, Any]
JSONLike = Union[JSONDict, List[Any], str, int, float, bool, None]


class VLLMOpenAIError(RuntimeError):
    def __init__(self, message: str, *, status: Optional[int] = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class VLLMOpenAIClient:
    """Small stdlib-only client for vLLM's OpenAI-compatible HTTP server.

    `base_url` accepts either the server root (`http://127.0.0.1:8000`) or
    the OpenAI-style v1 URL (`http://127.0.0.1:8000/v1`). Wrapper methods use
    explicit paths, so custom vLLM endpoints such as `/tokenize` still work.
    """

    base_url: str = field(default_factory=lambda: os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000"))
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", "EMPTY"))
    timeout: float = 600.0
    retries: int = 2
    retry_sleep: float = 1.0
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.server_url = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url

    def _url(self, path: str, query: Optional[Mapping[str, Any]] = None) -> str:
        path = "/" + path.lstrip("/")
        url = self.server_url + path
        if query:
            compact = {key: value for key, value in query.items() if value is not None}
            if compact:
                url += "?" + urllib.parse.urlencode(compact, doseq=True)
        return url

    def _headers(self, extra: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        headers.update(dict(self.extra_headers))
        if extra:
            headers.update(dict(extra))
        return headers

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[JSONLike] = None,
        data: Optional[bytes] = None,
        headers: Optional[Mapping[str, str]] = None,
        query: Optional[Mapping[str, Any]] = None,
        stream: bool = False,
    ) -> Any:
        request_headers = self._headers(headers)
        if json_body is not None:
            data = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")

        request = urllib.request.Request(
            self._url(path, query=query),
            data=data,
            headers=request_headers,
            method=method.upper(),
        )
        return self._open_with_retries(request, path=path, stream=stream)

    def get_json(self, path: str, *, query: Optional[Mapping[str, Any]] = None) -> Any:
        return self.request("GET", path, query=query)

    def post_json(self, path: str, payload: JSONLike) -> Any:
        return self.request("POST", path, json_body=payload)

    def delete_json(self, path: str, payload: Optional[JSONLike] = None) -> Any:
        return self.request("DELETE", path, json_body=payload)

    def stream_json(self, path: str, payload: JSONDict) -> Iterator[JSONDict]:
        response = self.request("POST", path, json_body={**payload, "stream": True}, stream=True)
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                if line == "[DONE]":
                    break
                yield json.loads(line)
        finally:
            response.close()

    def _open_with_retries(self, request: urllib.request.Request, *, path: str, stream: bool) -> Any:
        last_error: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                response = urllib.request.urlopen(request, timeout=self.timeout)
                if stream:
                    return response
                with response:
                    body = response.read().decode("utf-8")
                if not body:
                    return None
                return json.loads(body)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = VLLMOpenAIError(
                    f"HTTP {exc.code} from {path}: {body}",
                    status=exc.code,
                    body=body,
                )
            except urllib.error.URLError as exc:
                last_error = exc
            except json.JSONDecodeError as exc:
                last_error = VLLMOpenAIError(f"Invalid JSON from {path}: {exc}")
            if attempt < self.retries:
                time.sleep(self.retry_sleep)
        assert last_error is not None
        raise last_error

    # Basic / instrumentation endpoints.
    def health(self) -> Any:
        return self.get_json("/health")

    def version(self) -> Any:
        return self.get_json("/version")

    def load(self) -> Any:
        return self.get_json("/load")

    def metrics(self) -> str:
        response = self.request("GET", "/metrics", headers={"Accept": "text/plain"}, stream=True)
        try:
            return response.read().decode("utf-8")
        finally:
            response.close()

    def models(self) -> JSONDict:
        return self.get_json("/v1/models")

    # OpenAI-compatible generation APIs.
    def completions(self, *, model: str, prompt: Any, **params: Any) -> JSONDict:
        return self.post_json("/v1/completions", compact_dict({
            "model": model,
            "prompt": prompt,
            **params,
        }))

    def stream_completions(self, *, model: str, prompt: Any, **params: Any) -> Iterator[JSONDict]:
        return self.stream_json("/v1/completions", compact_dict({
            "model": model,
            "prompt": prompt,
            **params,
        }))

    def chat_completions(self, *, model: str, messages: Sequence[Mapping[str, Any]], **params: Any) -> JSONDict:
        return self.post_json("/v1/chat/completions", compact_dict({
            "model": model,
            "messages": list(messages),
            **params,
        }))

    def stream_chat_completions(self, *, model: str, messages: Sequence[Mapping[str, Any]], **params: Any) -> Iterator[JSONDict]:
        return self.stream_json("/v1/chat/completions", compact_dict({
            "model": model,
            "messages": list(messages),
            **params,
        }))

    def chat_completions_batch(self, payload: JSONDict) -> JSONDict:
        return self.post_json("/v1/chat/completions/batch", payload)

    def create_response(self, *, model: str, input: Any, **params: Any) -> JSONDict:
        return self.post_json("/v1/responses", compact_dict({
            "model": model,
            "input": input,
            **params,
        }))

    def get_response(self, response_id: str) -> JSONDict:
        return self.get_json(f"/v1/responses/{response_id}")

    def cancel_response(self, response_id: str) -> JSONDict:
        return self.post_json(f"/v1/responses/{response_id}/cancel", {})

    # Vector / pooling / ranking APIs.
    def embeddings(self, *, model: str, input: Any, **params: Any) -> JSONDict:
        return self.post_json("/v1/embeddings", compact_dict({
            "model": model,
            "input": input,
            **params,
        }))

    def cohere_embed(self, *, model: str, texts: Sequence[str], **params: Any) -> JSONDict:
        return self.post_json("/v2/embed", compact_dict({
            "model": model,
            "texts": list(texts),
            **params,
        }))

    def pooling(self, *, model: str, input: Any, **params: Any) -> JSONDict:
        return self.post_json("/pooling", compact_dict({
            "model": model,
            "input": input,
            **params,
        }))

    def classify(self, *, model: str, input: Any, **params: Any) -> JSONDict:
        return self.post_json("/classify", compact_dict({
            "model": model,
            "input": input,
            **params,
        }))

    def score(self, *, model: str, text_1: Any, text_2: Any, path: str = "/score", **params: Any) -> JSONDict:
        return self.post_json(path, compact_dict({
            "model": model,
            "text_1": text_1,
            "text_2": text_2,
            **params,
        }))

    def rerank(self, *, model: str, query: str, documents: Sequence[Any], path: str = "/rerank", **params: Any) -> JSONDict:
        return self.post_json(path, compact_dict({
            "model": model,
            "query": query,
            "documents": list(documents),
            **params,
        }))

    def generative_scoring(self, *, model: str, prompt: Any, label_token_ids: Sequence[int], **params: Any) -> JSONDict:
        return self.post_json("/generative_scoring", compact_dict({
            "model": model,
            "prompt": prompt,
            "label_token_ids": [int(token_id) for token_id in label_token_ids],
            **params,
        }))

    # Tokenizer APIs.
    def tokenize(self, *, model: str, prompt: str, add_special_tokens: bool = False, **params: Any) -> JSONDict:
        return self.post_json("/tokenize", compact_dict({
            "model": model,
            "prompt": prompt,
            "add_special_tokens": add_special_tokens,
            **params,
        }))

    def token_ids(self, *, model: str, prompt: str, add_special_tokens: bool = False, **params: Any) -> List[int]:
        response = self.tokenize(
            model=model,
            prompt=prompt,
            add_special_tokens=add_special_tokens,
            **params,
        )
        token_ids = response.get("token_ids") or response.get("tokens") or response.get("input_ids")
        if not isinstance(token_ids, list):
            raise ValueError(f"/tokenize response did not include token ids: {response}")
        return [int(token_id) for token_id in token_ids]

    def detokenize(self, *, model: str, tokens: Sequence[int], **params: Any) -> JSONDict:
        return self.post_json("/detokenize", compact_dict({
            "model": model,
            "tokens": [int(token_id) for token_id in tokens],
            **params,
        }))

    def tokenizer_info(self, *, model: Optional[str] = None) -> JSONDict:
        payload = {"model": model} if model else {}
        return self.post_json("/tokenizer_info", payload)

    # Anthropic-compatible APIs exposed by recent vLLM versions.
    def messages(self, *, model: str, messages: Sequence[Mapping[str, Any]], **params: Any) -> JSONDict:
        return self.post_json("/v1/messages", compact_dict({
            "model": model,
            "messages": list(messages),
            **params,
        }))

    def count_message_tokens(self, *, model: str, messages: Sequence[Mapping[str, Any]], **params: Any) -> JSONDict:
        return self.post_json("/v1/messages/count_tokens", compact_dict({
            "model": model,
            "messages": list(messages),
            **params,
        }))

    # Audio APIs.
    def transcribe(self, *, model: str, file_path: Union[str, Path], **params: Any) -> JSONDict:
        return self._multipart_audio("/v1/audio/transcriptions", model=model, file_path=file_path, params=params)

    def translate(self, *, model: str, file_path: Union[str, Path], **params: Any) -> JSONDict:
        return self._multipart_audio("/v1/audio/translations", model=model, file_path=file_path, params=params)

    # LoRA dynamic loading. Use only when you intentionally enabled it server-side.
    def load_lora_adapter(self, *, lora_name: str, lora_path: str) -> JSONDict:
        return self.post_json("/v1/load_lora_adapter", {
            "lora_name": lora_name,
            "lora_path": lora_path,
        })

    def unload_lora_adapter(self, *, lora_name: str) -> JSONDict:
        return self.post_json("/v1/unload_lora_adapter", {"lora_name": lora_name})

    def _multipart_audio(
        self,
        path: str,
        *,
        model: str,
        file_path: Union[str, Path],
        params: Mapping[str, Any],
    ) -> JSONDict:
        file_path = Path(file_path)
        fields = compact_dict({"model": model, **dict(params)})
        body, content_type = encode_multipart(fields, [("file", file_path)])
        return self.request("POST", path, data=body, headers={"Content-Type": content_type})


def compact_dict(payload: Mapping[str, Any]) -> JSONDict:
    return {key: value for key, value in payload.items() if value is not None}


def load_json_arg(raw: str) -> Any:
    if raw == "":
        return None
    if raw.startswith("@"):
        return json.loads(Path(raw[1:]).read_text(encoding="utf-8"))
    path = Path(raw)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(raw)


def load_text_arg(raw: str) -> str:
    if raw.startswith("@"):
        return Path(raw[1:]).read_text(encoding="utf-8")
    return raw


def parse_kv_pairs(items: Sequence[str]) -> JSONDict:
    out: JSONDict = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=JSON_VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key] = load_json_arg(value)
    return out


def encode_multipart(fields: Mapping[str, Any], files: Sequence[Tuple[str, Path]]) -> Tuple[bytes, str]:
    boundary = "----vllm-client-" + uuid.uuid4().hex
    chunks: List[bytes] = []
    for name, value in fields.items():
        chunks.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
            str(value).encode("utf-8"),
            b"\r\n",
        ])
    for field_name, path in files:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            path.read_bytes(),
            b"\r\n",
        ])
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def build_client(args: argparse.Namespace) -> VLLMOpenAIClient:
    return VLLMOpenAIClient(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default=os.environ.get("VLLM_OPENAI_BASE_URL", os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")),
                        help="vLLM server root or /v1 URL.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=1.0)


def add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)


def add_extra_params(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--param", action="append", default=[],
                        help="Extra request field as KEY=JSON_VALUE, e.g. --param top_k=20.")
    parser.add_argument("--params-json", default="",
                        help="Extra JSON object or @file merged into the request body.")


def merged_params(args: argparse.Namespace) -> JSONDict:
    out: JSONDict = {}
    if args.params_json:
        loaded = load_json_arg(args.params_json)
        if not isinstance(loaded, dict):
            raise TypeError("--params-json must decode to an object")
        out.update(loaded)
    out.update(parse_kv_pairs(args.param))
    return out


def cmd_models(args: argparse.Namespace) -> None:
    print_json(build_client(args).models())


def cmd_health(args: argparse.Namespace) -> None:
    print_json(build_client(args).health())


def cmd_version(args: argparse.Namespace) -> None:
    print_json(build_client(args).version())


def cmd_load(args: argparse.Namespace) -> None:
    print_json(build_client(args).load())


def cmd_metrics(args: argparse.Namespace) -> None:
    print(build_client(args).metrics())


def cmd_completion(args: argparse.Namespace) -> None:
    client = build_client(args)
    params = merged_params(args)
    response = client.completions(
        model=args.model,
        prompt=load_text_arg(args.prompt),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=False,
        **params,
    )
    print_json(response)


def cmd_completion_stream(args: argparse.Namespace) -> None:
    client = build_client(args)
    params = merged_params(args)
    for event in client.stream_completions(
        model=args.model,
        prompt=load_text_arg(args.prompt),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        **params,
    ):
        print_json(event)


def cmd_chat(args: argparse.Namespace) -> None:
    client = build_client(args)
    messages = load_json_arg(args.messages)
    if not isinstance(messages, list):
        raise TypeError("--messages must be a JSON list")
    response = client.chat_completions(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=False,
        **merged_params(args),
    )
    print_json(response)


def cmd_chat_stream(args: argparse.Namespace) -> None:
    client = build_client(args)
    messages = load_json_arg(args.messages)
    if not isinstance(messages, list):
        raise TypeError("--messages must be a JSON list")
    for event in client.stream_chat_completions(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        **merged_params(args),
    ):
        print_json(event)


def cmd_response(args: argparse.Namespace) -> None:
    client = build_client(args)
    response = client.create_response(
        model=args.model,
        input=load_text_arg(args.input),
        **merged_params(args),
    )
    print_json(response)


def cmd_messages(args: argparse.Namespace) -> None:
    client = build_client(args)
    messages = load_json_arg(args.messages)
    if not isinstance(messages, list):
        raise TypeError("--messages must be a JSON list")
    print_json(client.messages(model=args.model, messages=messages, **merged_params(args)))


def cmd_count_message_tokens(args: argparse.Namespace) -> None:
    client = build_client(args)
    messages = load_json_arg(args.messages)
    if not isinstance(messages, list):
        raise TypeError("--messages must be a JSON list")
    print_json(client.count_message_tokens(model=args.model, messages=messages, **merged_params(args)))


def cmd_embedding(args: argparse.Namespace) -> None:
    client = build_client(args)
    input_value = load_json_arg(args.input_json) if args.input_json else load_text_arg(args.input)
    print_json(client.embeddings(model=args.model, input=input_value, **merged_params(args)))


def cmd_cohere_embed(args: argparse.Namespace) -> None:
    texts = load_json_arg(args.texts)
    if not isinstance(texts, list):
        raise TypeError("--texts must be a JSON list of strings")
    print_json(build_client(args).cohere_embed(model=args.model, texts=texts, **merged_params(args)))


def cmd_tokenize(args: argparse.Namespace) -> None:
    print_json(build_client(args).tokenize(
        model=args.model,
        prompt=load_text_arg(args.prompt),
        add_special_tokens=args.add_special_tokens,
        **merged_params(args),
    ))


def cmd_detokenize(args: argparse.Namespace) -> None:
    token_ids = load_json_arg(args.tokens)
    if not isinstance(token_ids, list):
        raise TypeError("--tokens must be a JSON list of token ids")
    print_json(build_client(args).detokenize(model=args.model, tokens=token_ids, **merged_params(args)))


def cmd_tokenizer_info(args: argparse.Namespace) -> None:
    print_json(build_client(args).tokenizer_info(model=args.model))


def cmd_pooling(args: argparse.Namespace) -> None:
    input_value = load_json_arg(args.input_json) if args.input_json else load_text_arg(args.input)
    print_json(build_client(args).pooling(model=args.model, input=input_value, **merged_params(args)))


def cmd_classify(args: argparse.Namespace) -> None:
    input_value = load_json_arg(args.input_json) if args.input_json else load_text_arg(args.input)
    print_json(build_client(args).classify(model=args.model, input=input_value, **merged_params(args)))


def cmd_score(args: argparse.Namespace) -> None:
    print_json(build_client(args).score(
        model=args.model,
        text_1=load_text_arg(args.text_1),
        text_2=load_text_arg(args.text_2),
        path=args.path,
        **merged_params(args),
    ))


def cmd_rerank(args: argparse.Namespace) -> None:
    documents = load_json_arg(args.documents)
    if not isinstance(documents, list):
        raise TypeError("--documents must be a JSON list")
    print_json(build_client(args).rerank(
        model=args.model,
        query=args.query,
        documents=documents,
        path=args.path,
        **merged_params(args),
    ))


def cmd_generative_scoring(args: argparse.Namespace) -> None:
    token_ids = load_json_arg(args.label_token_ids)
    if not isinstance(token_ids, list):
        raise TypeError("--label-token-ids must be a JSON list of token ids")
    print_json(build_client(args).generative_scoring(
        model=args.model,
        prompt=load_text_arg(args.prompt),
        label_token_ids=token_ids,
        **merged_params(args),
    ))


def cmd_transcribe(args: argparse.Namespace) -> None:
    print_json(build_client(args).transcribe(
        model=args.model,
        file_path=args.file,
        **merged_params(args),
    ))


def cmd_translate(args: argparse.Namespace) -> None:
    print_json(build_client(args).translate(
        model=args.model,
        file_path=args.file,
        **merged_params(args),
    ))


def cmd_load_lora(args: argparse.Namespace) -> None:
    print_json(build_client(args).load_lora_adapter(
        lora_name=args.lora_name,
        lora_path=args.lora_path,
    ))


def cmd_unload_lora(args: argparse.Namespace) -> None:
    print_json(build_client(args).unload_lora_adapter(lora_name=args.lora_name))


def cmd_raw(args: argparse.Namespace) -> None:
    client = build_client(args)
    body = load_json_arg(args.body) if args.body else None
    response = client.request(args.method, args.path, json_body=body)
    print_json(response)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stdlib-only wrapper around vLLM's OpenAI-compatible and custom HTTP APIs."
    )
    add_common(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, func in (
        ("models", cmd_models),
        ("health", cmd_health),
        ("version", cmd_version),
        ("load", cmd_load),
        ("metrics", cmd_metrics),
    ):
        child = subparsers.add_parser(name)
        child.set_defaults(func=func)

    completion = subparsers.add_parser("completion")
    add_model(completion)
    add_extra_params(completion)
    completion.add_argument("--prompt", required=True, help="Prompt text or @file.")
    completion.add_argument("--max-tokens", type=int, default=128)
    completion.add_argument("--temperature", type=float, default=0.0)
    completion.add_argument("--top-p", type=float, default=1.0)
    completion.set_defaults(func=cmd_completion)

    completion_stream = subparsers.add_parser("completion-stream")
    add_model(completion_stream)
    add_extra_params(completion_stream)
    completion_stream.add_argument("--prompt", required=True, help="Prompt text or @file.")
    completion_stream.add_argument("--max-tokens", type=int, default=128)
    completion_stream.add_argument("--temperature", type=float, default=0.0)
    completion_stream.add_argument("--top-p", type=float, default=1.0)
    completion_stream.set_defaults(func=cmd_completion_stream)

    chat = subparsers.add_parser("chat")
    add_model(chat)
    add_extra_params(chat)
    chat.add_argument("--messages", required=True, help='JSON list or file path, e.g. \'[{"role":"user","content":"hi"}]\'.')
    chat.add_argument("--max-tokens", type=int, default=128)
    chat.add_argument("--temperature", type=float, default=0.0)
    chat.add_argument("--top-p", type=float, default=1.0)
    chat.set_defaults(func=cmd_chat)

    chat_stream = subparsers.add_parser("chat-stream")
    add_model(chat_stream)
    add_extra_params(chat_stream)
    chat_stream.add_argument("--messages", required=True)
    chat_stream.add_argument("--max-tokens", type=int, default=128)
    chat_stream.add_argument("--temperature", type=float, default=0.0)
    chat_stream.add_argument("--top-p", type=float, default=1.0)
    chat_stream.set_defaults(func=cmd_chat_stream)

    response = subparsers.add_parser("response")
    add_model(response)
    add_extra_params(response)
    response.add_argument("--input", required=True, help="Input text or @file.")
    response.set_defaults(func=cmd_response)

    messages = subparsers.add_parser("messages")
    add_model(messages)
    add_extra_params(messages)
    messages.add_argument("--messages", required=True, help="Anthropic-style JSON message list or @file.")
    messages.set_defaults(func=cmd_messages)

    count_message_tokens = subparsers.add_parser("count-message-tokens")
    add_model(count_message_tokens)
    add_extra_params(count_message_tokens)
    count_message_tokens.add_argument("--messages", required=True, help="Anthropic-style JSON message list or @file.")
    count_message_tokens.set_defaults(func=cmd_count_message_tokens)

    embedding = subparsers.add_parser("embedding")
    add_model(embedding)
    add_extra_params(embedding)
    embedding.add_argument("--input", default="", help="Input text or @file.")
    embedding.add_argument("--input-json", default="", help="JSON input, e.g. a list of strings.")
    embedding.set_defaults(func=cmd_embedding)

    cohere_embed = subparsers.add_parser("cohere-embed")
    add_model(cohere_embed)
    add_extra_params(cohere_embed)
    cohere_embed.add_argument("--texts", required=True, help="JSON list of strings or @file.")
    cohere_embed.set_defaults(func=cmd_cohere_embed)

    tokenize = subparsers.add_parser("tokenize")
    add_model(tokenize)
    add_extra_params(tokenize)
    tokenize.add_argument("--prompt", required=True, help="Prompt text or @file.")
    tokenize.add_argument("--add-special-tokens", action="store_true")
    tokenize.set_defaults(func=cmd_tokenize)

    detokenize = subparsers.add_parser("detokenize")
    add_model(detokenize)
    add_extra_params(detokenize)
    detokenize.add_argument("--tokens", required=True, help="JSON list of token ids.")
    detokenize.set_defaults(func=cmd_detokenize)

    tokenizer_info = subparsers.add_parser("tokenizer-info")
    tokenizer_info.add_argument("--model", default=None)
    tokenizer_info.set_defaults(func=cmd_tokenizer_info)

    pooling = subparsers.add_parser("pooling")
    add_model(pooling)
    add_extra_params(pooling)
    pooling.add_argument("--input", default="", help="Input text or @file.")
    pooling.add_argument("--input-json", default="", help="JSON input.")
    pooling.set_defaults(func=cmd_pooling)

    classify = subparsers.add_parser("classify")
    add_model(classify)
    add_extra_params(classify)
    classify.add_argument("--input", default="", help="Input text or @file.")
    classify.add_argument("--input-json", default="", help="JSON input.")
    classify.set_defaults(func=cmd_classify)

    score = subparsers.add_parser("score")
    add_model(score)
    add_extra_params(score)
    score.add_argument("--text-1", required=True, help="Text or @file.")
    score.add_argument("--text-2", required=True, help="Text or @file.")
    score.add_argument("--path", default="/score", choices=["/score", "/v1/score"])
    score.set_defaults(func=cmd_score)

    rerank = subparsers.add_parser("rerank")
    add_model(rerank)
    add_extra_params(rerank)
    rerank.add_argument("--query", required=True)
    rerank.add_argument("--documents", required=True, help="JSON list of strings or document objects.")
    rerank.add_argument("--path", default="/rerank", choices=["/rerank", "/v1/rerank", "/v2/rerank"])
    rerank.set_defaults(func=cmd_rerank)

    generative_scoring = subparsers.add_parser("generative-scoring")
    add_model(generative_scoring)
    add_extra_params(generative_scoring)
    generative_scoring.add_argument("--prompt", required=True, help="Prompt text or @file.")
    generative_scoring.add_argument("--label-token-ids", required=True, help="JSON list of token ids.")
    generative_scoring.set_defaults(func=cmd_generative_scoring)

    transcribe = subparsers.add_parser("transcribe")
    add_model(transcribe)
    add_extra_params(transcribe)
    transcribe.add_argument("--file", required=True)
    transcribe.set_defaults(func=cmd_transcribe)

    translate = subparsers.add_parser("translate")
    add_model(translate)
    add_extra_params(translate)
    translate.add_argument("--file", required=True)
    translate.set_defaults(func=cmd_translate)

    load_lora = subparsers.add_parser("load-lora")
    load_lora.add_argument("--lora-name", required=True)
    load_lora.add_argument("--lora-path", required=True)
    load_lora.set_defaults(func=cmd_load_lora)

    unload_lora = subparsers.add_parser("unload-lora")
    unload_lora.add_argument("--lora-name", required=True)
    unload_lora.set_defaults(func=cmd_unload_lora)

    raw = subparsers.add_parser("raw")
    raw.add_argument("--method", default="POST", choices=["GET", "POST", "DELETE"])
    raw.add_argument("--path", required=True)
    raw.add_argument("--body", default="", help="JSON object/array/scalar or file path.")
    raw.set_defaults(func=cmd_raw)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except BrokenPipeError:
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
