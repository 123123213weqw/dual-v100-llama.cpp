#!/usr/bin/env python3
"""Fail-closed, single-pass correctness/performance gate for a candidate llama-server.

The gate deliberately does not know how to stop, restart, or otherwise mutate the
production service.  It only starts the exact binary named in a JSON config, on
loopback, after proving that the requested port and both GPUs are idle.  The
child's process group is the only process group it will ever terminate.

This is a quick gate, not the seven-run V100-X benchmark.  It exercises:

* the legacy /completion path with exactly 71,351 input token IDs;
* a DeepSeek-Harness-shaped streaming grammar/tool request; and
* a second streaming request that replays reasoning_content + tool_calls and a
  tool result, thereby exercising prompt-cache continuation.

Every request, response, SSE stream, metrics snapshot/delta, server-log slice,
and reproducibility fact is written under a new result directory.  Any
"inconsistent sequence positions" or "llama_decode ... returned -1" message is
a hard failure.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import pathlib
import re
import shlex
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, BinaryIO, Iterable


LEGACY_PROMPT_TOKENS = 71_351
LEGACY_OUTPUT_TOKENS = 512
DEFAULT_CHAT_OUTPUT_TOKENS = 1_024
PRODUCTION_PORT = 8_000

ERROR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "position_inconsistency",
        re.compile(r"inconsistent sequence positions", re.IGNORECASE),
    ),
    (
        "decode_minus_one",
        re.compile(r"llama_decode(?:\[[^\]]+\])?\s+returned\s+-1\b", re.IGNORECASE),
    ),
)

METRIC_NAMES = (
    "llamacpp:prompt_tokens_total",
    "llamacpp:prompt_tokens_cached_total",
    "llamacpp:prompt_seconds_total",
    "llamacpp:tokens_predicted_total",
    "llamacpp:tokens_predicted_seconds_total",
    "llamacpp:n_decode_total",
    "llamacpp:n_tokens_max",
    "llamacpp:spec_decode_num_draft_tokens_total",
    "llamacpp:spec_decode_num_accepted_tokens_total",
    "llamacpp:spec_decode_num_drafts_total",
)

SENSITIVE_KEY = re.compile(r"(?:token|secret|password|passwd|api[_-]?key|credential)", re.I)


class GateError(RuntimeError):
    """A preflight, transport, protocol, or correctness gate failure."""


@dataclass
class ManagedServer:
    process: subprocess.Popen[bytes]
    log_file: BinaryIO
    log_path: pathlib.Path


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def json_dump(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def text_dump(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_token_ids(tokens: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(struct.pack("<i", int(token)))
    return digest.hexdigest()


def redact(key: str, value: Any) -> Any:
    if SENSITIVE_KEY.search(key):
        return "<redacted>"
    return value


def redacted_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: redact(key, value) for key, value in mapping.items()}


def capture_command(argv: list[str], timeout: float = 15.0) -> dict[str, Any]:
    try:
        result = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
            text=True,
        )
        return {
            "argv": argv,
            "returncode": result.returncode,
            "output": result.stdout,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error": str(exc)}


def gpu_compute_processes() -> list[dict[str, str]]:
    """Return active CUDA compute clients, or fail when the guard cannot run."""
    argv = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GateError(f"cannot enforce idle-GPU guard: {exc}") from exc
    if result.returncode != 0:
        raise GateError(
            "cannot enforce idle-GPU guard; nvidia-smi failed: "
            + (result.stderr.strip() or result.stdout.strip())
        )
    active: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) == 3 and fields[0] not in {"", "[N/A]", "N/A"}:
            active.append({"pid": fields[0], "process_name": fields[1], "used_memory_mib": fields[2]})
    return active


def assert_only_owned_gpu_process(process: subprocess.Popen[bytes]) -> None:
    active = gpu_compute_processes()
    foreign = [item for item in active if item.get("pid") != str(process.pid)]
    if foreign:
        raise GateError(
            "a foreign CUDA client appeared while the candidate was loading; refusing to send work: "
            + json.dumps(foreign, ensure_ascii=False)
        )
    if not active:
        raise GateError("healthy candidate has no visible CUDA compute process")


def assert_loopback(host: str) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise GateError(f"host must be loopback, got {host!r}")


def assert_safe_port(host: str, port: int) -> None:
    if port == PRODUCTION_PORT:
        raise GateError(f"port {PRODUCTION_PORT} is reserved for production and is never allowed")
    if not 1_024 <= port <= 65_535:
        raise GateError(f"candidate port must be in [1024, 65535], got {port}")
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    bind_host = "::1" if host == "::1" else "127.0.0.1"
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((bind_host, port))
        except OSError as exc:
            raise GateError(f"candidate port {host}:{port} is already in use: {exc}") from exc


def normalize_server_args(args: list[Any]) -> list[str]:
    values = [str(arg) for arg in args]
    forbidden = {"--host", "--port", "--metrics", "--no-metrics", "-m", "--model"}
    for arg in values:
        option = arg.split("=", 1)[0]
        if option in forbidden:
            raise GateError(
                f"server_args contains managed option {option!r}; use the top-level config fields instead"
            )
    return values


def request_headers(api_key: str | None = None) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "v100-extreme-quick-gate/1",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def http_get(url: str, timeout: float, api_key: str | None = None) -> bytes:
    request = urllib.request.Request(url, method="GET", headers=request_headers(api_key))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise GateError(f"GET {url} returned HTTP {exc.code}: {body[:2000]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise GateError(f"GET {url} failed: {exc}") from exc


def http_post_json(
    url: str,
    body: dict[str, Any],
    timeout: float,
    api_key: str | None = None,
) -> tuple[bytes, float]:
    payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers=request_headers(api_key),
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise GateError(f"POST {url} returned HTTP {exc.code}: {response_body[:4000]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise GateError(f"POST {url} failed: {exc}") from exc
    return raw, time.perf_counter() - started


def wait_for_server(base_url: str, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise GateError(f"candidate server exited during startup with code {process.returncode}")
        try:
            # /health is deliberately unauthenticated in llama-server.
            raw = http_get(f"{base_url}/health", timeout=2)
            payload = json.loads(raw)
            if payload.get("status") == "ok":
                return
            last_error = f"health={payload!r}"
        except (GateError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise GateError(f"candidate server did not become healthy in {timeout:.0f}s: {last_error}")


def wait_for_external_server(base_url: str, timeout: float) -> None:
    """Wait for a separately managed candidate without owning or signalling it."""
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            raw = http_get(f"{base_url}/health", timeout=2)
            payload = json.loads(raw)
            if payload.get("status") == "ok":
                return
            last_error = f"health={payload!r}"
        except (GateError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise GateError(f"external candidate did not become healthy in {timeout:.0f}s: {last_error}")


def external_server_config(config: dict[str, Any]) -> tuple[str, pathlib.Path] | None:
    value = config.get("external_server")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise GateError("external_server must be a JSON object")
    base_url = str(value.get("base_url", "")).rstrip("/")
    log_value = value.get("server_log")
    if not base_url or not log_value:
        raise GateError("external_server requires base_url and server_log")
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "http" or parsed.username or parsed.password or parsed.path not in {"", "/"}:
        raise GateError("external_server.base_url must be a plain loopback http://host:port URL")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise GateError(f"external candidate must be loopback, got {parsed.hostname!r}")
    try:
        port = parsed.port
    except ValueError as exc:
        raise GateError(f"invalid external candidate URL: {base_url}") from exc
    if port is None or port == PRODUCTION_PORT:
        raise GateError("external candidate needs an explicit non-production port")
    log_path = pathlib.Path(str(log_value)).expanduser().resolve()
    return base_url, log_path


def parse_prometheus(raw: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            parsed[fields[0]] = float(fields[1])
        except ValueError:
            continue
    return parsed


def metrics_snapshot(base_url: str, timeout: float, api_key: str | None) -> tuple[str, dict[str, float]]:
    raw = http_get(f"{base_url}/metrics", timeout=timeout, api_key=api_key).decode(
        "utf-8", errors="replace"
    )
    parsed = parse_prometheus(raw)
    missing = [name for name in METRIC_NAMES if name not in parsed]
    if missing:
        raise GateError(f"/metrics omitted required counters: {', '.join(missing)}")
    return raw, parsed


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, Any]:
    all_series = sorted(set(before) | set(after))
    delta = {
        name: after.get(name, 0.0) - before.get(name, 0.0)
        for name in all_series
        if name.startswith("llamacpp:")
    }
    prompt_tokens = delta.get("llamacpp:prompt_tokens_total", 0.0)
    prompt_seconds = delta.get("llamacpp:prompt_seconds_total", 0.0)
    predicted_tokens = delta.get("llamacpp:tokens_predicted_total", 0.0)
    predicted_seconds = delta.get("llamacpp:tokens_predicted_seconds_total", 0.0)
    drafted = delta.get("llamacpp:spec_decode_num_draft_tokens_total", 0.0)
    accepted = delta.get("llamacpp:spec_decode_num_accepted_tokens_total", 0.0)
    verifications = delta.get("llamacpp:spec_decode_num_drafts_total", 0.0)
    derived = {
        "prompt_tokens_per_second": prompt_tokens / prompt_seconds if prompt_seconds > 0 else None,
        "predicted_tokens_per_second": (
            predicted_tokens / predicted_seconds if predicted_seconds > 0 else None
        ),
        "mtp_acceptance_ratio": accepted / drafted if drafted > 0 else None,
        "mtp_mean_accepted_tokens_per_verification": (
            accepted / verifications if verifications > 0 else None
        ),
        "mtp_mean_committed_tokens_per_verification": (
            1.0 + accepted / verifications if verifications > 0 else None
        ),
    }
    return {"counters": delta, "derived": derived}


def log_slice(path: pathlib.Path, start: int) -> tuple[int, str]:
    with path.open("rb") as handle:
        handle.seek(start)
        raw = handle.read()
        end = handle.tell()
    return end, raw.decode("utf-8", errors="replace")


def scan_log(value: str) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for name, pattern in ERROR_PATTERNS:
        matching = [line for line in value.splitlines() if pattern.search(line)]
        if matching:
            hits[name] = matching
    return hits


def semantic_delta(delta: dict[str, Any]) -> bool:
    if delta.get("content"):
        return True
    if delta.get("reasoning_content"):
        return True
    for call in delta.get("tool_calls") or []:
        function = call.get("function") or {}
        if call.get("id") or function.get("name") or function.get("arguments"):
            return True
    return False


def stream_chat(
    url: str,
    body: dict[str, Any],
    raw_path: pathlib.Path,
    timeout: float,
    api_key: str | None,
) -> dict[str, Any]:
    payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={**request_headers(api_key), "Accept": "text/event-stream"},
    )
    started = time.perf_counter()
    first_semantic: float | None = None
    raw = bytearray()
    content: list[str] = []
    reasoning: list[str] = []
    tools: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    chunks = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                line = response.readline()
                if not line:
                    break
                raw.extend(line)
                stripped = line.strip()
                if not stripped.startswith(b"data:"):
                    continue
                data = stripped[len(b"data:") :].strip()
                if data == b"[DONE]":
                    break
                if not data:
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise GateError(f"malformed SSE JSON at chunk {chunks}: {data[:500]!r}") from exc
                chunks += 1
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
                    delta = choice.get("delta") or {}
                    if first_semantic is None and semantic_delta(delta):
                        first_semantic = time.perf_counter() - started
                    if delta.get("content") is not None:
                        content.append(str(delta["content"]))
                    if delta.get("reasoning_content") is not None:
                        reasoning.append(str(delta["reasoning_content"]))
                    for fragment in delta.get("tool_calls") or []:
                        index = int(fragment.get("index", 0))
                        accumulated = tools.setdefault(
                            index,
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if fragment.get("id"):
                            accumulated["id"] += str(fragment["id"])
                        if fragment.get("type"):
                            accumulated["type"] = str(fragment["type"])
                        function = fragment.get("function") or {}
                        if function.get("name"):
                            accumulated["function"]["name"] += str(function["name"])
                        if function.get("arguments"):
                            accumulated["function"]["arguments"] += str(function["arguments"])
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise GateError(f"POST {url} returned HTTP {exc.code}: {response_body[:4000]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise GateError(f"streaming POST {url} failed: {exc}") from exc
    finally:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw)

    wall = time.perf_counter() - started
    return {
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "tool_calls": [tools[index] for index in sorted(tools)],
        "finish_reason": finish_reason,
        "usage": usage,
        "chunks": chunks,
        "ttft_seconds": first_semantic,
        "wall_seconds": wall,
    }


def benchmark_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "write_benchmark_record",
            "description": "Persist one structured benchmark record. Call exactly once when requested.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["case", "summary", "measurements", "metadata"],
                "properties": {
                    "case": {"type": "string", "enum": ["grammar-tool", "continuation"]},
                    "summary": {"type": "string", "minLength": 8, "maxLength": 160},
                    "measurements": {
                        "type": "array",
                        "minItems": 16,
                        "maxItems": 16,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["index", "value", "status"],
                            "properties": {
                                "index": {"type": "integer", "minimum": 0, "maximum": 15},
                                "value": {"type": "number", "minimum": -1000000, "maximum": 1000000},
                                "status": {"type": "string", "enum": ["ok", "warmup", "outlier"]},
                            },
                        },
                    },
                    "metadata": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["accelerator", "precision", "exact_context_tokens"],
                        "properties": {
                            "accelerator": {"type": "string", "enum": ["V100"]},
                            "precision": {"type": "string", "enum": ["Q8_0"]},
                            "exact_context_tokens": {"type": "integer", "minimum": 0},
                        },
                    },
                },
            },
        },
    }


def initial_chat_request(model_alias: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model_alias,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a deterministic benchmark fixture. Follow the requested tool schema "
                    "exactly. Do not print prose after making the tool call."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Call write_benchmark_record exactly once. Set case to grammar-tool. Produce "
                    "exactly 16 measurements with indexes 0 through 15 in order, value equal to "
                    "index * 1.25, and status ok except index 0 is warmup. Set accelerator V100, "
                    "precision Q8_0, and exact_context_tokens 71351. Do not answer outside the tool call."
                ),
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "thinking": {"type": "enabled"},
        "reasoning_effort": "xhigh",
        "tools": [benchmark_tool()],
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def continuation_chat_request(
    first: dict[str, Any], model_alias: str, max_tokens: int
) -> dict[str, Any]:
    if not first.get("tool_calls"):
        raise GateError("turn 1 produced no tool call; a faithful continuation cannot be constructed")
    assistant: dict[str, Any] = {
        "role": "assistant",
        # Exact DeepSeek Harness passback rule: never null on a tool-call turn.
        "content": first.get("content") or "",
        "tool_calls": first["tool_calls"],
    }
    if first.get("reasoning_content"):
        assistant["reasoning_content"] = first["reasoning_content"]
    request = initial_chat_request(model_alias, max_tokens)
    request["messages"].append(assistant)
    for index, call in enumerate(first["tool_calls"]):
        request["messages"].append(
            {
                "role": "tool",
                "tool_call_id": call.get("id") or f"quick-gate-call-{index}",
                "content": json.dumps(
                    {"stored": True, "record_id": f"gate-turn-1-{index}"}, separators=(",", ":")
                ),
            }
        )
    request["messages"].append(
        {
            "role": "user",
            "content": (
                "Now call write_benchmark_record exactly once again. Set case to continuation. "
                "Again emit exactly 16 ordered measurements; value is index * -2, every status is ok, "
                "and metadata remains V100, Q8_0, 71351. Do not answer outside the tool call."
            ),
        }
    )
    return request


def validate_tool_response(response: dict[str, Any], expected_case: str) -> dict[str, Any]:
    errors: list[str] = []
    calls = response.get("tool_calls") or []
    if len(calls) != 1:
        errors.append(f"expected exactly one tool call, got {len(calls)}")
    arguments: Any = None
    if calls:
        call = calls[0]
        if not call.get("id"):
            errors.append("tool call id is empty")
        name = ((call.get("function") or {}).get("name"))
        if name != "write_benchmark_record":
            errors.append(f"unexpected tool function {name!r}")
        raw_arguments = (call.get("function") or {}).get("arguments", "")
        try:
            arguments = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append(f"tool arguments are not valid JSON: {exc}")
    if isinstance(arguments, dict):
        if arguments.get("case") != expected_case:
            errors.append(f"case is {arguments.get('case')!r}, expected {expected_case!r}")
        measurements = arguments.get("measurements")
        if not isinstance(measurements, list) or len(measurements) != 16:
            errors.append("measurements must contain exactly 16 items")
        elif [item.get("index") for item in measurements if isinstance(item, dict)] != list(range(16)):
            errors.append("measurement indexes must be exactly 0..15 in order")
        else:
            for index, item in enumerate(measurements):
                if not isinstance(item, dict):
                    errors.append(f"measurement {index} is not an object")
                    continue
                expected_value = index * 1.25 if expected_case == "grammar-tool" else index * -2
                expected_status = "warmup" if expected_case == "grammar-tool" and index == 0 else "ok"
                if item.get("value") != expected_value:
                    errors.append(
                        f"measurement {index} value is {item.get('value')!r}, expected {expected_value!r}"
                    )
                if item.get("status") != expected_status:
                    errors.append(
                        f"measurement {index} status is {item.get('status')!r}, expected {expected_status!r}"
                    )
        metadata = arguments.get("metadata")
        expected_metadata = {
            "accelerator": "V100",
            "precision": "Q8_0",
            "exact_context_tokens": LEGACY_PROMPT_TOKENS,
        }
        if not isinstance(metadata, dict) or any(
            metadata.get(key) != value for key, value in expected_metadata.items()
        ):
            errors.append(f"metadata does not match {expected_metadata!r}")
    else:
        errors.append("tool arguments did not decode to an object")
    if response.get("ttft_seconds") is None:
        errors.append("stream had no semantic delta, so TTFT is undefined")
    return {"passed": not errors, "errors": errors, "arguments": arguments}


def semantic_response_hash(response: dict[str, Any]) -> str:
    """Hash model semantics only; exclude wall time, TTFT, chunks, and usage."""
    canonical = {
        "content": response.get("content") or "",
        "reasoning_content": response.get("reasoning_content") or "",
        "tool_calls": response.get("tool_calls") or [],
        "finish_reason": response.get("finish_reason"),
    }
    return sha256_text(json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def run_case(
    *,
    name: str,
    out: pathlib.Path,
    base_url: str,
    body: dict[str, Any],
    endpoint: str,
    server_log: pathlib.Path,
    log_offset: int,
    timeout: float,
    api_key: str | None,
    streaming: bool,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    case_dir = out / name
    case_dir.mkdir(parents=True, exist_ok=False)
    json_dump(case_dir / "request.json", body)

    before_raw, before = metrics_snapshot(base_url, timeout=30, api_key=api_key)
    text_dump(case_dir / "metrics.before.prom", before_raw)
    request_started = utc_now()
    if streaming:
        response = stream_chat(
            f"{base_url}{endpoint}",
            body,
            case_dir / "response.sse",
            timeout,
            api_key,
        )
    else:
        raw, wall = http_post_json(f"{base_url}{endpoint}", body, timeout, api_key)
        (case_dir / "response.raw.json").write_bytes(raw)
        try:
            response = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GateError(f"{name}: response is not JSON: {raw[:500]!r}") from exc
        response["wall_seconds"] = wall
    request_finished = utc_now()
    json_dump(case_dir / "response.json", response)

    after_raw, after = metrics_snapshot(base_url, timeout=30, api_key=api_key)
    text_dump(case_dir / "metrics.after.prom", after_raw)
    delta = metric_delta(before, after)
    json_dump(case_dir / "metrics.delta.json", delta)

    # Give libc enough time to flush the line-oriented server logger; this is
    # observational only and never signals or reconfigures the service.
    time.sleep(0.2)
    next_offset, segment = log_slice(server_log, log_offset)
    text_dump(case_dir / "server.log", segment)
    log_errors = scan_log(segment)

    record = {
        "name": name,
        "started_at": request_started,
        "finished_at": request_finished,
        "endpoint": endpoint,
        "metrics": delta,
        "server_log_errors": log_errors,
        "response_wall_seconds": response.get("wall_seconds"),
        "ttft_seconds": response.get("ttft_seconds"),
    }
    return response, next_offset, record


def terminate_owned_server(server: ManagedServer) -> None:
    process = server.process
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    server.log_file.flush()
    server.log_file.close()


def start_server(config: dict[str, Any], out: pathlib.Path) -> tuple[ManagedServer, str, list[str]]:
    binary = pathlib.Path(str(config["binary"])).expanduser().resolve()
    model = pathlib.Path(str(config["model"])).expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise GateError(f"binary is not an executable file: {binary}")
    if not model.is_file():
        raise GateError(f"model is not a file: {model}")
    mmproj_value = config.get("mmproj")
    mmproj = pathlib.Path(str(mmproj_value)).expanduser().resolve() if mmproj_value else None
    if mmproj is not None and not mmproj.is_file():
        raise GateError(f"mmproj is not a file: {mmproj}")

    host = str(config.get("host", "127.0.0.1"))
    port = int(config.get("port", 18_081))
    assert_loopback(host)
    assert_safe_port(host, port)
    active = gpu_compute_processes()
    if active:
        raise GateError(
            "refusing to launch while CUDA compute clients are active; production is never stopped: "
            + json.dumps(active, ensure_ascii=False)
        )

    args = normalize_server_args(config.get("server_args") or [])
    command = [
        str(binary),
        "--model",
        str(model),
        "--host",
        host,
        "--port",
        str(port),
        "--metrics",
    ]
    if mmproj is not None:
        command.extend(["--mmproj", str(mmproj)])
    command.extend(args)

    child_env = os.environ.copy()
    for key, value in (config.get("env") or {}).items():
        child_env[str(key)] = str(value)
    log_path = out / "server.log"
    log_file = log_path.open("wb")
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=child_env,
            start_new_session=True,
        )
    except Exception:
        log_file.close()
        raise
    url_host = f"[{host}]" if ":" in host else host
    base_url = f"http://{url_host}:{port}"
    return ManagedServer(process, log_file, log_path), base_url, command


def load_config(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot load config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError("config root must be a JSON object")
    for required in ("binary", "model"):
        if not value.get(required):
            raise GateError(f"config is missing required field {required!r}")
    if not isinstance(value.get("server_args", []), list):
        raise GateError("server_args must be a JSON array")
    if not isinstance(value.get("env", {}), dict):
        raise GateError("env must be a JSON object")
    return value


def output_directory(root: pathlib.Path, binary: pathlib.Path) -> pathlib.Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_hash = sha256_file(binary)[:12]
    candidate = root / f"quick-gate-{stamp}-{short_hash}"
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = root / f"quick-gate-{stamp}-{short_hash}-{suffix}"
    candidate.mkdir(parents=True)
    return candidate


def make_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# V100 quick gate",
        "",
        f"- Result: **{'PASS' if summary.get('passed') else 'FAIL'}**",
        f"- Started: `{summary.get('started_at')}`",
        f"- Finished: `{summary.get('finished_at')}`",
        f"- Binary SHA-256: `{summary.get('binary_sha256')}`",
        "",
        "| Case | Result | TTFT | Predicted tok/s | MTP acceptance | Log errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in summary.get("cases", []):
        derived = ((case.get("metrics") or {}).get("derived") or {})
        tps = derived.get("predicted_tokens_per_second")
        acceptance = derived.get("mtp_acceptance_ratio")
        ttft = case.get("ttft_seconds")
        checks = case.get("checks") or {}
        lines.append(
            "| {name} | {result} | {ttft} | {tps} | {acceptance} | {errors} |".format(
                name=case.get("name", "?"),
                result="PASS" if checks.get("passed") else "FAIL",
                ttft="-" if ttft is None else f"{ttft * 1000:.1f} ms",
                tps="-" if tps is None else f"{tps:.3f}",
                acceptance="-" if acceptance is None else f"{acceptance:.3%}",
                errors=sum(len(values) for values in (case.get("server_log_errors") or {}).values()),
            )
        )
    if summary.get("errors"):
        lines.extend(["", "## Gate errors", ""])
        lines.extend(f"- {error}" for error in summary["errors"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True, help="candidate JSON config")
    parser.add_argument("--binary", type=pathlib.Path, help="override config.binary")
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "results",
        help="parent directory for a new immutable result bundle",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="validate config and print the managed command without touching GPU or starting the server",
    )
    return parser.parse_args(argv)


def print_plan(config: dict[str, Any]) -> None:
    external = external_server_config(config)
    if external is not None:
        base_url, log_path = external
        print("External candidate plan (no process management):")
        print(f"  base URL: {base_url}")
        print(f"  followed server log: {log_path}")
        print("Safety: loopback only; port 8000 forbidden; the external process is never signalled.")
        return
    host = str(config.get("host", "127.0.0.1"))
    port = int(config.get("port", 18_081))
    assert_loopback(host)
    if port == PRODUCTION_PORT:
        raise GateError(f"port {PRODUCTION_PORT} is reserved for production and is never allowed")
    args = normalize_server_args(config.get("server_args") or [])
    command = [
        str(pathlib.Path(str(config["binary"])).expanduser()),
        "--model",
        str(pathlib.Path(str(config["model"])).expanduser()),
        "--host",
        host,
        "--port",
        str(port),
        "--metrics",
    ]
    if config.get("mmproj"):
        command.extend(["--mmproj", str(pathlib.Path(str(config["mmproj"])).expanduser())])
    command.extend(args)
    print("Managed candidate command (not executed):")
    print("  " + shlex.join(command))
    print("Child environment additions:")
    print(json.dumps(redacted_mapping(config.get("env") or {}), indent=2, sort_keys=True))
    print("Safety: loopback only; port 8000 forbidden; active CUDA clients cause a hard refusal.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        config_path = args.config.expanduser().resolve()
        config = load_config(config_path)
        if args.binary:
            config["binary"] = str(args.binary.expanduser().resolve())
        if args.print_plan:
            print_plan(config)
            return 0

        binary = pathlib.Path(str(config["binary"])).expanduser().resolve()
        if not binary.is_file():
            raise GateError(f"binary is not a file: {binary}")
        output_root = args.output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        lock_path = output_root / ".quick-gate.lock"
        with lock_path.open("a+b") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise GateError(f"another quick gate holds {lock_path}") from exc

            out = output_directory(output_root, binary)
            started_at = utc_now()
            effective_config = dict(config)
            effective_config["env"] = redacted_mapping(config.get("env") or {})
            for key in list(effective_config):
                effective_config[key] = redact(key, effective_config[key])
            json_dump(out / "config.redacted.json", effective_config)
            summary: dict[str, Any] = {
                "schema_version": 1,
                "passed": False,
                "started_at": started_at,
                "finished_at": None,
                "binary_sha256": sha256_file(binary),
                "cases": [],
                "errors": [],
            }
            server: ManagedServer | None = None
            server_log_path: pathlib.Path | None = None
            try:
                external = external_server_config(config)
                if external is None:
                    server, base_url, command = start_server(config, out)
                    server_log_path = server.log_path
                    process_mode = "managed-direct"
                else:
                    base_url, server_log_path = external
                    if not server_log_path.is_file():
                        raise GateError(f"external server log is not a file: {server_log_path}")
                    command = config.get("candidate_command") or []
                    process_mode = "external-client-only"
                manifest = {
                    "schema_version": 1,
                    "created_at": started_at,
                    "config_path": str(config_path),
                    "binary": str(binary),
                    "binary_sha256": summary["binary_sha256"],
                    "model": str(pathlib.Path(str(config["model"])).expanduser().resolve()),
                    "mmproj": (
                        str(pathlib.Path(str(config["mmproj"])).expanduser().resolve())
                        if config.get("mmproj")
                        else None
                    ),
                    "base_url": base_url,
                    "process_mode": process_mode,
                    "command": command,
                    "env_additions": redacted_mapping(config.get("env") or {}),
                    "host_facts": {
                        "uname": capture_command(["uname", "-a"]),
                        "gpus": capture_command(
                            [
                                "nvidia-smi",
                                "--query-gpu=index,name,uuid,driver_version,memory.total,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw",
                                "--format=csv,noheader",
                            ]
                        ),
                        "topology": capture_command(["nvidia-smi", "topo", "-m"]),
                    },
                    "workloads": {
                        "legacy_exact_prompt_tokens": LEGACY_PROMPT_TOKENS,
                        "legacy_output_tokens": LEGACY_OUTPUT_TOKENS,
                        "chat_output_tokens": int(
                            config.get("chat_max_tokens", DEFAULT_CHAT_OUTPUT_TOKENS)
                        ),
                    },
                }
                json_dump(out / "manifest.json", manifest)
                if server is None:
                    wait_for_external_server(
                        base_url, float(config.get("startup_timeout_seconds", 900))
                    )
                else:
                    wait_for_server(
                        base_url,
                        server.process,
                        float(config.get("startup_timeout_seconds", 900)),
                    )
                    assert_only_owned_gpu_process(server.process)
                log_offset = 0

                legacy_request = {
                    "prompt": [1] * LEGACY_PROMPT_TOKENS,
                    "n_predict": LEGACY_OUTPUT_TOKENS,
                    "temperature": 0,
                    "top_k": 1,
                    "seed": 1234,
                    "cache_prompt": False,
                    "ignore_eos": True,
                    "return_tokens": True,
                    "stream": False,
                }
                legacy, log_offset, legacy_record = run_case(
                    name="01-legacy-71k-exact",
                    out=out,
                    base_url=base_url,
                    body=legacy_request,
                    endpoint="/completion",
                    server_log=server_log_path,
                    log_offset=log_offset,
                    timeout=float(config.get("request_timeout_seconds", 3600)),
                    api_key=None,
                    streaming=False,
                )
                generated = legacy.get("tokens") or []
                legacy_errors: list[str] = []
                if len(generated) != LEGACY_OUTPUT_TOKENS:
                    legacy_errors.append(
                        f"expected {LEGACY_OUTPUT_TOKENS} generated tokens, got {len(generated)}"
                    )
                timings = legacy.get("timings") or {}
                observed_prompt_tokens = legacy.get("tokens_evaluated", timings.get("prompt_n"))
                if observed_prompt_tokens != LEGACY_PROMPT_TOKENS:
                    legacy_errors.append(
                        f"server evaluated {observed_prompt_tokens!r} prompt tokens, "
                        f"expected exactly {LEGACY_PROMPT_TOKENS}"
                    )
                metric_prompt_tokens = (
                    (legacy_record.get("metrics") or {})
                    .get("counters", {})
                    .get("llamacpp:prompt_tokens_total")
                )
                if metric_prompt_tokens != float(LEGACY_PROMPT_TOKENS):
                    legacy_errors.append(
                        f"metrics report {metric_prompt_tokens!r} uncached prompt tokens, "
                        f"expected exactly {LEGACY_PROMPT_TOKENS}"
                    )
                if legacy.get("truncated") is True:
                    legacy_errors.append("legacy 71K request was truncated")
                expected_hash = config.get("expected_legacy_content_sha256")
                content = str(legacy.get("content", ""))
                content_hash = sha256_text(content)
                if expected_hash and content_hash != expected_hash:
                    legacy_errors.append(
                        f"legacy content SHA-256 {content_hash} != expected {expected_hash}"
                    )
                legacy_record["correctness_hashes"] = {
                    "content_sha256": content_hash,
                    "token_ids_le_i32_sha256": sha256_token_ids(generated),
                }
                legacy_record["checks"] = {
                    "passed": not legacy_errors and not legacy_record["server_log_errors"],
                    "errors": legacy_errors,
                    "exact_prompt_tokens": LEGACY_PROMPT_TOKENS,
                    "observed_prompt_tokens": observed_prompt_tokens,
                    "generated_tokens": len(generated),
                }
                summary["cases"].append(legacy_record)

                chat_max = int(config.get("chat_max_tokens", DEFAULT_CHAT_OUTPUT_TOKENS))
                model_alias = str(config.get("model_alias", "qwen3.8-27b-uncensored"))
                first_request = initial_chat_request(model_alias, chat_max)
                first, log_offset, first_record = run_case(
                    name="02-grammar-tool-turn-1",
                    out=out,
                    base_url=base_url,
                    body=first_request,
                    endpoint="/v1/chat/completions",
                    server_log=server_log_path,
                    log_offset=log_offset,
                    timeout=float(config.get("request_timeout_seconds", 3600)),
                    api_key=None,
                    streaming=True,
                )
                first_check = validate_tool_response(first, "grammar-tool")
                first_check["passed"] = bool(
                    first_check["passed"] and not first_record["server_log_errors"]
                )
                first_record["checks"] = first_check
                first_record["correctness_hashes"] = {
                    "semantic_response_sha256": semantic_response_hash(first)
                }
                summary["cases"].append(first_record)

                second_request = continuation_chat_request(first, model_alias, chat_max)
                second, log_offset, second_record = run_case(
                    name="03-continuation-turn-2",
                    out=out,
                    base_url=base_url,
                    body=second_request,
                    endpoint="/v1/chat/completions",
                    server_log=server_log_path,
                    log_offset=log_offset,
                    timeout=float(config.get("request_timeout_seconds", 3600)),
                    api_key=None,
                    streaming=True,
                )
                second_check = validate_tool_response(second, "continuation")
                second_check["passed"] = bool(
                    second_check["passed"] and not second_record["server_log_errors"]
                )
                second_record["checks"] = second_check
                second_record["correctness_hashes"] = {
                    "semantic_response_sha256": semantic_response_hash(second)
                }
                summary["cases"].append(second_record)

                summary["passed"] = bool(
                    all((case.get("checks") or {}).get("passed") for case in summary["cases"])
                    and not summary["errors"]
                )
            except Exception as exc:
                summary["errors"].append(f"{type(exc).__name__}: {exc}")
            finally:
                if server is not None:
                    terminate_owned_server(server)
                if server_log_path is not None and server_log_path.is_file():
                    # Always scan after managed cleanup (or after the last
                    # external request), including error/timeout paths.
                    whole_log = server_log_path.read_text(encoding="utf-8", errors="replace")
                    whole_errors = scan_log(whole_log)
                    json_dump(out / "server-log-errors.json", whole_errors)
                    if whole_errors:
                        summary["passed"] = False
                        summary["errors"].append(
                            f"whole server log contains fatal markers: {whole_errors}"
                        )
                summary["finished_at"] = utc_now()
                json_dump(out / "summary.json", summary)
                text_dump(out / "summary.md", make_summary_markdown(summary))
                print(out)
            return 0 if summary["passed"] else 1
    except GateError as exc:
        print(f"quick-gate: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
