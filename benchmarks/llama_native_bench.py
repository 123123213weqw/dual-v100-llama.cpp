#!/usr/bin/env python3
"""Small dependency-free benchmark client for llama.cpp's /completion API."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run_once(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
        "n_predict": args.n_predict,
        "temperature": args.temperature,
        "seed": args.seed,
        "cache_prompt": args.cache_prompt,
        "stream": False,
    }
    started = time.perf_counter()
    response = post_json(f"{args.url.rstrip('/')}/completion", payload, args.timeout)
    wall_seconds = time.perf_counter() - started
    timings = response.get("timings") or {}
    content = response.get("content", "")
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_seconds": wall_seconds,
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_n": timings.get("predicted_n"),
        "predicted_ms": timings.get("predicted_ms"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "stop": response.get("stop"),
        "stopped_eos": response.get("stopped_eos"),
        "stopped_limit": response.get("stopped_limit"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--n-predict", type=int, default=512)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cache-prompt", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    prompt = args.prompt_file.read_text(encoding="utf-8")
    for _ in range(args.warmup):
        run_once(args, prompt)

    results = []
    for index in range(args.runs):
        result = run_once(args, prompt)
        result["run"] = index + 1
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    rates = [
        float(item["predicted_per_second"])
        for item in results
        if item.get("predicted_per_second") is not None
    ]
    if rates:
        summary = {
            "runs": len(rates),
            "decode_tps_mean": statistics.fmean(rates),
            "decode_tps_min": min(rates),
            "decode_tps_max": max(rates),
            "decode_tps_stdev": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        }
        print("summary=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
