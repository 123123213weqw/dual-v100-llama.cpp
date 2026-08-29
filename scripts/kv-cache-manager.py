#!/usr/bin/env python3
"""Crash-safe last-session KV checkpoint manager for llama-server.

RAM/VRAM reuse remains owned by llama-server. This daemon only writes a cold
checkpoint after a real request followed by a quiet period, and restores the
latest complete checkpoint after a server restart.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8000")
CACHE_DIR = Path(os.environ.get("KV_CACHE_DIR", "/data/wzu/llama-kv"))
QUIET_SECONDS = float(os.environ.get("KV_QUIET_SECONDS", "30"))
POLL_SECONDS = float(os.environ.get("KV_POLL_SECONDS", "1"))
MIN_SAVE_INTERVAL = float(os.environ.get("KV_MIN_SAVE_INTERVAL", "300"))
MIN_TOKEN_DELTA = int(os.environ.get("KV_MIN_TOKEN_DELTA", "8192"))
LATEST = CACHE_DIR / "latest.bin"
PREVIOUS = CACHE_DIR / "previous.bin"
NEXT = CACHE_DIR / "latest.next.bin"
META = CACHE_DIR / "latest.json"


def log(message, **fields):
    payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "message": message, **fields}
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def request(path, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def get_slot():
    slots = request("/slots", timeout=5)
    if not isinstance(slots, list) or not slots:
        raise RuntimeError("llama-server returned no slots")
    return slots[0]


def wait_ready():
    while True:
        try:
            return get_slot()
        except Exception as error:
            log("waiting-for-server", error=str(error))
            time.sleep(2)


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.chmod(temp, 0o600)
    os.replace(temp, path)


def restore_latest():
    if not LATEST.is_file() or LATEST.stat().st_size == 0:
        log("no-checkpoint-to-restore")
        return 0
    try:
        result = request("/slots/0?action=restore", {"filename": LATEST.name}, timeout=600)
        restored = int(result.get("n_restored", 0))
        log("checkpoint-restored", tokens=restored, bytes=int(result.get("n_read", 0)), file=LATEST.name)
        return restored
    except Exception as error:
        log("checkpoint-restore-failed", error=str(error), file=LATEST.name)
        return 0


def save_checkpoint(slot):
    NEXT.unlink(missing_ok=True)
    # Pre-create as the unprivileged manager. llama-server runs as root inside
    # its container and truncates this inode, preserving safe ownership/mode.
    NEXT.touch(mode=0o600, exist_ok=False)
    result = request("/slots/0?action=save", {"filename": NEXT.name}, timeout=1800)
    if not NEXT.is_file() or NEXT.stat().st_size == 0:
        raise RuntimeError("save endpoint succeeded without producing a checkpoint")
    if LATEST.exists():
        os.replace(LATEST, PREVIOUS)
    os.replace(NEXT, LATEST)
    os.chmod(LATEST, 0o600)
    metadata = {
        "savedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tokens": int(result.get("n_saved", slot.get("n_prompt_tokens", 0))),
        "bytes": LATEST.stat().st_size,
        "slot": int(slot.get("id", 0)),
    }
    atomic_json(META, metadata)
    log("checkpoint-saved", **metadata)
    return metadata["tokens"]


def metadata_tokens():
    try:
        return int(json.loads(META.read_text()).get("tokens", 0))
    except Exception:
        return 0


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(CACHE_DIR, 0o700)
    NEXT.unlink(missing_ok=True)
    slot = wait_ready()
    if bool(slot.get("is_processing")) or int(slot.get("n_prompt_tokens", 0) or 0) > 0:
        last_saved_tokens = metadata_tokens()
        log("live-slot-preserved", tokens=int(slot.get("n_prompt_tokens", 0) or 0))
    else:
        last_saved_tokens = restore_latest()
    last_save_time = time.monotonic()
    dirty = False
    idle_since = None
    last_task = None
    server_was_down = False
    while True:
        try:
            slot = get_slot()
            if server_was_down:
                if bool(slot.get("is_processing")) or int(slot.get("n_prompt_tokens", 0) or 0) > 0:
                    last_saved_tokens = metadata_tokens()
                    log("restart-restore-skipped-live-slot", tokens=int(slot.get("n_prompt_tokens", 0) or 0))
                else:
                    last_saved_tokens = restore_latest()
                last_save_time = time.monotonic()
                dirty = False
                idle_since = None
                server_was_down = False
                slot = get_slot()
            processing = bool(slot.get("is_processing"))
            if processing:
                dirty = True
                idle_since = None
                last_task = slot.get("id_task")
            else:
                if idle_since is None:
                    idle_since = time.monotonic()
                tokens = int(slot.get("n_prompt_tokens", 0) or 0)
                # A request can finish between two polls. Token-count drift is a
                # second dirty signal, so short calls are checkpointed too.
                if tokens > 0 and tokens != last_saved_tokens:
                    dirty = True
                token_delta = abs(tokens - last_saved_tokens)
                interval_elapsed = time.monotonic() - last_save_time >= MIN_SAVE_INTERVAL
                should_persist = last_saved_tokens == 0 or token_delta >= MIN_TOKEN_DELTA or interval_elapsed
                if dirty and tokens > 0 and should_persist and time.monotonic() - idle_since >= QUIET_SECONDS:
                    last_saved_tokens = save_checkpoint(slot)
                    last_save_time = time.monotonic()
                    dirty = False
                    idle_since = None
                    log("checkpoint-cycle-complete", task=last_task, tokens=last_saved_tokens)
            time.sleep(POLL_SECONDS)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            server_was_down = True
            log("server-unavailable", error=str(error))
            time.sleep(2)
        except Exception as error:
            log("checkpoint-cycle-failed", error=str(error))
            NEXT.unlink(missing_ok=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
