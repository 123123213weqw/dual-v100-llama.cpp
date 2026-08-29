#!/usr/bin/env python3
"""Maintenance-window A/B orchestrator for the WZU dual-V100 service.

The default mode is a local, offline dry-run: it validates the JSON plan and
prints every state transition, but makes no SSH connection and runs no model.
Actual execution requires BOTH ``--execute`` and the configured confirmation
phrase.  In execute mode the orchestrator runs locally and coordinates the WZU
host plus the Tokyo Caddy ingress through SSH.

Safety invariants:

* persistent flock holders serialize maintenance on both hosts;
* Caddy is gracefully reloaded to a 503 maintenance route before draining;
* the production slot must remain idle for N consecutive polls;
* the production container is stopped, never removed or recreated;
* candidates run only in short-lived CUDA containers on a non-production port;
* quick_gate.py runs client-only on WZU and cannot signal the candidate;
* all candidate containers are name/run-id scoped;
* production is restored and healthy before ingress is restored;
* exact original container ID/config and Caddy bytes are verified on restore;
* a Tokyo watchdog restores the original Caddyfile if the orchestrator dies.

This file intentionally contains no benchmark defaults beyond invoking the
separate quick gate.  It must not be used as a concurrent-serving benchmark.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import pathlib
import re
import select
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
QUICK_GATE = SCRIPT_DIR / "quick_gate.py"
PRODUCTION_PORT = 8000
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class MaintenanceError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def json_dump(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def shell_join(argv: list[str]) -> str:
    return shlex.join([str(value) for value in argv])


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MaintenanceError(f"{name} must be a JSON object")
    return value


def require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise MaintenanceError(f"{name} must be a JSON array")
    return value


def load_config(path: pathlib.Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaintenanceError(f"cannot read {path}: {exc}") from exc
    config = require_mapping(config, "config")
    for key in ("ssh", "production", "gateway", "candidate_runtime", "drain", "variants"):
        if key not in config:
            raise MaintenanceError(f"config is missing {key!r}")
    require_mapping(config["ssh"], "ssh")
    production = require_mapping(config["production"], "production")
    gateway = require_mapping(config["gateway"], "gateway")
    runtime = require_mapping(config["candidate_runtime"], "candidate_runtime")
    drain = require_mapping(config["drain"], "drain")
    variants = require_list(config["variants"], "variants")
    if len(variants) != 2:
        raise MaintenanceError("exactly two variants (A then B) are required")
    names: set[str] = set()
    for index, raw in enumerate(variants):
        variant = require_mapping(raw, f"variants[{index}]")
        name = str(variant.get("name", ""))
        if not NAME_RE.fullmatch(name):
            raise MaintenanceError(f"invalid variant name {name!r}")
        if name in names:
            raise MaintenanceError(f"duplicate variant name {name!r}")
        names.add(name)
        if not str(variant.get("release_bin", "")).startswith("/"):
            raise MaintenanceError(f"{name}: release_bin must be absolute")
    container = str(production.get("container", ""))
    if not NAME_RE.fullmatch(container):
        raise MaintenanceError("production.container is invalid")
    host_port = int(runtime.get("host_port", 0))
    if host_port in {0, PRODUCTION_PORT} or not 1024 <= host_port <= 65535:
        raise MaintenanceError("candidate_runtime.host_port must be a non-production high port")
    if not str(runtime.get("image_id", "")).startswith("sha256:"):
        raise MaintenanceError("candidate_runtime.image_id must be an immutable sha256 image ID")
    if int(drain.get("stable_polls", 0)) < 2:
        raise MaintenanceError("drain.stable_polls must be at least 2")
    for key in ("begin_marker", "end_marker", "caddyfile", "site_address", "public_health_url"):
        if not gateway.get(key):
            raise MaintenanceError(f"gateway.{key} is required")
    if not QUICK_GATE.is_file():
        raise MaintenanceError(f"quick gate is missing: {QUICK_GATE}")
    return config


@dataclasses.dataclass(frozen=True)
class Remote:
    host: str
    proxy_jump: str | None = None

    def ssh_argv(self) -> list[str]:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
        if self.proxy_jump:
            argv.extend(["-J", self.proxy_jump])
        argv.append(self.host)
        return argv

    def run(
        self,
        script: str,
        *,
        input_bytes: bytes | None = None,
        timeout: float | None = 60,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        result = subprocess.run(
            [*self.ssh_argv(), "sh", "-s"],
            input=script.encode("utf-8") if input_bytes is None else script.encode("utf-8") + input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if check and result.returncode != 0:
            raise MaintenanceError(
                f"{self.host}: remote command failed ({result.returncode}):\n"
                + result.stderr.decode("utf-8", errors="replace")[-8000:]
            )
        return result

    def command(
        self,
        command: str,
        *,
        input_bytes: bytes | None = None,
        timeout: float | None = 60,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        result = subprocess.run(
            [*self.ssh_argv(), command],
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if check and result.returncode != 0:
            raise MaintenanceError(
                f"{self.host}: {command!r} failed ({result.returncode}):\n"
                + result.stderr.decode("utf-8", errors="replace")[-8000:]
            )
        return result

    def text(self, command: str, *, timeout: float | None = 60) -> str:
        return self.command(command, timeout=timeout).stdout.decode("utf-8", errors="replace")

    def upload(self, path: str, content: bytes, mode: str = "0600") -> None:
        if not path.startswith("/"):
            raise MaintenanceError(f"remote upload path must be absolute: {path}")
        encoded = base64.b64encode(content)
        command = (
            f"umask 077; base64 -d > {shlex.quote(path)}; "
            f"chmod {shlex.quote(mode)} {shlex.quote(path)}"
        )
        self.command(command, input_bytes=encoded)


class RemoteLock:
    def __init__(self, remote: Remote, path: str) -> None:
        self.remote = remote
        self.path = path
        self.process: subprocess.Popen[str] | None = None

    def acquire(self, timeout: float = 15) -> None:
        command = (
            f"flock -n {shlex.quote(self.path)} sh -c "
            + shlex.quote("printf '__MAINTENANCE_LOCKED__\\n'; cat >/dev/null")
        )
        self.process = subprocess.Popen(
            [*self.remote.ssh_argv(), command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert self.process.stdout is not None
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready or self.process.stdout.readline().strip() != "__MAINTENANCE_LOCKED__":
            stderr = ""
            if self.process.poll() is not None and self.process.stderr is not None:
                stderr = self.process.stderr.read()
            self.release()
            raise MaintenanceError(f"cannot acquire {self.remote.host}:{self.path}: {stderr}")

    def release(self) -> None:
        if self.process is None:
            return
        if self.process.stdin is not None:
            with contextlib.suppress(BrokenPipeError):
                self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=5)
        self.process = None


def caddy_maintenance(original: bytes, gateway: dict[str, Any], run_id: str) -> bytes:
    text = original.decode("utf-8")
    begin = str(gateway["begin_marker"])
    end = str(gateway["end_marker"])
    if text.count(begin) != 1 or text.count(end) != 1:
        raise MaintenanceError("Caddy markers are not unique")
    start = text.index(begin)
    finish = text.index(end, start) + len(end)
    site = str(gateway["site_address"])
    block = f'''{begin}
{site} {{
\theader Retry-After "300"
\theader Cache-Control "no-store"
\trespond "Qwen maintenance {run_id}" 503
}}
{end}'''
    return (text[:start] + block + text[finish:]).encode("utf-8")


def remote_json(remote: Remote, command: str, timeout: float = 60) -> Any:
    raw = remote.command(command, timeout=timeout).stdout
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MaintenanceError(
            f"{remote.host}: expected JSON from {command!r}, got {raw[:1000]!r}"
        ) from exc


def inspect_container(remote: Remote, name: str) -> dict[str, Any]:
    value = remote_json(remote, f"docker inspect {shlex.quote(name)}")
    if not isinstance(value, list) or len(value) != 1:
        raise MaintenanceError(f"expected exactly one container named {name}")
    return value[0]


def config_fingerprint(inspect: dict[str, Any]) -> str:
    selected = {
        "Id": inspect.get("Id"),
        "Path": inspect.get("Path"),
        "Args": inspect.get("Args"),
        "Config": {
            key: (inspect.get("Config") or {}).get(key)
            for key in ("Image", "Entrypoint", "Cmd", "Env", "Healthcheck", "Labels")
        },
        "HostConfig": {
            key: (inspect.get("HostConfig") or {}).get(key)
            for key in (
                "Binds",
                "NetworkMode",
                "PortBindings",
                "RestartPolicy",
                "AutoRemove",
                "Runtime",
                "DeviceRequests",
                "IpcMode",
                "CapAdd",
            )
        },
        # HostConfig.Binds already pins the byte-for-byte mount configuration.
        # Docker may reorder the derived runtime `Mounts` array on start, so
        # including it makes an unchanged container fail the restore check.
    }
    canonical = json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def gpu_processes(remote: Remote) -> list[dict[str, str]]:
    command = (
        "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory "
        "--format=csv,noheader,nounits"
    )
    lines = remote.text(command).splitlines()
    result: list[dict[str, str]] = []
    for line in lines:
        fields = [value.strip() for value in line.split(",", 3)]
        if len(fields) == 4 and fields[1] not in {"", "N/A", "[N/A]"}:
            result.append(
                {"gpu_uuid": fields[0], "pid": fields[1], "process_name": fields[2], "used_mib": fields[3]}
            )
    return result


def assert_production_gpu_owner(remote: Remote, inspect: dict[str, Any]) -> None:
    pid = str((inspect.get("State") or {}).get("Pid", 0))
    active = gpu_processes(remote)
    if not active or any(item["pid"] != pid for item in active):
        raise MaintenanceError(f"GPU ownership is not exclusively production PID {pid}: {active}")


def wait_http(remote: Remote, url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = "not attempted"
    while time.monotonic() < deadline:
        result = remote.command(
            f"curl -fsS --max-time 5 {shlex.quote(url)}",
            timeout=15,
            check=False,
        )
        if result.returncode == 0:
            try:
                if json.loads(result.stdout).get("status") == "ok":
                    return
            except (json.JSONDecodeError, AttributeError):
                pass
        last = (result.stderr or result.stdout).decode("utf-8", errors="replace")[-1000:]
        time.sleep(2)
    raise MaintenanceError(f"health timeout for {remote.host}:{url}: {last}")


def wait_docker_healthy(remote: Remote, name: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = "unknown"
    while time.monotonic() < deadline:
        result = remote.command(
            f"docker inspect -f '{{{{.State.Running}}}} {{{{if .State.Health}}}}{{{{.State.Health.Status}}}}{{{{end}}}}' {shlex.quote(name)}",
            check=False,
        )
        last = result.stdout.decode("utf-8", errors="replace").strip()
        if result.returncode == 0 and last == "true healthy":
            return
        if last.startswith("false"):
            logs = remote.text(f"docker logs --tail 200 {shlex.quote(name)} 2>&1")
            raise MaintenanceError(f"{name} exited while loading:\n{logs}")
        time.sleep(2)
    raise MaintenanceError(f"container {name} health timeout; last={last!r}")


def wait_public_status(url: str, expected: int, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    last: int | str = "not attempted"
    while time.monotonic() < deadline:
        request = urllib.request.Request(url, method="GET", headers={"User-Agent": "v100-maintenance-ab/1"})
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                last = response.status
        except urllib.error.HTTPError as exc:
            last = exc.code
        except (urllib.error.URLError, TimeoutError) as exc:
            last = str(exc)
        if last == expected:
            return
        time.sleep(1)
    raise MaintenanceError(f"public gateway {url} returned {last!r}, expected {expected}")


def drain_slots(remote: Remote, production: dict[str, Any], drain: dict[str, Any]) -> list[dict[str, Any]]:
    url = str(production["base_url"]).rstrip("/") + "/slots"
    stable_required = int(drain["stable_polls"])
    interval = float(drain["poll_interval_seconds"])
    deadline = time.monotonic() + float(drain["timeout_seconds"])
    stable = 0
    latest: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        payload = remote_json(remote, f"curl -fsS --max-time 10 {shlex.quote(url)}")
        if not isinstance(payload, list) or not payload:
            raise MaintenanceError(f"/slots returned an invalid payload: {payload!r}")
        latest = payload
        if all(not bool(slot.get("is_processing")) for slot in payload):
            stable += 1
            if stable >= stable_required:
                return latest
        else:
            stable = 0
        time.sleep(interval)
    raise MaintenanceError(f"production did not drain before timeout; last slots={latest!r}")


def release_manifest_hash(remote: Remote, directory: str) -> str:
    command = (
        f"cd {shlex.quote(directory)} && "
        "find . -maxdepth 1 -type f -printf '%P\\0' | sort -z | "
        "xargs -0 sha256sum | sha256sum | awk '{print $1}'"
    )
    return remote.text(command).strip()


def docker_run_command(config: dict[str, Any], variant: dict[str, Any], name: str) -> list[str]:
    runtime = config["candidate_runtime"]
    argv = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--restart=no",
        "--gpus",
        "all",
        "--ipc=host",
        "--cap-add",
        "SYS_NICE",
        "--label",
        f"com.wzu.qwen.ab-run={name}",
        "-p",
        f"127.0.0.1:{int(runtime['host_port'])}:{int(runtime.get('container_port', 8000))}",
        "--health-cmd",
        'timeout 5 bash -c "</dev/tcp/127.0.0.1/8000"',
        "--health-interval",
        "10s",
        "--health-timeout",
        "8s",
        "--health-start-period",
        "180s",
        "--health-retries",
        "3",
    ]
    environment = dict(runtime.get("env", {}))
    environment.update(variant.get("env", {}))
    for key, value in environment.items():
        argv.extend(["-e", f"{key}={value}"])
    argv.extend(["-v", f"{variant['release_bin']}:/opt/llama/bin:ro"])
    for mount in runtime["model_mounts"]:
        argv.extend(["-v", str(mount)])
    argv.extend(
        [
            "--entrypoint",
            "/opt/llama/bin/llama-server",
            str(runtime["image_id"]),
            "-m",
            str(runtime["model"]),
            "--alias",
            str(runtime["model_alias"]),
            "--host",
            "0.0.0.0",
            "--port",
            str(runtime.get("container_port", 8000)),
            "--metrics",
        ]
    )
    argv.extend(str(value) for value in runtime["server_args"])
    return argv


def render_plan(config: dict[str, Any]) -> str:
    production = config["production"]
    gateway = config["gateway"]
    runtime = config["candidate_runtime"]
    lines = [
        "DRY RUN ONLY — no SSH connection, container, or model request was made.",
        "",
        "1. Acquire persistent flock holders:",
        f"   - WZU: {config['ssh']['wzu']}:{config['locks']['wzu']}",
        f"   - Tokyo: {config['ssh']['tokyo']}:{config['locks']['tokyo']}",
        "2. Snapshot exact Docker inspect/config hash, Caddy bytes/hash, slots, GPU owners, disk and topology.",
        f"3. Gracefully reload {gateway['caddyfile']} to an all-request HTTP 503 maintenance route.",
        "   A timed Tokyo watchdog restores the byte-exact original if this process dies.",
        f"4. Require {config['drain']['stable_polls']} consecutive idle /slots polls, then stop (never remove) {production['container']}.",
        "5. Require zero CUDA compute PIDs and launch A/B sequentially in disposable CUDA containers:",
    ]
    for index, variant in enumerate(config["variants"]):
        name = f"qwen38-ab-<run>-{'a' if index == 0 else 'b'}"
        lines.append(f"   [{index + 1}] {variant['name']} manifest={variant.get('expected_manifest_sha256') or '<must fill for execute>'}")
        lines.append("       " + shell_join(docker_run_command(config, variant, name)))
        lines.append(
            f"       quick_gate.py client-only -> http://127.0.0.1:{runtime['host_port']} (logs from docker logs -f)"
        )
    lines.extend(
        [
            f"6. Start the SAME container ID {production['container']}; wait Docker healthy + /health + exclusive GPU ownership.",
            "7. Verify the production Docker config fingerprint is unchanged.",
            "8. Restore byte-exact Caddyfile, validate/reload it, verify public unauthenticated status 401 and tunnel health 200.",
            "9. Cancel watchdog, remove owned temporary containers/files, release both locks.",
            "",
            "No call to /home/wzu/bin/qwen-model is permitted: that script removes/recreates production and has stale rollback labels.",
        ]
    )
    return "\n".join(lines) + "\n"


class Orchestrator:
    def __init__(self, config: dict[str, Any], output: pathlib.Path, run_id: str) -> None:
        self.config = config
        self.output = output
        self.run_id = run_id
        ssh = config["ssh"]
        self.wzu = Remote(str(ssh["wzu"]))
        self.tokyo = Remote(str(ssh["tokyo"]), str(ssh.get("tokyo_proxy_jump") or ssh["wzu"]))
        self.locks: list[RemoteLock] = []
        self.original_inspect: dict[str, Any] | None = None
        self.original_fingerprint: str | None = None
        self.original_caddy: bytes | None = None
        self.original_caddy_sha: str | None = None
        self.maintenance_caddy_sha: str | None = None
        self.gateway_paths: dict[str, str] = {}
        self.watchdog_pid: str | None = None
        self.production_watchdog_pid: str | None = None
        self.production_watchdog_paths: list[str] = []
        self.gateway_gated = False
        self.production_stopped = False
        self.owned_candidates: list[str] = []
        self.remote_run_root = f"{config['remote_work_root'].rstrip('/')}/{run_id}"

    def record(self, event: str, **fields: Any) -> None:
        path = self.output / "events.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": utc_now(), "event": event, **fields}, ensure_ascii=False) + "\n")

    def acquire_locks(self) -> None:
        for remote, path in (
            (self.tokyo, str(self.config["locks"]["tokyo"])),
            (self.wzu, str(self.config["locks"]["wzu"])),
        ):
            lock = RemoteLock(remote, path)
            lock.acquire()
            self.locks.append(lock)
            self.record("lock-acquired", host=remote.host, path=path)

    def snapshot(self) -> None:
        production = self.config["production"]
        name = str(production["container"])
        caddy_states = self.tokyo.text("systemctl is-active caddy haproxy").splitlines()
        tunnel_state = self.wzu.text("systemctl is-active qwen-tokyo-tunnel.service").strip()
        if caddy_states != ["active", "active"] or tunnel_state != "active":
            raise MaintenanceError(
                f"ingress is not in the audited active state: Caddy/HAProxy={caddy_states}, "
                f"tunnel={tunnel_state!r}"
            )
        inspect = inspect_container(self.wzu, name)
        state = inspect.get("State") or {}
        if not state.get("Running") or (state.get("Health") or {}).get("Status") != "healthy":
            raise MaintenanceError(f"production must begin running+healthy, got {state}")
        if int((inspect.get("HostConfig") or {}).get("PortBindings", {}).get("8000/tcp", [{}])[0].get("HostPort", 0)) != PRODUCTION_PORT:
            raise MaintenanceError("production port binding is not the audited 127.0.0.1:8000")
        self.original_inspect = inspect
        self.original_fingerprint = config_fingerprint(inspect)
        assert_production_gpu_owner(self.wzu, inspect)
        runtime = self.config["candidate_runtime"]
        image_id = str(runtime["image_id"])
        actual_image = self.wzu.text(
            f"docker image inspect -f '{{{{.Id}}}}' {shlex.quote(image_id)}"
        ).strip()
        if actual_image != image_id:
            raise MaintenanceError(f"candidate image ID mismatch: {actual_image!r} != {image_id!r}")
        file_check = self.wzu.command(
            " && ".join(
                [
                    f"test -s {shlex.quote(str(runtime['host_model']))}",
                    f"test -s {shlex.quote(str(runtime['host_mmproj']))}",
                    "command -v python3 >/dev/null",
                    "command -v docker >/dev/null",
                ]
            ),
            check=False,
        )
        if file_check.returncode != 0:
            raise MaintenanceError("candidate model/mmproj, Python, or Docker preflight failed")
        free_bytes = int(self.wzu.text("df -B1 --output=avail / | tail -1").strip())
        minimum_free = int(runtime.get("minimum_free_bytes", 50 * 1024**3))
        if free_bytes < minimum_free:
            raise MaintenanceError(f"free disk {free_bytes} is below required {minimum_free}")
        gpu_names = [
            line.strip()
            for line in self.wzu.text(
                "nvidia-smi --query-gpu=name --format=csv,noheader"
            ).splitlines()
            if line.strip()
        ]
        if len(gpu_names) != 2 or any("V100" not in name for name in gpu_names):
            raise MaintenanceError(f"expected exactly two V100 GPUs, got {gpu_names}")
        host_port = int(runtime["host_port"])
        if self.wzu.command(
            f"ss -ltnH 'sport = :{host_port}' | grep -q .", check=False
        ).returncode == 0:
            raise MaintenanceError(f"candidate port {host_port} is already occupied")
        variant_manifests: dict[str, str] = {}
        for variant in self.config["variants"]:
            variant_manifests[str(variant["name"])] = self.preflight_variant(variant)
        slots = remote_json(self.wzu, "curl -fsS --max-time 10 http://127.0.0.1:8000/slots")
        gpu = gpu_processes(self.wzu)
        disk = self.wzu.text("df -hT /; nvidia-smi topo -m")
        caddy_path = str(self.config["gateway"]["caddyfile"])
        self.original_caddy = self.tokyo.command(f"cat {shlex.quote(caddy_path)}").stdout
        self.original_caddy_sha = sha256_bytes(self.original_caddy)
        snapshot = {
            "at": utc_now(),
            "production_id": inspect.get("Id"),
            "production_fingerprint": self.original_fingerprint,
            "production_state": state,
            "slots": slots,
            "gpu_processes": gpu,
            "gpu_names": gpu_names,
            "free_bytes": free_bytes,
            "variant_manifests": variant_manifests,
            "gateway_services": {"caddy": caddy_states[0], "haproxy": caddy_states[1], "tunnel": tunnel_state},
            "disk_and_topology": disk,
            "caddy_sha256": self.original_caddy_sha,
        }
        json_dump(self.output / "preflight.json", snapshot)
        self.record("snapshot-complete", production_id=inspect.get("Id"))

    def gate_ingress(self) -> None:
        assert self.original_caddy is not None and self.original_caddy_sha is not None
        gateway = self.config["gateway"]
        caddy_path = str(gateway["caddyfile"])
        new = caddy_maintenance(self.original_caddy, gateway, self.run_id)
        maintenance_sha = sha256_bytes(new)
        prefix = f"/tmp/qwen-ab-{self.run_id}"
        paths = {
            "new": prefix + ".Caddyfile",
            "backup": prefix + ".Caddyfile.orig",
            "watchdog": prefix + ".rollback.sh",
            "watchdog_log": prefix + ".rollback.log",
        }
        self.gateway_paths = paths
        self.maintenance_caddy_sha = maintenance_sha
        self.tokyo.upload(paths["new"], new, "0600")
        env_file = str(gateway["environment_file"])
        caddy_bin = str(gateway.get("caddy_binary", "/usr/local/bin/caddy"))
        prepare_script = f"""set -eu
current=$(sha256sum {shlex.quote(caddy_path)} | awk '{{print $1}}')
[ "$current" = {shlex.quote(self.original_caddy_sha)} ]
test ! -e {shlex.quote(paths['backup'])}
cp -a {shlex.quote(caddy_path)} {shlex.quote(paths['backup'])}
set -a; . {shlex.quote(env_file)}; set +a
{shlex.quote(caddy_bin)} validate --config {shlex.quote(paths['new'])}
"""
        self.tokyo.command("sh -s", input_bytes=prepare_script.encode("utf-8"))

        rollback_seconds = int(gateway["rollback_seconds"])
        watchdog = f"""#!/bin/sh
sleep {rollback_seconds}
while :; do
  current=$(sha256sum {shlex.quote(caddy_path)} | awk '{{print $1}}')
  [ "$current" = {shlex.quote(maintenance_sha)} ] || exit 0
  if curl -fsS --max-time 5 http://127.0.0.1:18000/health >/dev/null 2>&1; then
    cp -p {shlex.quote(paths['backup'])} {shlex.quote(caddy_path)}
    set -a; . {shlex.quote(env_file)}; set +a
    {shlex.quote(caddy_bin)} validate --config {shlex.quote(caddy_path)} && systemctl reload caddy
    exit $?
  fi
  sleep 30
done
""".encode("utf-8")
        self.tokyo.upload(paths["watchdog"], watchdog, "0700")
        command = (
            f"nohup {shlex.quote(paths['watchdog'])} >{shlex.quote(paths['watchdog_log'])} 2>&1 "
            "<&- & echo $!"
        )
        self.watchdog_pid = self.tokyo.text(command).strip()
        install_script = f"""set -eu
current=$(sha256sum {shlex.quote(caddy_path)} | awk '{{print $1}}')
[ "$current" = {shlex.quote(self.original_caddy_sha)} ]
install -m 0644 {shlex.quote(paths['new'])} {shlex.quote(caddy_path)}
systemctl reload caddy
"""
        try:
            self.tokyo.command("sh -s", input_bytes=install_script.encode("utf-8"))
            self.gateway_gated = True
        except Exception:
            current = self.tokyo.text(
                f"sha256sum {shlex.quote(caddy_path)} | awk '{{print $1}}'"
            ).strip()
            self.gateway_gated = current == maintenance_sha
            if not self.gateway_gated:
                if self.watchdog_pid:
                    self.tokyo.command(
                        f"kill {shlex.quote(self.watchdog_pid)} 2>/dev/null || true",
                        check=False,
                    )
                self.tokyo.command(
                    "rm -f " + " ".join(shlex.quote(value) for value in paths.values()),
                    check=False,
                )
            raise
        wait_public_status(str(gateway["public_health_url"]), 503)
        self.record("ingress-maintenance", maintenance_sha256=maintenance_sha, watchdog_pid=self.watchdog_pid)

    def drain_and_stop_production(self) -> None:
        slots = drain_slots(self.wzu, self.config["production"], self.config["drain"])
        self.record("drained", slots=slots)
        name = str(self.config["production"]["container"])
        rollback_seconds = int(self.config["gateway"]["rollback_seconds"])
        prefix = f"/tmp/qwen-ab-{self.run_id}.production-rollback"
        script_path = prefix + ".sh"
        log_path = prefix + ".log"
        candidate_a = f"qwen38-ab-{self.run_id}-a"
        candidate_b = f"qwen38-ab-{self.run_id}-b"
        watchdog = f"""#!/bin/sh
sleep {rollback_seconds}
docker rm -f {shlex.quote(candidate_a)} {shlex.quote(candidate_b)} >/dev/null 2>&1 || true
running=$(docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(name)} 2>/dev/null || true)
[ "$running" = true ] || docker start {shlex.quote(name)} >/dev/null
for _ in $(seq 1 450); do
  curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && exit 0
  sleep 2
done
exit 1
""".encode("utf-8")
        self.wzu.upload(script_path, watchdog, "0700")
        self.production_watchdog_paths = [script_path, log_path]
        self.production_watchdog_pid = self.wzu.text(
            f"nohup {shlex.quote(script_path)} >{shlex.quote(log_path)} 2>&1 < /dev/null & echo $!"
        ).strip()
        self.record("production-watchdog-armed", pid=self.production_watchdog_pid)
        timeout = int(self.config["production"].get("stop_timeout_seconds", 120))
        self.wzu.command(f"docker stop --time {timeout} {shlex.quote(name)}")
        self.production_stopped = True
        state = self.wzu.text(f"docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(name)}").strip()
        if state != "false":
            raise MaintenanceError(f"production failed to stop: Running={state!r}")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and gpu_processes(self.wzu):
            time.sleep(1)
        active = gpu_processes(self.wzu)
        if active:
            raise MaintenanceError(f"CUDA processes remain after production stop: {active}")
        self.record("production-stopped", container=name)

    def preflight_variant(self, variant: dict[str, Any]) -> str:
        directory = str(variant["release_bin"])
        check = self.wzu.command(
            f"test -x {shlex.quote(directory + '/llama-server')} && "
            f"test -f {shlex.quote(directory + '/libggml-cuda.so.0.20.0')}",
            check=False,
        )
        if check.returncode != 0:
            raise MaintenanceError(f"variant {variant['name']} release is incomplete: {directory}")
        actual = release_manifest_hash(self.wzu, directory)
        expected = variant.get("expected_manifest_sha256")
        if not expected:
            raise MaintenanceError(f"variant {variant['name']} needs expected_manifest_sha256")
        if actual != expected:
            raise MaintenanceError(
                f"variant {variant['name']} manifest {actual} != pinned {expected}"
            )
        return actual

    def run_variant(self, index: int, variant: dict[str, Any]) -> dict[str, Any]:
        manifest_hash = self.preflight_variant(variant)
        suffix = "a" if index == 0 else "b"
        container = f"qwen38-ab-{self.run_id}-{suffix}"
        if not NAME_RE.fullmatch(container):
            raise MaintenanceError(f"generated container name is invalid: {container}")
        self.owned_candidates.append(container)
        exists = self.wzu.command(f"docker inspect {shlex.quote(container)} >/dev/null 2>&1", check=False)
        if exists.returncode == 0:
            raise MaintenanceError(f"candidate container already exists: {container}")
        host_port = int(self.config["candidate_runtime"]["host_port"])
        port_busy = self.wzu.command(f"ss -ltnH 'sport = :{host_port}' | grep -q .", check=False)
        if port_busy.returncode == 0:
            raise MaintenanceError(f"candidate host port {host_port} is occupied")
        if gpu_processes(self.wzu):
            raise MaintenanceError("GPU became busy before candidate launch")
        command = docker_run_command(self.config, variant, container)
        self.record("candidate-command", variant=variant["name"], argv=command)
        self.wzu.command(shell_join(command), timeout=120)
        wait_docker_healthy(
            self.wzu,
            container,
            float(self.config["candidate_runtime"].get("startup_timeout_seconds", 900)),
        )
        wait_http(self.wzu, f"http://127.0.0.1:{host_port}/health", 30)
        inspect = inspect_container(self.wzu, container)
        assert_production_gpu_owner(self.wzu, inspect)

        remote_case = f"{self.remote_run_root}/{suffix}"
        remote_quick_gate = f"{remote_case}/quick_gate.py"
        remote_config = f"{remote_case}/quick-gate.json"
        remote_log = f"{remote_case}/candidate.log"
        self.wzu.command(f"mkdir -p {shlex.quote(remote_case)}")
        self.wzu.upload(remote_quick_gate, QUICK_GATE.read_bytes(), "0700")
        log_pid = self.wzu.text(
            f"nohup docker logs --follow --timestamps {shlex.quote(container)} "
            f">{shlex.quote(remote_log)} 2>&1 < /dev/null & echo $!"
        ).strip()
        qconfig = {
            "binary": str(variant["release_bin"]) + "/llama-server",
            "model": str(self.config["candidate_runtime"]["host_model"]),
            "mmproj": str(self.config["candidate_runtime"]["host_mmproj"]),
            "model_alias": str(self.config["candidate_runtime"]["model_alias"]),
            "startup_timeout_seconds": 60,
            "request_timeout_seconds": int(
                self.config["candidate_runtime"].get("request_timeout_seconds", 3600)
            ),
            "chat_max_tokens": int(self.config["candidate_runtime"].get("chat_max_tokens", 1024)),
            "expected_legacy_content_sha256": variant.get("expected_legacy_content_sha256"),
            "external_server": {
                "base_url": f"http://127.0.0.1:{host_port}",
                "server_log": remote_log,
            },
            "candidate_command": command,
        }
        self.wzu.upload(
            remote_config,
            (json.dumps(qconfig, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
            "0600",
        )
        result = self.wzu.command(
            f"python3 {shlex.quote(remote_quick_gate)} --config {shlex.quote(remote_config)} "
            f"--output-root {shlex.quote(remote_case + '/results')}",
            timeout=float(self.config["candidate_runtime"].get("gate_timeout_seconds", 7200)),
            check=False,
        )
        self.wzu.command(f"kill {shlex.quote(log_pid)} 2>/dev/null || true", check=False)
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        (self.output / f"{suffix}-quick-gate.stdout").write_text(stdout, encoding="utf-8")
        (self.output / f"{suffix}-quick-gate.stderr").write_text(stderr, encoding="utf-8")
        if result.returncode not in {0, 1}:
            raise MaintenanceError(
                f"quick gate infrastructure failed for {variant['name']} ({result.returncode}): {stderr[-4000:]}"
            )
        result_path = next((line.strip() for line in reversed(stdout.splitlines()) if line.strip()), "")
        if not result_path.startswith(remote_case + "/results/"):
            raise MaintenanceError(f"cannot identify quick-gate result path from: {stdout[-2000:]}")
        summary = remote_json(self.wzu, f"cat {shlex.quote(result_path + '/summary.json')}")
        summary["variant"] = variant["name"]
        summary["release_manifest_sha256"] = manifest_hash
        summary["remote_result_path"] = result_path
        json_dump(self.output / f"{suffix}-summary.json", summary)
        self.record("candidate-finished", variant=variant["name"], returncode=result.returncode)

        self.wzu.command(f"docker stop --time 30 {shlex.quote(container)} >/dev/null", check=False)
        self.wzu.command(f"docker rm {shlex.quote(container)} >/dev/null", check=False)
        self.owned_candidates.remove(container)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and gpu_processes(self.wzu):
            time.sleep(1)
        if gpu_processes(self.wzu):
            raise MaintenanceError(f"GPU did not become idle after {variant['name']}")
        return summary

    def restore_production(self) -> None:
        if self.original_inspect is None:
            return
        name = str(self.config["production"]["container"])
        current = inspect_container(self.wzu, name)
        if current.get("Id") != self.original_inspect.get("Id"):
            raise MaintenanceError("production container ID changed; refusing to claim restoration")
        if config_fingerprint(current) != self.original_fingerprint:
            raise MaintenanceError("production container configuration changed during maintenance")
        if not (current.get("State") or {}).get("Running"):
            self.wzu.command(f"docker start {shlex.quote(name)}")
        timeout = float(self.config["production"].get("startup_timeout_seconds", 900))
        wait_docker_healthy(self.wzu, name, timeout)
        wait_http(self.wzu, str(self.config["production"]["base_url"]).rstrip("/") + "/health", 30)
        current = inspect_container(self.wzu, name)
        assert_production_gpu_owner(self.wzu, current)
        if config_fingerprint(current) != self.original_fingerprint:
            raise MaintenanceError("production fingerprint differs after start")
        self.production_stopped = False
        if self.production_watchdog_pid:
            self.wzu.command(
                f"kill {shlex.quote(self.production_watchdog_pid)} 2>/dev/null || true",
                check=False,
            )
        if self.production_watchdog_paths:
            self.wzu.command(
                "rm -f "
                + " ".join(shlex.quote(path) for path in self.production_watchdog_paths),
                check=False,
            )
        self.record("production-restored", container_id=current.get("Id"))

    def restore_gateway(self) -> None:
        if not self.gateway_gated:
            return
        assert self.original_caddy_sha and self.maintenance_caddy_sha
        gateway = self.config["gateway"]
        caddy_path = str(gateway["caddyfile"])
        env_file = str(gateway["environment_file"])
        caddy_bin = str(gateway.get("caddy_binary", "/usr/local/bin/caddy"))
        restore = f"""set -eu
current=$(sha256sum {shlex.quote(caddy_path)} | awk '{{print $1}}')
[ "$current" = {shlex.quote(self.maintenance_caddy_sha)} ]
cp -p {shlex.quote(self.gateway_paths['backup'])} {shlex.quote(caddy_path)}
restored=$(sha256sum {shlex.quote(caddy_path)} | awk '{{print $1}}')
[ "$restored" = {shlex.quote(self.original_caddy_sha)} ]
set -a; . {shlex.quote(env_file)}; set +a
{shlex.quote(caddy_bin)} validate --config {shlex.quote(caddy_path)}
systemctl reload caddy
"""
        self.tokyo.command("sh -s", input_bytes=restore.encode("utf-8"))
        wait_public_status(str(gateway["public_health_url"]), 401)
        wait_http(self.tokyo, "http://127.0.0.1:18000/health", 30)
        if self.watchdog_pid:
            self.tokyo.command(f"kill {shlex.quote(self.watchdog_pid)} 2>/dev/null || true", check=False)
        cleanup = "rm -f " + " ".join(shlex.quote(value) for value in self.gateway_paths.values())
        self.tokyo.command(cleanup, check=False)
        self.gateway_gated = False
        self.record("gateway-restored", caddy_sha256=self.original_caddy_sha)

    def cleanup_candidates(self) -> None:
        for name in list(self.owned_candidates):
            self.wzu.command(f"docker rm -f {shlex.quote(name)} >/dev/null 2>&1 || true", check=False)
            self.owned_candidates.remove(name)

    def run(self) -> dict[str, Any]:
        try:
            self.acquire_locks()
        except Exception:
            for lock in reversed(self.locks):
                lock.release()
            self.locks.clear()
            raise
        summaries: list[dict[str, Any]] = []
        primary_error: Exception | None = None
        try:
            self.snapshot()
            self.gate_ingress()
            self.drain_and_stop_production()
            for index, variant in enumerate(self.config["variants"]):
                summaries.append(self.run_variant(index, variant))
        except Exception as exc:
            primary_error = exc
            self.record("primary-error", error=f"{type(exc).__name__}: {exc}")
        restore_errors: list[str] = []
        try:
            self.cleanup_candidates()
        except Exception as exc:
            restore_errors.append(f"candidate cleanup: {exc}")
        try:
            self.restore_production()
        except Exception as exc:
            restore_errors.append(f"production restore: {exc}")
        # Ingress is restored only after production restoration succeeded.
        if not any(error.startswith("production restore:") for error in restore_errors):
            try:
                self.restore_gateway()
            except Exception as exc:
                restore_errors.append(f"gateway restore: {exc}")
        for lock in reversed(self.locks):
            lock.release()
        self.locks.clear()
        result = {
            "run_id": self.run_id,
            "finished_at": utc_now(),
            "passed": primary_error is None and not restore_errors and all(s.get("passed") for s in summaries),
            "primary_error": None if primary_error is None else f"{type(primary_error).__name__}: {primary_error}",
            "restore_errors": restore_errors,
            "variants": summaries,
            "production_restored": not self.production_stopped,
            "gateway_restored": not self.gateway_gated,
        }
        json_dump(self.output / "result.json", result)
        if primary_error is not None or restore_errors:
            raise MaintenanceError(json.dumps(result, ensure_ascii=False))
        return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--execute", action="store_true", help="perform the maintenance A/B")
    parser.add_argument("--confirm", default="", help="must equal config.execute_confirmation")
    parser.add_argument("--run-id", help="lowercase run identifier; generated by default")
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=SCRIPT_DIR / "results",
        help="local audit/result parent",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        config_path = args.config.expanduser().resolve()
        config = load_config(config_path)
        if not args.execute:
            print(render_plan(config), end="")
            return 0
        expected = str(config.get("execute_confirmation", "MAINTENANCE-AB"))
        if args.confirm != expected:
            raise MaintenanceError(
                f"execution requires --confirm {expected!r}; default mode remains dry-run"
            )
        for variant in config["variants"]:
            if not variant.get("expected_manifest_sha256") or "REPLACE" in str(variant["release_bin"]):
                raise MaintenanceError("all execute variants need real paths and pinned manifest hashes")
        run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        if not RUN_ID_RE.fullmatch(run_id):
            raise MaintenanceError(f"invalid run id {run_id!r}")
        output = args.output_root.expanduser().resolve() / f"maintenance-ab-{run_id}"
        output.mkdir(parents=True, exist_ok=False)
        json_dump(output / "config.json", config)
        result = Orchestrator(config, output, run_id).run()
        print(output)
        return 0 if result["passed"] else 1
    except MaintenanceError as exc:
        print(f"maintenance-ab: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
