"""Detached server process lifecycle used by the CLI and TUI launcher."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import psutil
import requests
import yaml

from thin_tts.diagnostics import FailureReport, advise_failure


def default_state_dir() -> Path:
    override = os.environ.get("THIN_TTS_STATE_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "thin-tts"


@dataclass
class ServiceState:
    pid: int
    create_time: float
    config_path: str
    command: list[str]
    health_url: Optional[str]
    stdout_log: str
    stderr_log: str
    started_at: float

    @classmethod
    def from_dict(cls, values: dict) -> "ServiceState":
        return cls(**values)


@dataclass
class ServiceSnapshot:
    state: str
    pid: Optional[int] = None
    process_alive: bool = False
    health: Optional[dict] = None
    failure: Optional[FailureReport] = None
    service: Optional[ServiceState] = None


class ServiceManager:
    """Launch a server that deliberately outlives this manager instance."""

    def __init__(self, state_dir: str | Path | None = None) -> None:
        self.state_dir = Path(state_dir or default_state_dir()).expanduser()
        self.state_path = self.state_dir / "service.json"

    def start(
        self,
        config_path: str | Path,
        *,
        command: Optional[list[str]] = None,
        health_url: Optional[str] = None,
    ) -> ServiceState:
        current = self.status()
        if current.process_alive:
            raise RuntimeError(f"thin-tts-server is already running with PID {current.pid}")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        config_path = Path(config_path).expanduser()
        if command is None:
            command = [
                sys.executable,
                "-m",
                "thin_tts",
                "serve",
                "--config",
                str(config_path.resolve()),
            ]
        if health_url is None and config_path.is_file():
            health_url = self._health_url_from_config(config_path)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        stdout_path = self.state_dir / f"service-{stamp}.stdout.log"
        stderr_path = self.state_dir / f"service-{stamp}.stderr.log"
        popen_options = {
            "stdin": subprocess.DEVNULL,
            "cwd": str(self.state_dir),
            "env": os.environ.copy(),
            "close_fds": True,
        }
        if os.name == "nt":
            popen_options["creationflags"] = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            popen_options["start_new_session"] = True
        with stdout_path.open("ab", buffering=0) as stdout_handle, stderr_path.open(
            "ab", buffering=0
        ) as stderr_handle:
            process = subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                **popen_options,
            )
        try:
            create_time = psutil.Process(process.pid).create_time()
        except psutil.NoSuchProcess:
            # Preserve log paths so a launcher can explain an immediate startup failure.
            create_time = 0.0
        state = ServiceState(
            pid=process.pid,
            create_time=create_time,
            config_path=str(config_path),
            command=list(command),
            health_url=health_url,
            stdout_log=str(stdout_path),
            stderr_log=str(stderr_path),
            started_at=time.time(),
        )
        self._write_state(state)
        return state

    def status(self) -> ServiceSnapshot:
        state = self._read_state()
        if state is None:
            return ServiceSnapshot(state="stopped")
        alive = self._process_matches(state)
        if alive and state.health_url:
            try:
                response = requests.get(state.health_url, timeout=0.5)
                response.raise_for_status()
                return ServiceSnapshot(
                    state="ready",
                    pid=state.pid,
                    process_alive=True,
                    health=response.json(),
                    service=state,
                )
            except (requests.RequestException, ValueError):
                pass
        if alive:
            return ServiceSnapshot(
                state="starting",
                pid=state.pid,
                process_alive=True,
                service=state,
            )
        failure = self._failure_from_logs(state)
        return ServiceSnapshot(
            state="failed" if failure else "stopped",
            pid=state.pid,
            process_alive=False,
            failure=failure,
            service=state,
        )

    def stop(self, timeout: float = 10) -> bool:
        state = self._read_state()
        if state is None or not self._process_matches(state):
            return False
        process = psutil.Process(state.pid)
        targets = process.children(recursive=True)
        targets.append(process)
        for target in reversed(targets):
            try:
                target.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(targets, timeout=timeout)
        for target in alive:
            try:
                target.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            self.state_path.unlink()
        except FileNotFoundError:
            pass
        return True

    def events(self, limit: int = 200) -> list[dict]:
        state = self._read_state()
        if state is None:
            return []
        return self._json_events(Path(state.stdout_log), limit=limit)

    def logs(self, limit: int = 200) -> str:
        state = self._read_state()
        if state is None:
            return ""
        chunks = []
        for label, value in (("STDOUT", state.stdout_log), ("STDERR", state.stderr_log)):
            path = Path(value)
            if path.exists():
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                chunks.append(f"--- {label} ---\n" + "\n".join(lines[-limit:]))
        return "\n".join(chunks)

    def _read_state(self) -> Optional[ServiceState]:
        try:
            values = json.loads(self.state_path.read_text(encoding="utf-8"))
            return ServiceState.from_dict(values)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return None

    def _write_state(self, state: ServiceState) -> None:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".service.", suffix=".tmp", dir=self.state_dir
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                json.dump(asdict(state), handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.state_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _process_matches(state: ServiceState) -> bool:
        try:
            process = psutil.Process(state.pid)
            return (
                process.is_running()
                and process.status() != psutil.STATUS_ZOMBIE
                and abs(process.create_time() - state.create_time) < 1
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    @staticmethod
    def _health_url_from_config(config_path: Path) -> str:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        server = document.get("server", {})
        host = str(server.get("host", "127.0.0.1"))
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        return f"http://{host}:{int(server.get('port', 9881))}/health"

    @staticmethod
    def _json_events(path: Path, *, limit: int) -> list[dict]:
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        events = []
        for line in lines[-limit:]:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "event" in value:
                events.append(value)
        return events

    def _failure_from_logs(self, state: ServiceState) -> Optional[FailureReport]:
        events = self._json_events(Path(state.stdout_log), limit=500)
        for event in reversed(events):
            if event.get("event") == "thin_tts_startup_failure":
                return advise_failure(event.get("stage", "startup"), event.get("reason", ""))
        stderr_path = Path(state.stderr_log)
        if stderr_path.exists():
            lines = stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if lines:
                return advise_failure("process", "\n".join(lines[-20:]))
        return None
