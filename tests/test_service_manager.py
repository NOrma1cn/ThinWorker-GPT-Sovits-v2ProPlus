import json
import os
import subprocess
import sys
import time

import thin_tts.service_manager as service_manager
from thin_tts.service_manager import ServiceManager


def test_detached_process_is_rediscovered_by_new_manager(tmp_path):
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    first = ServiceManager(state_dir=tmp_path)
    state = first.start("unused.yaml", command=command, health_url=None)

    try:
        del first
        second = ServiceManager(state_dir=tmp_path)
        snapshot = second.status()

        assert snapshot.process_alive is True
        assert snapshot.pid == state.pid
        assert snapshot.state == "starting"
    finally:
        ServiceManager(state_dir=tmp_path).stop(timeout=3)


def test_service_outlives_launcher_process(tmp_path):
    launcher = (
        "import sys; "
        "from thin_tts.service_manager import ServiceManager; "
        "ServiceManager(state_dir=sys.argv[1]).start("
        "'unused.yaml', command=[sys.executable, '-c', 'import time; time.sleep(60)'], "
        "health_url=None)"
    )
    environment = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", launcher, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        env=environment,
    )

    manager = ServiceManager(state_dir=tmp_path)
    try:
        assert result.returncode == 0, result.stderr
        snapshot = manager.status()
        assert snapshot.process_alive is True
        assert snapshot.state == "starting"
    finally:
        manager.stop(timeout=3)


def test_start_persists_logs_when_process_exits_before_create_time(tmp_path, monkeypatch):
    class FinishedProcess:
        pid = 4242

    monkeypatch.setattr(service_manager.subprocess, "Popen", lambda *args, **kwargs: FinishedProcess())

    def process_already_gone(pid):
        raise service_manager.psutil.NoSuchProcess(pid)

    monkeypatch.setattr(service_manager.psutil, "Process", process_already_gone)
    manager = ServiceManager(state_dir=tmp_path)

    state = manager.start("unused.yaml", command=["python"], health_url=None)

    assert state.create_time == 0.0
    assert manager.state_path.is_file()
    assert state.stdout_log.endswith(".stdout.log")
    assert state.stderr_log.endswith(".stderr.log")


def test_zombie_process_is_not_reported_as_alive(monkeypatch):
    class ZombieProcess:
        def is_running(self):
            return True

        def status(self):
            return service_manager.psutil.STATUS_ZOMBIE

        def create_time(self):
            return 100.0

    state = service_manager.ServiceState(
        pid=42,
        create_time=100.0,
        config_path="config.yaml",
        command=["python"],
        health_url=None,
        stdout_log="stdout.log",
        stderr_log="stderr.log",
        started_at=100.0,
    )
    monkeypatch.setattr(service_manager.psutil, "Process", lambda pid: ZombieProcess())

    assert ServiceManager._process_matches(state) is False


def test_service_manager_recovers_startup_failure_from_event_log(tmp_path):
    manager = ServiceManager(state_dir=tmp_path)
    manager.state_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = tmp_path / "service.stdout.log"
    stdout_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "event": "thin_tts_startup_failure",
                "stage": "g2pw",
                "reason": "CUDAExecutionProvider is unavailable",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manager.state_path.write_text(
        json.dumps(
            {
                "pid": 999999,
                "create_time": 0,
                "config_path": "config.yaml",
                "command": ["python"],
                "health_url": None,
                "stdout_log": str(stdout_path),
                "stderr_log": str(tmp_path / "service.stderr.log"),
                "started_at": time.time(),
            }
        ),
        encoding="utf-8",
    )

    snapshot = manager.status()

    assert snapshot.state == "failed"
    assert snapshot.failure.stage == "g2pw"
    assert "CUDA execution provider" in snapshot.failure.reason
    assert snapshot.failure.suggestions


def test_stop_without_state_is_idempotent(tmp_path):
    manager = ServiceManager(state_dir=tmp_path)

    assert manager.stop() is False
