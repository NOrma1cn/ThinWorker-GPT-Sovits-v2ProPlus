import io
import json

import pytest

from thin_tts.events import StartupEvents


def _events(output):
    return [json.loads(line) for line in output.getvalue().splitlines()]


def test_startup_stage_emits_started_and_completed_events():
    output = io.StringIO()
    events = StartupEvents(output=output, clock=iter([10.0, 10.125]).__next__)

    with events.stage("configuration"):
        pass

    emitted = _events(output)
    assert [item["status"] for item in emitted] == ["started", "completed"]
    assert emitted[0]["event"] == "thin_tts_startup_stage"
    assert emitted[0]["schema"] == 1
    assert emitted[1]["elapsed_ms"] == 125


def test_startup_stage_emits_failure_without_swallowing_exception():
    output = io.StringIO()
    events = StartupEvents(output=output, clock=iter([4.0, 4.5]).__next__)

    with pytest.raises(RuntimeError, match="CUDA provider unavailable"):
        with events.stage("g2pw"):
            raise RuntimeError("CUDA provider unavailable")

    emitted = _events(output)
    assert emitted[-1]["event"] == "thin_tts_startup_failure"
    assert emitted[-1]["stage"] == "g2pw"
    assert emitted[-1]["reason"] == "CUDA provider unavailable"
    assert emitted[-1]["exception_type"] == "RuntimeError"
