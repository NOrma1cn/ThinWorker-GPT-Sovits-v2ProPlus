"""Structured process events consumed by the launcher and TUI."""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from typing import Callable, TextIO


class StartupEvents:
    """Emit stable newline-delimited JSON startup events."""

    def __init__(
        self,
        *,
        output: TextIO | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.output = output or sys.stdout
        self.clock = clock

    def emit(self, event: str, **fields) -> None:
        payload = {"schema": 1, "event": event, **fields}
        print(json.dumps(payload, ensure_ascii=False), file=self.output, flush=True)

    @contextmanager
    def stage(self, name: str):
        started = self.clock()
        self.emit("thin_tts_startup_stage", stage=name, status="started")
        try:
            yield
        except BaseException as exc:
            elapsed_ms = round((self.clock() - started) * 1000)
            self.emit(
                "thin_tts_startup_failure",
                stage=name,
                status="failed",
                elapsed_ms=elapsed_ms,
                reason=str(exc) or type(exc).__name__,
                exception_type=type(exc).__name__,
            )
            raise
        else:
            elapsed_ms = round((self.clock() - started) * 1000)
            self.emit(
                "thin_tts_startup_stage",
                stage=name,
                status="completed",
                elapsed_ms=elapsed_ms,
            )


startup_events = StartupEvents()
