"""Small, request-local scheduling helpers for streaming inference."""

from __future__ import annotations

import time
from collections.abc import Callable


class BufferAwareDeadlineScheduler:
    """Apply a semantic-token cap only while predicted playback buffer is low."""

    def __init__(
        self,
        *,
        low_buffer_tokens: int,
        target_buffer_ms: int,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if low_buffer_tokens <= 0:
            raise ValueError("low_buffer_tokens must be greater than zero")
        if target_buffer_ms <= 0:
            raise ValueError("target_buffer_ms must be greater than zero")
        self.low_buffer_tokens = int(low_buffer_tokens)
        self.target_buffer_ms = int(target_buffer_ms)
        self._clock = clock
        self._playback_started_at: float | None = None
        self._emitted_audio_s = 0.0
        self._deadline_chunk_key = object()
        self._deadline_tokens = self.low_buffer_tokens

    def record_audio(self, *, samples: int, sample_rate: int) -> None:
        if samples < 0:
            raise ValueError("samples must not be negative")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if self._playback_started_at is None:
            self._playback_started_at = self._clock()
        self._emitted_audio_s += samples / sample_rate

    def buffer_ahead_ms(self) -> int:
        if self._playback_started_at is None:
            return 0
        elapsed_s = max(0.0, self._clock() - self._playback_started_at)
        buffer_s = max(0.0, self._emitted_audio_s - elapsed_s)
        return round(buffer_s * 1000)

    def deadline_tokens(self, *, chunk_key) -> int:
        if chunk_key != self._deadline_chunk_key:
            self._deadline_chunk_key = chunk_key
            self._deadline_tokens = (
                self.low_buffer_tokens
                if self.buffer_ahead_ms() < self.target_buffer_ms
                else 0
            )
        return self._deadline_tokens
