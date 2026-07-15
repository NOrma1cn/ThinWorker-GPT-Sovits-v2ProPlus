from types import SimpleNamespace

import torch

from thin_tts.models.t2s_model import (
    hybrid_deadline_for_chunk,
    slice_semantic_boundary,
)
from thin_tts.streaming_scheduler import BufferAwareDeadlineScheduler
from thin_tts.server import _request_for


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_hybrid_scheduler_can_use_a_larger_steady_deadline():
    assert hybrid_deadline_for_chunk(50, 120, chunk_index=0) == 50
    assert hybrid_deadline_for_chunk(50, 120, chunk_index=1) == 120
    assert hybrid_deadline_for_chunk(50, 120, chunk_index=8) == 120


def test_hybrid_scheduler_preserves_existing_deadline_without_override():
    assert hybrid_deadline_for_chunk(50, 0, chunk_index=0) == 50
    assert hybrid_deadline_for_chunk(50, 0, chunk_index=4) == 50


def test_server_forwards_explicit_steady_deadline():
    cfg = SimpleNamespace(ref_audio="reference.wav", ref_text="reference text")

    request = _request_for(
        "测试文本。",
        cfg,
        mode=4,
        seed=314159,
        hybrid_switch_tokens=50,
        hybrid_steady_tokens=120,
    )

    assert request["hybrid_switch_tokens"] == 50
    assert request["hybrid_steady_tokens"] == 120
    assert request["hybrid_buffer_target_ms"] == 0


def test_mode4_defaults_to_a_buffer_aware_deadline():
    cfg = SimpleNamespace(ref_audio="reference.wav", ref_text="reference text")

    request = _request_for("测试文本。", cfg, mode=4, seed=314159)

    assert request["hybrid_switch_tokens"] == 50
    assert request["hybrid_steady_tokens"] == 0
    assert request["hybrid_buffer_target_ms"] == 500


def test_buffer_scheduler_relaxes_after_target_is_filled():
    clock = FakeClock()
    scheduler = BufferAwareDeadlineScheduler(
        low_buffer_tokens=50,
        target_buffer_ms=500,
        clock=clock,
    )

    assert scheduler.deadline_tokens(chunk_key=0) == 50
    scheduler.record_audio(samples=11520, sample_rate=32000)
    assert scheduler.deadline_tokens(chunk_key=1) == 50

    clock.now = 0.1
    scheduler.record_audio(samples=64000, sample_rate=32000)

    assert scheduler.deadline_tokens(chunk_key=2) == 0
    assert scheduler.buffer_ahead_ms() == 2260


def test_buffer_scheduler_restores_cap_before_predicted_underrun():
    clock = FakeClock()
    scheduler = BufferAwareDeadlineScheduler(
        low_buffer_tokens=50,
        target_buffer_ms=500,
        clock=clock,
    )
    scheduler.record_audio(samples=32000, sample_rate=32000)
    assert scheduler.deadline_tokens(chunk_key=0) == 0

    clock.now = 0.51

    assert scheduler.deadline_tokens(chunk_key=1) == 50
    assert scheduler.buffer_ahead_ms() == 490


def test_buffer_scheduler_freezes_the_deadline_within_a_semantic_chunk():
    clock = FakeClock()
    scheduler = BufferAwareDeadlineScheduler(
        low_buffer_tokens=50,
        target_buffer_ms=500,
        clock=clock,
    )
    scheduler.record_audio(samples=32000, sample_rate=32000)
    assert scheduler.deadline_tokens(chunk_key=0) == 0

    clock.now = 0.75

    assert scheduler.deadline_tokens(chunk_key=0) == 0
    assert scheduler.deadline_tokens(chunk_key=1) == 50


def test_mute_boundary_does_not_emit_lookahead_tokens_twice():
    pending = torch.arange(14).unsqueeze(0)

    emitted, next_ptr = slice_semantic_boundary(
        pending,
        start=0,
        boundary_tokens=12,
    )

    assert emitted.tolist() == [list(range(12))]
    assert next_ptr == 12
    assert pending[:, next_ptr:].tolist() == [[12, 13]]
