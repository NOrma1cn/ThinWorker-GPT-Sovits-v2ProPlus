from types import SimpleNamespace

import torch

from thin_tts.models.t2s_model import (
    hybrid_deadline_for_chunk,
    slice_semantic_boundary,
)
from thin_tts.server import _request_for


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
