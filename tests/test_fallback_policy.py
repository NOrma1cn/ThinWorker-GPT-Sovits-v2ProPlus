from types import SimpleNamespace

import pytest

from thin_tts.server import enforce_fallback_policy


def test_fail_policy_rejects_triton_fallback():
    cfg = SimpleNamespace(
        fallback_policy="fail",
        t2s_backend="triton",
        g2pw_backend="cuda",
    )

    with pytest.raises(RuntimeError, match="T2S requested triton but active backend is sdpa"):
        enforce_fallback_policy(
            cfg,
            t2s_status={"active": "sdpa", "fallback_reason": "Triton missing"},
            g2pw_status={"active": "cuda", "fallback_reason": None},
            event_sink=lambda *_args, **_kwargs: None,
        )


def test_warn_policy_reports_g2pw_fallback_without_failing():
    cfg = SimpleNamespace(
        fallback_policy="warn",
        t2s_backend="auto",
        g2pw_backend="cuda",
    )
    emitted = []

    enforce_fallback_policy(
        cfg,
        t2s_status={"active": "sdpa", "fallback_reason": None},
        g2pw_status={"active": "cpu", "fallback_reason": "CUDA provider missing"},
        event_sink=lambda event, **fields: emitted.append({"event": event, **fields}),
    )

    assert emitted == [
        {
            "event": "thin_tts_degraded",
            "policy": "warn",
            "reasons": ["G2PW requested cuda but active backend is cpu: CUDA provider missing"],
        }
    ]
