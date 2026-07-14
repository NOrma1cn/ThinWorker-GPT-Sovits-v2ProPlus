import asyncio

import pytest

from thin_tts.backends import configure_t2s_backend
from thin_tts.pipeline.tts import TTS_Config
from thin_tts.server import _backend_status, health


class FakeBackend:
    def __init__(self):
        self.installed = False
        self.status = {
            "requested": "triton",
            "active": "triton",
            "state": "installed",
            "fallback_reason": None,
            "capture_count": 0,
            "graph_memory_mb": None,
        }

    def install(self):
        self.installed = True

    def status_dict(self):
        return dict(self.status)


def test_sdpa_request_never_probes_or_builds_triton():
    calls = []
    backend = configure_t2s_backend(
        object(),
        requested="sdpa",
        device="cuda",
        is_half=True,
        capability_probe=lambda **_: calls.append("probe"),
        backend_factory=lambda *_args, **_kwargs: calls.append("factory"),
    )

    assert backend.status_dict()["active"] == "sdpa"
    assert backend.status_dict()["state"] == "disabled"
    assert calls == []


def test_auto_falls_back_when_environment_is_incompatible():
    backend = configure_t2s_backend(
        object(),
        requested="auto",
        device="cuda",
        is_half=True,
        capability_probe=lambda **_: "Linux is required",
    )

    status = backend.status_dict()
    assert status["active"] == "sdpa"
    assert status["state"] == "fallback"
    assert status["fallback_reason"] == "Linux is required"


def test_compatible_environment_installs_triton_backend():
    candidate = FakeBackend()
    backend = configure_t2s_backend(
        object(),
        requested="auto",
        device="cuda",
        is_half=True,
        capability_probe=lambda **_: None,
        backend_factory=lambda *_args, **_kwargs: candidate,
    )

    assert backend is candidate
    assert candidate.installed is True


def test_backend_initialization_failure_falls_back_with_reason():
    backend = configure_t2s_backend(
        object(),
        requested="triton",
        device="cuda",
        is_half=True,
        capability_probe=lambda **_: None,
        backend_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("capture setup failed")),
    )

    status = backend.status_dict()
    assert status["active"] == "sdpa"
    assert status["state"] == "fallback"
    assert "capture setup failed" in status["fallback_reason"]


def test_strict_triton_request_raises_instead_of_falling_back():
    with pytest.raises(RuntimeError, match="Triton backend unavailable"):
        configure_t2s_backend(
            object(),
            requested="triton",
            device="cuda",
            is_half=True,
            strict=True,
            capability_probe=lambda **_: "Triton package is missing",
        )


def test_invalid_backend_name_fails_fast():
    with pytest.raises(ValueError, match="Unsupported T2S backend"):
        configure_t2s_backend(object(), requested="fastest", device="cuda", is_half=True)


def test_server_backend_status_handles_loaded_and_unloaded_pipeline():
    assert _backend_status(None)["active"] == "unloaded"

    candidate = FakeBackend()
    pipeline = type("Pipeline", (), {"t2s_backend": candidate})()
    assert _backend_status(pipeline)["active"] == "triton"


def test_health_exposes_g2pw_backend_status():
    result = asyncio.run(health())

    assert result["g2pw_backend"]["active"] == "unloaded"


def test_environment_can_force_sdpa_without_probing_triton(monkeypatch):
    monkeypatch.setenv("THIN_TTS_T2S_BACKEND", "sdpa")
    calls = []
    backend = configure_t2s_backend(
        object(),
        requested=None,
        capability_probe=lambda **_: calls.append("probe"),
    )

    assert backend.status_dict()["requested"] == "sdpa"
    assert calls == []


def test_tts_config_preserves_backend_selection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = {}
    for name in ("t2s", "vits", "bert", "hubert", "sv"):
        path = tmp_path / name
        path.touch()
        paths[name] = str(path)
    config = TTS_Config(
        {
            "custom": {
                "device": "cpu",
                "is_half": False,
                "version": "v2ProPlus",
                "t2s_weights_path": paths["t2s"],
                "vits_weights_path": paths["vits"],
                "bert_base_path": paths["bert"],
                "cnhuhbert_base_path": paths["hubert"],
                "sv_path": paths["sv"],
                "t2s_backend": "sdpa",
            }
        }
    )

    assert config.t2s_backend == "sdpa"
    assert config.update_configs()["t2s_backend"] == "sdpa"
