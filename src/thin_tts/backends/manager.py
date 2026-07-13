"""Select the production T2S backend without importing Triton eagerly."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
from typing import Callable, Optional


VALID_BACKENDS = {"auto", "sdpa", "triton"}


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _emit_default(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


class SDPABackend:
    def __init__(self, requested: str, state: str = "disabled", fallback_reason: Optional[str] = None):
        self._status = {
            "requested": requested,
            "active": "sdpa",
            "state": state,
            "fallback_reason": fallback_reason,
            "capture_count": 0,
            "graph_memory_mb": None,
        }

    def install(self) -> None:
        return None

    def restore(self) -> None:
        return None

    def status_dict(self) -> dict:
        return dict(self._status)


def probe_triton_capability(*, device, is_half: bool) -> Optional[str]:
    if platform.system() != "Linux":
        return "Linux is required"
    if "cuda" not in str(device).lower():
        return "CUDA device is required"
    if not is_half:
        return "FP16 model is required"
    try:
        import torch

        if not torch.cuda.is_available():
            return "CUDA is not available"
    except Exception as exc:
        return f"PyTorch CUDA probe failed: {exc}"
    try:
        if importlib.util.find_spec("triton") is None:
            return "Triton package is missing"
    except (ImportError, ValueError) as exc:
        return f"Triton package probe failed: {exc}"
    return None


def _default_factory(
    model, *, requested: str, strict: bool, event_sink: Callable[[dict], None]
):
    from thin_tts.backends.triton_graph import TritonGraphBackend

    return TritonGraphBackend(
        model, requested=requested, strict=strict, event_sink=event_sink
    )


def configure_t2s_backend(
    model,
    *,
    requested: Optional[str] = None,
    device="cuda",
    is_half: bool = True,
    strict: Optional[bool] = None,
    capability_probe: Callable[..., Optional[str]] = probe_triton_capability,
    backend_factory: Callable = _default_factory,
    event_sink: Callable[[dict], None] = _emit_default,
):
    requested = str(requested or os.environ.get("THIN_TTS_T2S_BACKEND", "auto")).strip().lower()
    if requested not in VALID_BACKENDS:
        raise ValueError(
            f"Unsupported T2S backend {requested!r}; expected one of {sorted(VALID_BACKENDS)}"
        )
    if strict is None:
        strict = _truthy(os.environ.get("THIN_TTS_T2S_BACKEND_STRICT"))

    if requested == "sdpa":
        backend = SDPABackend(requested)
        event_sink({"event": "thin_tts_backend", **backend.status_dict()})
        return backend

    unavailable_reason = capability_probe(device=device, is_half=is_half)
    if unavailable_reason:
        if requested == "triton" and strict:
            raise RuntimeError(f"Triton backend unavailable: {unavailable_reason}")
        backend = SDPABackend(requested, state="fallback", fallback_reason=unavailable_reason)
        event_sink({"event": "thin_tts_backend", **backend.status_dict()})
        return backend

    candidate = None
    try:
        candidate = backend_factory(
            model, requested=requested, strict=bool(strict), event_sink=event_sink
        )
        candidate.install()
        event_sink({"event": "thin_tts_backend", **candidate.status_dict()})
        return candidate
    except Exception as exc:
        if candidate is not None:
            try:
                candidate.restore()
            except Exception:
                pass
        if requested == "triton" and strict:
            raise
        reason = f"{type(exc).__name__}: {exc}"
        backend = SDPABackend(requested, state="fallback", fallback_reason=reason)
        event_sink({"event": "thin_tts_backend", **backend.status_dict()})
        return backend
