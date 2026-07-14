"""Select the ONNX Runtime provider used by the G2PW frontend."""

from __future__ import annotations

import json
import os
from typing import Callable, Optional


VALID_G2PW_BACKENDS = {"auto", "cpu", "cuda"}


def _emit_default(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _optional_positive_int(value, *, name: str) -> Optional[int]:
    if value in (None, ""):
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def cuda_device_id_from_device(device) -> int:
    value = str(device).strip().lower()
    if not value.startswith("cuda:"):
        return 0
    return int(value.split(":", 1)[1])


class G2PWBackend:
    def __init__(
        self,
        *,
        requested: str = "auto",
        cuda_device_id: int = 0,
        cuda_memory_limit_mb: Optional[int] = None,
        event_sink: Callable[[dict], None] = _emit_default,
    ):
        requested = str(requested).strip().lower()
        if requested not in VALID_G2PW_BACKENDS:
            raise ValueError(
                f"Unsupported G2PW backend {requested!r}; expected one of "
                f"{sorted(VALID_G2PW_BACKENDS)}"
            )
        cuda_device_id = int(cuda_device_id)
        if cuda_device_id < 0:
            raise ValueError("cuda_device_id must not be negative")

        self.requested = requested
        self.cuda_device_id = cuda_device_id
        self.cuda_memory_limit_mb = _optional_positive_int(
            cuda_memory_limit_mb,
            name="cuda_memory_limit_mb",
        )
        self.event_sink = event_sink
        self._status = {
            "requested": requested,
            "active": "unloaded",
            "state": "unloaded",
            "fallback_reason": None,
            "available_providers": [],
            "session_providers": [],
            "cuda_device_id": cuda_device_id,
            "cuda_memory_limit_mb": self.cuda_memory_limit_mb,
        }

    def status_dict(self) -> dict:
        return dict(self._status)

    def _cuda_provider(self):
        options = {"device_id": str(self.cuda_device_id)}
        if self.cuda_memory_limit_mb is not None:
            options["gpu_mem_limit"] = str(self.cuda_memory_limit_mb * 1024 * 1024)
        return "CUDAExecutionProvider", options

    def _activate(self, session, *, active: str, state: str, fallback_reason=None):
        self._status.update(
            {
                "active": active,
                "state": state,
                "fallback_reason": fallback_reason,
                "session_providers": list(session.get_providers()),
            }
        )
        self.event_sink({"event": "thin_tts_g2pw_backend", **self.status_dict()})
        return session

    def _create_cpu_session(self, ort, onnx_path, sess_options, *, fallback_reason=None):
        if "CPUExecutionProvider" not in self._status["available_providers"]:
            raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        return self._activate(
            session,
            active="cpu",
            state="fallback" if fallback_reason else "active",
            fallback_reason=fallback_reason,
        )

    def create_session(self, ort, onnx_path, sess_options):
        available = list(ort.get_available_providers())
        self._status["available_providers"] = available

        if self.requested == "cpu":
            return self._create_cpu_session(ort, onnx_path, sess_options)

        if "CUDAExecutionProvider" not in available:
            return self._create_cpu_session(
                ort,
                onnx_path,
                sess_options,
                fallback_reason="CUDAExecutionProvider is unavailable",
            )

        try:
            session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=[self._cuda_provider(), "CPUExecutionProvider"],
            )
            if "CUDAExecutionProvider" not in session.get_providers():
                raise RuntimeError("ONNX Runtime did not activate CUDAExecutionProvider")
            return self._activate(session, active="cuda", state="active")
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            return self._create_cpu_session(
                ort,
                onnx_path,
                sess_options,
                fallback_reason=reason,
            )


_ACTIVE_BACKEND: Optional[G2PWBackend] = None


def configure_g2pw_backend(
    *,
    requested: Optional[str] = None,
    cuda_device_id: int = 0,
    cuda_memory_limit_mb: Optional[int] = None,
    event_sink: Callable[[dict], None] = _emit_default,
) -> G2PWBackend:
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = G2PWBackend(
        requested=requested or os.environ.get("THIN_TTS_G2PW_BACKEND", "auto"),
        cuda_device_id=cuda_device_id,
        cuda_memory_limit_mb=cuda_memory_limit_mb,
        event_sink=event_sink,
    )
    return _ACTIVE_BACKEND


def get_g2pw_backend() -> G2PWBackend:
    global _ACTIVE_BACKEND
    if _ACTIVE_BACKEND is None:
        _ACTIVE_BACKEND = configure_g2pw_backend(
            requested=os.environ.get("THIN_TTS_G2PW_BACKEND", "auto"),
            cuda_device_id=int(os.environ.get("THIN_TTS_G2PW_CUDA_DEVICE_ID", "0")),
            cuda_memory_limit_mb=os.environ.get("THIN_TTS_G2PW_CUDA_MEMORY_LIMIT_MB"),
        )
    return _ACTIVE_BACKEND


def g2pw_backend_status() -> dict:
    if _ACTIVE_BACKEND is None:
        return {
            "requested": None,
            "active": "unloaded",
            "state": "unloaded",
            "fallback_reason": None,
            "available_providers": [],
            "session_providers": [],
            "cuda_device_id": None,
            "cuda_memory_limit_mb": None,
        }
    return _ACTIVE_BACKEND.status_dict()
