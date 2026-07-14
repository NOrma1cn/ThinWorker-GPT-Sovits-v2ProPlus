import pytest

from thin_tts.g2pw_backend import (
    G2PWBackend,
    cuda_device_id_from_device,
    g2pw_backend_status,
)


class FakeSession:
    def __init__(self, providers):
        self._providers = providers

    def get_providers(self):
        return list(self._providers)


class FakeOrt:
    def __init__(self, available, *, cuda_error=None):
        self.available = available
        self.cuda_error = cuda_error
        self.calls = []

    def get_available_providers(self):
        return list(self.available)

    def InferenceSession(self, path, *, sess_options, providers):
        self.calls.append((path, sess_options, providers))
        names = [provider[0] if isinstance(provider, tuple) else provider for provider in providers]
        if names[0] == "CUDAExecutionProvider" and self.cuda_error:
            raise self.cuda_error
        return FakeSession([name for name in names if name in self.available])


def test_cpu_request_never_initializes_cuda():
    ort = FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])
    backend = G2PWBackend(requested="cpu", event_sink=lambda _: None)

    backend.create_session(ort, "g2pw.onnx", object())

    assert ort.calls[0][2] == ["CPUExecutionProvider"]
    assert backend.status_dict()["active"] == "cpu"


def test_auto_uses_cuda_with_device_and_memory_options():
    ort = FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])
    backend = G2PWBackend(
        requested="auto",
        cuda_device_id=1,
        cuda_memory_limit_mb=1536,
        event_sink=lambda _: None,
    )

    backend.create_session(ort, "g2pw.onnx", object())

    cuda_provider = ort.calls[0][2][0]
    assert cuda_provider == (
        "CUDAExecutionProvider",
        {"device_id": "1", "gpu_mem_limit": str(1536 * 1024 * 1024)},
    )
    assert backend.status_dict()["active"] == "cuda"


def test_cuda_initialization_failure_falls_back_to_cpu():
    ort = FakeOrt(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        cuda_error=RuntimeError("CUDA library mismatch"),
    )
    backend = G2PWBackend(requested="cuda", event_sink=lambda _: None)

    backend.create_session(ort, "g2pw.onnx", object())

    assert ort.calls[-1][2] == ["CPUExecutionProvider"]
    assert backend.status_dict()["state"] == "fallback"
    assert "CUDA library mismatch" in backend.status_dict()["fallback_reason"]


def test_auto_without_cuda_provider_uses_cpu_with_reason():
    ort = FakeOrt(["CPUExecutionProvider"])
    backend = G2PWBackend(requested="auto", event_sink=lambda _: None)

    backend.create_session(ort, "g2pw.onnx", object())

    assert backend.status_dict()["active"] == "cpu"
    assert backend.status_dict()["fallback_reason"] == "CUDAExecutionProvider is unavailable"


def test_invalid_backend_name_fails_fast():
    with pytest.raises(ValueError, match="Unsupported G2PW backend"):
        G2PWBackend(requested="tensorrt")


def test_unconfigured_runtime_status_is_unloaded():
    assert g2pw_backend_status()["active"] == "unloaded"


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cuda", 0), ("cuda:0", 0), ("cuda:2", 2), ("cpu", 0)],
)
def test_cuda_device_id_follows_torch_device(device, expected):
    assert cuda_device_id_from_device(device) == expected
