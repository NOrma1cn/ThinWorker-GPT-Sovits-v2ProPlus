import sys
import types
from io import BytesIO

import torch

from thin_tts.process_ckpt import load_sovits_new


def _write_legacy_checkpoint(path, *, custom_header=False):
    legacy_utils = types.ModuleType("utils")

    def init(self, **kwargs):
        self.__dict__.update(kwargs)

    legacy_hparams = type("HParams", (), {"__init__": init})
    legacy_hparams.__module__ = "utils"
    legacy_hparams.__qualname__ = "HParams"
    legacy_utils.HParams = legacy_hparams
    sys.modules["utils"] = legacy_utils
    try:
        if custom_header:
            buffer = BytesIO()
            torch.save({"config": legacy_hparams(model={"version": "v2ProPlus"})}, buffer)
            path.write_bytes(b"06" + buffer.getvalue()[2:])
        else:
            torch.save({"config": legacy_hparams(model={"version": "v2ProPlus"})}, path)
    finally:
        sys.modules.pop("utils", None)


def _remove_upstream_gpt_sovits_path(monkeypatch):
    monkeypatch.setattr(
        sys,
        "path",
        [entry for entry in sys.path if "GPT-SoVITS-v2pro-20250604-nvidia50" not in entry],
    )


def test_load_sovits_supports_legacy_utils_hparams_without_upstream_module(tmp_path, monkeypatch):
    path = tmp_path / "legacy.pth"
    _write_legacy_checkpoint(path)
    _remove_upstream_gpt_sovits_path(monkeypatch)

    loaded = load_sovits_new(path)

    assert loaded["config"]["model"]["version"] == "v2ProPlus"
    assert "utils" not in sys.modules


def test_load_sovits_custom_header_supports_legacy_utils_hparams(tmp_path, monkeypatch):
    path = tmp_path / "legacy-v2-pro-plus.pth"
    _write_legacy_checkpoint(path, custom_header=True)
    _remove_upstream_gpt_sovits_path(monkeypatch)

    loaded = load_sovits_new(path)

    assert loaded["config"]["model"]["version"] == "v2ProPlus"
    assert "utils" not in sys.modules
