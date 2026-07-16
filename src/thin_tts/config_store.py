"""Human-editable configuration storage for the launcher and TUI."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import MutableMapping

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap


PRESETS = {
    "max-performance": {
        "device": "cuda",
        "half": True,
        "t2s_backend": "triton",
        "g2pw_backend": "cuda",
        "g2pw_cuda_memory_limit_mb": 1536,
        "fallback_policy": "fail",
        "rng_isolation": True,
        "cache_vits_encoded_text": True,
    },
    "balanced": {
        "device": "cuda",
        "half": True,
        "t2s_backend": "triton",
        "g2pw_backend": "cpu",
        "g2pw_cuda_memory_limit_mb": None,
        "fallback_policy": "warn",
        "rng_isolation": True,
        "cache_vits_encoded_text": True,
    },
    "compatible": {
        "device": "cuda",
        "half": True,
        "t2s_backend": "sdpa",
        "g2pw_backend": "cpu",
        "g2pw_cuda_memory_limit_mb": None,
        "fallback_policy": "allow",
        "rng_isolation": True,
        "cache_vits_encoded_text": True,
    },
}


def default_config_path() -> Path:
    override = os.environ.get("THIN_TTS_CONFIG")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "thin-tts" / "config.yaml"


def new_config_document(path: Path | None = None) -> CommentedMap:
    config_path = Path(path or default_config_path()).expanduser()
    document = CommentedMap()
    document["server"] = CommentedMap(
        {
            "preset": "max-performance",
            "host": "0.0.0.0",
            "port": 9881,
            **PRESETS["max-performance"],
        }
    )
    document["weights"] = CommentedMap(
        {
            "t2s_weights": "",
            "vits_weights": "",
            "vits_lora": "",
            "bert_path": "",
            "hubert_path": "",
            "sv_path": "",
            "ref_audio": "",
            "ref_text": "",
            "voice_profile": str(config_path.parent / "voice-profile.pt"),
        }
    )
    return document


def apply_preset(document: MutableMapping, preset: str) -> None:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}")
    server = document.setdefault("server", CommentedMap())
    server["preset"] = preset
    for name, value in PRESETS[preset].items():
        server[name] = value


class ConfigStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or default_config_path()).expanduser()
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.indent(mapping=2, sequence=4, offset=2)

    def load(self):
        if not self.path.exists():
            return new_config_document(self.path)
        with self.path.open("r", encoding="utf-8") as handle:
            document = self.yaml.load(handle)
        return document or new_config_document(self.path)

    def save(self, document) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            shutil.copy2(self.path, Path(f"{self.path}.bak"))
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
                self.yaml.dump(document, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
