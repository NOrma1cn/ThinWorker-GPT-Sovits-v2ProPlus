"""Server configuration and weight path resolution."""
import os
import re
from dataclasses import dataclass
from typing import Optional

import yaml

from thin_tts.config_store import PRESETS


class ConfigurationError(ValueError):
    """A user-correctable configuration problem."""


@dataclass
class ServerConfig:
    preset: str = "custom"
    host: str = "0.0.0.0"
    port: int = 9881
    device: str = "cuda"
    half: bool = True
    t2s_backend: str = "auto"
    g2pw_backend: str = "auto"
    g2pw_cuda_memory_limit_mb: Optional[int] = None
    fallback_policy: str = "warn"
    rng_isolation: bool = True
    cache_vits_encoded_text: bool = True

    # Weight paths
    t2s_weights: str = ""
    vits_weights: str = ""
    vits_lora: Optional[str] = None
    bert_path: str = ""
    hubert_path: str = ""
    sv_path: str = ""
    ref_audio: str = ""
    ref_text: str = ""
    voice_profile: Optional[str] = None

    @classmethod
    def from_args(cls, args) -> "ServerConfig":
        """Build ServerConfig from CLI args, with env var and YAML fallback."""
        # Start from YAML config if provided
        yaml_cfg = {}
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                yaml_cfg = yaml.safe_load(f) or {}

        server_cfg = yaml_cfg.get("server", {})
        weights_cfg = yaml_cfg.get("weights", {})
        preset = getattr(args, "preset", None)
        if preset is None:
            preset = os.environ.get("THIN_TTS_PRESET")
        if preset is None:
            preset = server_cfg.get("preset", "custom")
        preset = str(preset).strip().lower()
        preset_defaults = PRESETS.get(preset, {})

        def resolve(cli_val, env_key, yaml_key, default=""):
            if cli_val is not None:
                return cli_val
            env_val = os.environ.get(env_key)
            if env_val is not None:
                return env_val
            return yaml_key if yaml_key else default

        def server_value(name, default):
            cli_val = getattr(args, name, None)
            if cli_val is not None:
                return cli_val
            env_val = os.environ.get(f"THIN_TTS_{name.upper()}")
            if env_val is not None:
                return env_val
            if name in server_cfg:
                return server_cfg[name]
            return preset_defaults.get(name, default)

        cfg = cls(
            preset=preset,
            host=server_value("host", "0.0.0.0"),
            port=server_value("port", 9881),
            device=server_value("device", "cuda"),
            half=server_value("half", True),
            t2s_backend=server_value("t2s_backend", "auto"),
            g2pw_backend=server_value("g2pw_backend", "auto"),
            g2pw_cuda_memory_limit_mb=server_value("g2pw_cuda_memory_limit_mb", None),
            fallback_policy=server_value("fallback_policy", "warn"),
            rng_isolation=server_value("rng_isolation", True),
            cache_vits_encoded_text=server_value("cache_vits_encoded_text", True),
            t2s_weights=resolve(args.t2s_weights, "THIN_TTS_T2S_WEIGHTS",
                                weights_cfg.get("t2s_weights")),
            vits_weights=resolve(args.vits_weights, "THIN_TTS_VITS_WEIGHTS",
                                 weights_cfg.get("vits_weights")),
            vits_lora=resolve(args.vits_lora, "THIN_TTS_VITS_LORA",
                              weights_cfg.get("vits_lora")),
            bert_path=resolve(args.bert_path, "THIN_TTS_BERT_PATH",
                              weights_cfg.get("bert_path")),
            hubert_path=resolve(args.hubert_path, "THIN_TTS_HUBERT_PATH",
                                weights_cfg.get("hubert_path")),
            sv_path=resolve(args.sv_path, "THIN_TTS_SV_PATH",
                            weights_cfg.get("sv_path")),
            ref_audio=resolve(args.ref_audio, "THIN_TTS_REF_AUDIO",
                              weights_cfg.get("ref_audio")),
            ref_text=resolve(args.ref_text, "THIN_TTS_REF_TEXT",
                              weights_cfg.get("ref_text", "")),
            voice_profile=resolve(
                args.voice_profile,
                "THIN_TTS_VOICE_PROFILE",
                weights_cfg.get("voice_profile"),
                None,
            ),
        )

        cfg._validate()
        return cfg

    def _validate(self):
        self.preset = str(self.preset).strip().lower()
        if self.preset not in {"custom", "max-performance", "balanced", "compatible"}:
            raise ConfigurationError(
                f"Invalid preset {self.preset!r}; expected custom, max-performance, balanced, or compatible"
            )
        self.host = str(self.host).strip()
        try:
            self.port = int(self.port)
        except (TypeError, ValueError):
            raise ConfigurationError("port must be an integer") from None
        if not 1 <= self.port <= 65535:
            raise ConfigurationError("port must be between 1 and 65535")
        self.device = str(self.device).strip().lower()
        if not re.fullmatch(r"(?:cpu|cuda(?::\d+)?)", self.device):
            raise ConfigurationError("device must be cpu, cuda, or cuda:<index>")
        self.half = self._parse_bool("half", self.half)
        self.rng_isolation = self._parse_bool("rng_isolation", self.rng_isolation)
        self.cache_vits_encoded_text = self._parse_bool(
            "cache_vits_encoded_text", self.cache_vits_encoded_text
        )
        self.fallback_policy = str(self.fallback_policy).strip().lower()
        if self.fallback_policy not in {"fail", "warn", "allow"}:
            raise ConfigurationError(
                f"Invalid fallback_policy {self.fallback_policy!r}; expected fail, warn, or allow"
            )
        self.t2s_backend = str(self.t2s_backend).strip().lower()
        if self.t2s_backend not in {"auto", "sdpa", "triton"}:
            raise ConfigurationError(
                f"Invalid t2s_backend {self.t2s_backend!r}; expected auto, sdpa, or triton"
            )
        self.g2pw_backend = str(self.g2pw_backend).strip().lower()
        if self.g2pw_backend not in {"auto", "cpu", "cuda"}:
            raise ConfigurationError(
                f"Invalid g2pw_backend {self.g2pw_backend!r}; expected auto, cpu, or cuda"
            )
        if self.g2pw_cuda_memory_limit_mb not in (None, ""):
            try:
                self.g2pw_cuda_memory_limit_mb = int(self.g2pw_cuda_memory_limit_mb)
            except (TypeError, ValueError):
                raise ConfigurationError(
                    "g2pw_cuda_memory_limit_mb must be an integer"
                ) from None
            if self.g2pw_cuda_memory_limit_mb <= 0:
                raise ConfigurationError(
                    "g2pw_cuda_memory_limit_mb must be greater than zero"
                )
        else:
            self.g2pw_cuda_memory_limit_mb = None
        required = {
            "t2s_weights": self.t2s_weights,
            "vits_weights": self.vits_weights,
            "bert_path": self.bert_path,
            "hubert_path": self.hubert_path,
            "sv_path": self.sv_path,
            "ref_audio": self.ref_audio,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ConfigurationError(
                f"Missing required weight paths: {', '.join(missing)}. "
                "Provide them via CLI args, env vars (THIN_TTS_*), or a YAML config file."
            )

        for name, path in required.items():
            if not os.path.exists(path):
                raise ConfigurationError(f"{name} path does not exist: {path}")

    @staticmethod
    def _parse_bool(name: str, value) -> bool:
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ConfigurationError(f"{name} must be true or false")
