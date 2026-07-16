from pathlib import Path

import yaml

from thin_tts.cli import build_parser
from thin_tts.config import ServerConfig


def _write_config(tmp_path: Path, *, half: bool = False) -> Path:
    weights = {}
    for name in ("t2s_weights", "vits_weights", "bert_path", "hubert_path", "sv_path", "ref_audio"):
        path = tmp_path / name
        path.touch()
        weights[name] = str(path)
    weights["ref_text"] = "参考文本"
    weights["voice_profile"] = str(tmp_path / "voice-profile.pt")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "server": {
                    "preset": "custom",
                    "host": "127.0.0.1",
                    "port": 9894,
                    "device": "cpu",
                    "half": half,
                    "t2s_backend": "sdpa",
                    "g2pw_backend": "cpu",
                    "g2pw_cuda_memory_limit_mb": 1024,
                    "fallback_policy": "warn",
                },
                "weights": weights,
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return config_path


def test_omitted_cli_options_preserve_yaml_server_values(tmp_path):
    config_path = _write_config(tmp_path, half=False)

    args = build_parser().parse_args(["--config", str(config_path)])
    config = ServerConfig.from_args(args)

    assert config.host == "127.0.0.1"
    assert config.preset == "custom"
    assert config.port == 9894
    assert config.device == "cpu"
    assert config.half is False
    assert config.t2s_backend == "sdpa"
    assert config.g2pw_backend == "cpu"
    assert config.g2pw_cuda_memory_limit_mb == 1024
    assert config.fallback_policy == "warn"
    assert config.voice_profile == str(tmp_path / "voice-profile.pt")


def test_explicit_cli_options_override_yaml_server_values(tmp_path):
    config_path = _write_config(tmp_path, half=True)

    args = build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--host",
            "0.0.0.0",
            "--port",
            "9999",
            "--device",
            "cuda:1",
            "--no-half",
            "--t2s-backend",
            "auto",
            "--g2pw-backend",
            "cuda",
            "--g2pw-cuda-memory-limit-mb",
            "1536",
            "--voice-profile",
            str(tmp_path / "override-profile.pt"),
        ]
    )
    config = ServerConfig.from_args(args)

    assert config.host == "0.0.0.0"
    assert config.port == 9999
    assert config.device == "cuda:1"
    assert config.half is False
    assert config.t2s_backend == "auto"
    assert config.g2pw_backend == "cuda"
    assert config.g2pw_cuda_memory_limit_mb == 1536
    assert config.voice_profile == str(tmp_path / "override-profile.pt")


def test_environment_can_select_g2pw_backend(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    monkeypatch.setenv("THIN_TTS_G2PW_BACKEND", "auto")
    monkeypatch.setenv("THIN_TTS_G2PW_CUDA_MEMORY_LIMIT_MB", "2048")

    args = build_parser().parse_args(["--config", str(config_path)])
    config = ServerConfig.from_args(args)

    assert config.g2pw_backend == "auto"
    assert config.g2pw_cuda_memory_limit_mb == 2048


def test_environment_overrides_yaml_server_values(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, half=True)
    monkeypatch.setenv("THIN_TTS_PRESET", "compatible")
    monkeypatch.setenv("THIN_TTS_HOST", "0.0.0.0")
    monkeypatch.setenv("THIN_TTS_PORT", "9988")
    monkeypatch.setenv("THIN_TTS_DEVICE", "cuda")
    monkeypatch.setenv("THIN_TTS_HALF", "false")
    monkeypatch.setenv("THIN_TTS_FALLBACK_POLICY", "allow")
    monkeypatch.setenv("THIN_TTS_RNG_ISOLATION", "false")
    monkeypatch.setenv("THIN_TTS_CACHE_VITS_ENCODED_TEXT", "false")

    args = build_parser().parse_args(["--config", str(config_path)])
    config = ServerConfig.from_args(args)

    assert config.preset == "compatible"
    assert config.host == "0.0.0.0"
    assert config.port == 9988
    assert config.device == "cuda"
    assert config.half is False
    assert config.fallback_policy == "allow"
    assert config.rng_isolation is False
    assert config.cache_vits_encoded_text is False


def test_preset_supplies_backend_defaults_when_yaml_omits_them(tmp_path):
    config_path = _write_config(tmp_path)
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["server"] = {"preset": "max-performance"}
    config_path.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")

    config = ServerConfig.from_args(build_parser().parse_args(["--config", str(config_path)]))

    assert config.device == "cuda"
    assert config.half is True
    assert config.t2s_backend == "triton"
    assert config.g2pw_backend == "cuda"
    assert config.g2pw_cuda_memory_limit_mb == 1536
    assert config.fallback_policy == "fail"
    assert config.rng_isolation is True
    assert config.cache_vits_encoded_text is True
