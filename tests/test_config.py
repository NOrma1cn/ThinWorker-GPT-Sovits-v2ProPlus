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

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "server": {
                    "host": "127.0.0.1",
                    "port": 9894,
                    "device": "cpu",
                    "half": half,
                    "t2s_backend": "sdpa",
                    "g2pw_backend": "cpu",
                    "g2pw_cuda_memory_limit_mb": 1024,
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
    assert config.port == 9894
    assert config.device == "cpu"
    assert config.half is False
    assert config.t2s_backend == "sdpa"
    assert config.g2pw_backend == "cpu"
    assert config.g2pw_cuda_memory_limit_mb == 1024


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
            "cuda",
            "--no-half",
            "--t2s-backend",
            "auto",
            "--g2pw-backend",
            "cuda",
            "--g2pw-cuda-memory-limit-mb",
            "1536",
        ]
    )
    config = ServerConfig.from_args(args)

    assert config.host == "0.0.0.0"
    assert config.port == 9999
    assert config.device == "cuda"
    assert config.half is False
    assert config.t2s_backend == "auto"
    assert config.g2pw_backend == "cuda"
    assert config.g2pw_cuda_memory_limit_mb == 1536


def test_environment_can_select_g2pw_backend(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    monkeypatch.setenv("THIN_TTS_G2PW_BACKEND", "auto")
    monkeypatch.setenv("THIN_TTS_G2PW_CUDA_MEMORY_LIMIT_MB", "2048")

    args = build_parser().parse_args(["--config", str(config_path)])
    config = ServerConfig.from_args(args)

    assert config.g2pw_backend == "auto"
    assert config.g2pw_cuda_memory_limit_mb == 2048
