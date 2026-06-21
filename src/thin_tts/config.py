"""Server configuration and weight path resolution."""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 9881
    device: str = "cuda"
    half: bool = True

    # Weight paths
    t2s_weights: str = ""
    vits_weights: str = ""
    vits_lora: Optional[str] = None
    bert_path: str = ""
    hubert_path: str = ""
    sv_path: str = ""
    ref_audio: str = ""
    ref_text: str = ""

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

        def resolve(cli_val, env_key, yaml_key, default=""):
            if cli_val is not None:
                return cli_val
            env_val = os.environ.get(env_key)
            if env_val is not None:
                return env_val
            return yaml_key if yaml_key else default

        cfg = cls(
            host=args.host or server_cfg.get("host", "0.0.0.0"),
            port=args.port or server_cfg.get("port", 9881),
            device=args.device or server_cfg.get("device", "cuda"),
            half=args.half,
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
        )

        cfg._validate()
        return cfg

    def _validate(self):
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
            print(f"[ERROR] Missing required weight paths: {', '.join(missing)}", file=sys.stderr)
            print("Provide them via CLI args, env vars (THIN_TTS_*), or a YAML config file.", file=sys.stderr)
            sys.exit(1)

        for name, path in required.items():
            if not os.path.exists(path):
                print(f"[ERROR] {name} path does not exist: {path}", file=sys.stderr)
                sys.exit(1)
