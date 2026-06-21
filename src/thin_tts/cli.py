"""CLI entry point for thin-tts-server."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="thin-tts-server",
        description="GPT-SoVITS v2ProPlus streaming TTS inference server",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9881, help="Bind port (default: 9881)")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Torch device")
    parser.add_argument("--half", action="store_true", default=True, help="Enable FP16 (default: True)")
    parser.add_argument("--no-half", dest="half", action="store_false", help="Disable FP16")

    # Weight paths
    parser.add_argument("--t2s-weights", default=None, help="Path to T2S/GPT checkpoint (.ckpt)")
    parser.add_argument("--vits-weights", default=None, help="Path to VITS base model (.pth)")
    parser.add_argument("--vits-lora", default=None, help="Path to VITS LoRA adapter (.pth)")
    parser.add_argument("--bert-path", default=None, help="Path to chinese-roberta-wwm-ext-large directory")
    parser.add_argument("--hubert-path", default=None, help="Path to chinese-hubert-base directory")
    parser.add_argument("--sv-path", default=None, help="Path to SV model checkpoint (.ckpt)")
    parser.add_argument("--ref-audio", default=None, help="Path to reference audio (.wav)")
    parser.add_argument("--ref-text", default=None, help="Reference audio transcript text")

    args = parser.parse_args()

    from thin_tts.config import ServerConfig
    from thin_tts.server import run

    cfg = ServerConfig.from_args(args)
    run(cfg)


if __name__ == "__main__":
    main()
