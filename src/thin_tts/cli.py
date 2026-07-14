"""CLI entry point for thin-tts-server."""
import argparse
import sys


def build_parser():
    parser = argparse.ArgumentParser(
        prog="thin-tts-server",
        description="GPT-SoVITS v2ProPlus streaming TTS inference server",
    )
    parser.add_argument("--host", default=None, help="Bind host (default: YAML or 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: YAML or 9881)")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument(
        "--device", default=None, choices=["cuda", "cpu"], help="Torch device (default: YAML or cuda)"
    )
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--half", dest="half", action="store_true", help="Enable FP16")
    precision.add_argument("--no-half", dest="half", action="store_false", help="Disable FP16")
    parser.set_defaults(half=None)
    parser.add_argument(
        "--t2s-backend",
        choices=["auto", "sdpa", "triton"],
        default=None,
        help="T2S backend (default: auto; Linux CUDA may use Triton)",
    )
    parser.add_argument(
        "--g2pw-backend",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="G2PW ONNX Runtime backend (default: auto)",
    )
    parser.add_argument(
        "--g2pw-cuda-memory-limit-mb",
        type=int,
        default=None,
        help="Optional ONNX Runtime CUDA memory limit in MiB",
    )

    # Weight paths
    parser.add_argument("--t2s-weights", default=None, help="Path to T2S/GPT checkpoint (.ckpt)")
    parser.add_argument("--vits-weights", default=None, help="Path to VITS base model (.pth)")
    parser.add_argument("--vits-lora", default=None, help="Path to VITS LoRA adapter (.pth)")
    parser.add_argument("--bert-path", default=None, help="Path to chinese-roberta-wwm-ext-large directory")
    parser.add_argument("--hubert-path", default=None, help="Path to chinese-hubert-base directory")
    parser.add_argument("--sv-path", default=None, help="Path to SV model checkpoint (.ckpt)")
    parser.add_argument("--ref-audio", default=None, help="Path to reference audio (.wav)")
    parser.add_argument("--ref-text", default=None, help="Reference audio transcript text")
    return parser


def main():
    parser = build_parser()

    args = parser.parse_args()

    from thin_tts.config import ServerConfig
    from thin_tts.server import run

    cfg = ServerConfig.from_args(args)
    run(cfg)


if __name__ == "__main__":
    main()
