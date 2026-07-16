"""Command-line entry points for serving, launching, and diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from thin_tts.config_store import default_config_path
from thin_tts.events import startup_events


COMMANDS = {"serve", "tui", "doctor", "status", "stop"}


def build_parser():
    """Build the backward-compatible server argument parser."""
    parser = argparse.ArgumentParser(
        prog="thin-tts-server",
        description="GPT-SoVITS v2ProPlus streaming TTS inference server",
    )
    parser.add_argument("--preset", default=None, choices=["custom", "max-performance", "balanced", "compatible"])
    parser.add_argument("--fallback-policy", default=None, choices=["fail", "warn", "allow"])
    parser.add_argument(
        "--rng-isolation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable request-local T2S/VITS random generators",
    )
    parser.add_argument(
        "--cache-vits-encoded-text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reuse target-text conditioning across streaming VITS chunks",
    )
    parser.add_argument("--host", default=None, help="Bind host (default: YAML or 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: YAML or 9881)")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="Torch device: cpu, cuda, or cuda:<index> (default: YAML or cuda)",
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
    parser.add_argument("--t2s-weights", default=None, help="Path to T2S/GPT checkpoint (.ckpt)")
    parser.add_argument("--vits-weights", default=None, help="Path to VITS base model (.pth)")
    parser.add_argument("--vits-lora", default=None, help="Path to VITS LoRA adapter (.pth)")
    parser.add_argument("--bert-path", default=None, help="Path to chinese-roberta-wwm-ext-large directory")
    parser.add_argument("--hubert-path", default=None, help="Path to chinese-hubert-base directory")
    parser.add_argument("--sv-path", default=None, help="Path to SV model checkpoint (.ckpt)")
    parser.add_argument("--ref-audio", default=None, help="Path to reference audio (.wav)")
    parser.add_argument("--ref-text", default=None, help="Reference audio transcript text")
    parser.add_argument("--voice-profile", default=None, help="Optional compiled fixed-voice profile path (.pt)")
    return parser


def serve(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from thin_tts.config import ConfigurationError, ServerConfig

    try:
        with startup_events.stage("configuration"):
            cfg = ServerConfig.from_args(args)
    except ConfigurationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    try:
        from thin_tts.server import run

        run(cfg)
    except KeyboardInterrupt:
        return 130
    except BaseException as exc:
        startup_events.emit(
            "thin_tts_startup_failure",
            stage="process",
            status="failed",
            reason=str(exc) or type(exc).__name__,
            exception_type=type(exc).__name__,
        )
        raise
    return 0


def launch_tui(config_path=None) -> None:
    from thin_tts.tui import run_tui

    run_tui(config_path)


def service_status():
    from thin_tts.service_manager import ServiceManager

    return ServiceManager().status()


def _doctor(config_path: str) -> int:
    from thin_tts.diagnostics import inspect_environment

    checks = inspect_environment(config_path)
    for check in checks:
        print(json.dumps(asdict(check), ensure_ascii=False))
    return 1 if any(check.status == "error" for check in checks) else 0


def _root_help() -> None:
    print(
        "usage: thin-tts-server {tui|serve|doctor|status|stop} ...\n\n"
        "commands:\n"
        "  tui       Open the configuration and monitoring launcher\n"
        "  serve     Run the server in the foreground\n"
        "  doctor    Run read-only environment checks\n"
        "  status    Report the detached service status\n"
        "  stop      Stop the detached service\n\n"
        "Interactive terminals may run `thin-tts-server` with no arguments to open the TUI.\n"
        "The TUI launches a detached service; closing it does not stop the server.\n"
        "Legacy server options such as `--config` remain supported."
    )


def main(argv=None, *, interactive=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not arguments:
        if interactive:
            launch_tui(None)
            return 0
        _root_help()
        return 2
    if arguments[0] in {"-h", "--help"}:
        _root_help()
        return 0
    command = arguments[0]
    if command not in COMMANDS:
        return serve(arguments)
    if command == "serve":
        return serve(arguments[1:])
    if command == "tui":
        parser = argparse.ArgumentParser(prog="thin-tts-server tui")
        parser.add_argument("--config", default=None)
        args = parser.parse_args(arguments[1:])
        launch_tui(args.config)
        return 0
    if command == "doctor":
        parser = argparse.ArgumentParser(prog="thin-tts-server doctor")
        parser.add_argument("--config", default=str(default_config_path()))
        args = parser.parse_args(arguments[1:])
        return _doctor(args.config)
    if command == "status":
        snapshot = service_status()
        print(
            json.dumps(
                {"state": snapshot.state, "pid": snapshot.pid, "health": snapshot.health},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if snapshot.process_alive else 1
    if command == "stop":
        from thin_tts.service_manager import ServiceManager

        stopped = ServiceManager().stop()
        print("Service stopped." if stopped else "No managed service is running.")
        return 0 if stopped else 1
    raise AssertionError(f"Unhandled command: {command}")
