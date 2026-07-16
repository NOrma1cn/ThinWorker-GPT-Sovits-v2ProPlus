"""Read-only diagnostics and actionable failure explanations."""

from __future__ import annotations

import re
import importlib.metadata
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FailureReport:
    stage: str
    reason: str
    suggestions: tuple[str, ...]
    technical_details: str


@dataclass(frozen=True)
class DiagnosticCheck:
    name: str
    status: str
    detail: str
    suggestions: tuple[str, ...] = ()


def inspect_environment(config_path: str | Path) -> list[DiagnosticCheck]:
    """Perform read-only checks. This function never installs or repairs anything."""
    path = Path(config_path).expanduser()
    checks = []
    if not path.is_file():
        checks.append(
            DiagnosticCheck(
                name="configuration",
                status="error",
                detail=f"Configuration file does not exist: {path}",
                suggestions=("Create or select a configuration file in the TUI.",),
            )
        )
        return checks
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        checks.append(
            DiagnosticCheck(
                name="configuration",
                status="error",
                detail=f"Configuration could not be read: {exc}",
                suggestions=("Correct the YAML syntax, then run doctor again.",),
            )
        )
        return checks
    checks.append(DiagnosticCheck("configuration", "ok", str(path)))
    server = document.get("server", {})
    weights = document.get("weights", {})
    for field in ("t2s_weights", "vits_weights", "bert_path", "hubert_path", "sv_path", "ref_audio"):
        value = weights.get(field)
        if not value:
            checks.append(
                DiagnosticCheck(
                    f"path:{field}",
                    "error",
                    f"Required path is empty: {field}",
                    ("Set the path in Configuration.",),
                )
            )
        elif not Path(value).expanduser().exists():
            checks.append(
                DiagnosticCheck(
                    f"path:{field}",
                    "error",
                    f"Configured path does not exist: {value}",
                    ("Correct the path for the operating system that will run the server.",),
                )
            )
    for distribution in ("torch", "torchaudio", "torchmetrics", "transformers", "onnxruntime-gpu"):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            checks.append(
                DiagnosticCheck(
                    f"package:{distribution}",
                    "error",
                    f"Python distribution is missing: {distribution}",
                    ("Follow the documented dependency instructions for this runtime.",),
                )
            )
        else:
            checks.append(DiagnosticCheck(f"package:{distribution}", "ok", version))
    if server.get("t2s_backend") == "triton" and platform.system() != "Linux":
        checks.append(
            DiagnosticCheck(
                "triton-platform",
                "error",
                "Triton was requested, but the current operating system is not Linux.",
                ("Run the production preset in WSL2/Linux, or select SDPA.",),
            )
        )
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        checks.append(DiagnosticCheck("gpu", "warning", f"nvidia-smi unavailable: {exc}"))
    else:
        status = "ok" if gpu.returncode == 0 else "warning"
        checks.append(DiagnosticCheck("gpu", status, gpu.stdout.strip() or gpu.stderr.strip()))
    return checks


def advise_failure(stage: str, details: str) -> FailureReport:
    """Translate a technical error into cause and user-controlled remedies."""
    technical = details.strip() or "Unknown error"
    module_match = re.search(r"No module named ['\"]([^'\"]+)['\"]", technical)
    if module_match:
        module = module_match.group(1)
        return FailureReport(
            stage=stage,
            reason=f"Python module '{module}' is not installed.",
            suggestions=(
                "Follow the project documentation for the verified dependency versions.",
                f"Confirm that '{module}' is installed in the same Python environment used to launch the server.",
                "Restart the launcher after changing the Python environment.",
            ),
            technical_details=technical,
        )
    lowered = technical.lower()
    if "address already in use" in lowered or "only one usage of each socket" in lowered:
        return FailureReport(
            stage=stage,
            reason="The configured server port is already in use.",
            suggestions=(
                "Stop the process currently using the port, or choose another port in Configuration.",
                "Check whether another thin-tts-server instance is already running.",
            ),
            technical_details=technical,
        )
    if "cudaexecutionprovider" in lowered or "cuda provider" in lowered:
        return FailureReport(
            stage=stage,
            reason="ONNX Runtime could not activate the CUDA execution provider.",
            suggestions=(
                "Verify that the documented onnxruntime-gpu build is present in this Python environment.",
                "Check CUDA and cuDNN visibility, then restart the launcher.",
                "Select the CPU G2PW backend when GPU memory or CUDA support is unavailable.",
            ),
            technical_details=technical,
        )
    if "triton" in lowered:
        return FailureReport(
            stage=stage,
            reason="The Triton backend could not be activated.",
            suggestions=(
                "Use WSL2 or Linux with a CUDA-enabled PyTorch and the documented Triton version.",
                "Select SDPA in Configuration when Triton is not supported.",
                "Review the technical log for the exact capability probe failure.",
            ),
            technical_details=technical,
        )
    if "cuda out of memory" in lowered or "outofmemoryerror" in lowered:
        return FailureReport(
            stage=stage,
            reason="The GPU did not have enough free memory to start the selected configuration.",
            suggestions=(
                "Close other GPU workloads and retry.",
                "Select the balanced preset to move G2PW to CPU.",
                "Disable optional GPU features or use a GPU with more memory.",
            ),
            technical_details=technical,
        )
    if "does not exist" in lowered or "no such file" in lowered:
        return FailureReport(
            stage=stage,
            reason="A configured model or reference file could not be found.",
            suggestions=(
                "Open Configuration and correct the reported path.",
                "Confirm that Windows drive paths are translated to /mnt paths when running in WSL.",
            ),
            technical_details=technical,
        )
    return FailureReport(
        stage=stage,
        reason=technical.splitlines()[-1],
        suggestions=(
            "Review the technical log for the failing startup stage.",
            "Check the configuration and environment, then retry the start operation.",
        ),
        technical_details=technical,
    )
