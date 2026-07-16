from pathlib import Path
import tomllib

import pytest
from pydantic import ValidationError

from thin_tts import __version__
from thin_tts.config import ServerConfig
from thin_tts.server import StreamRequest


def test_stream_request_defaults_to_validated_production_policy():
    request = StreamRequest(text="测试文本。")

    assert request.mode == 4
    assert request.rng_isolation is None
    assert request.cache_vits_encoded_text is None
    assert ServerConfig().rng_isolation is True
    assert ServerConfig().cache_vits_encoded_text is True


@pytest.mark.parametrize("mode", [0, 1, 5])
def test_stream_request_rejects_unsupported_modes(mode):
    with pytest.raises(ValidationError):
        StreamRequest(text="测试文本。", mode=mode)


def test_stream_request_rejects_removed_per_request_voice_fields():
    with pytest.raises(ValidationError):
        StreamRequest(
            text="测试文本。",
            ref_audio_path="reference.wav",
            ref_text="参考文本。",
        )


def test_package_metadata_versions_agree():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert __version__ == "1.1.0"
    assert metadata["project"]["version"] == __version__


def test_release_licenses_and_public_api_docs_are_present():
    root = Path(__file__).parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert (root / "LICENSE").is_file()
    assert (root / "THIRD_PARTY_NOTICES.md").is_file()
    assert (root / "LICENSES" / "Apache-2.0.txt").is_file()
    assert "### POST /tts" not in readme
    assert "| `ref_audio_path`" not in readme
    assert "python -m thin_tts tui" in readme
    assert "%APPDATA%\\thin-tts\\config.yaml" in readme
    assert "必须在 WSL 内安装 wheel 并启动 TUI" in readme
    assert "关闭 TUI 或终端不会停止服务" in readme


def test_current_release_notes_are_included_in_source_distributions():
    root = Path(__file__).parents[1]
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")

    assert (root / f"RELEASE_NOTES_v{__version__}.md").is_file()
    assert "include RELEASE_NOTES_v*.md" in manifest
