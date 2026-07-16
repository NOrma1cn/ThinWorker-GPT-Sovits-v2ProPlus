from pathlib import Path

from thin_tts.config_store import ConfigStore, apply_preset, new_config_document


def test_max_performance_preset_selects_strict_gpu_backends():
    document = new_config_document()

    apply_preset(document, "max-performance")

    server = document["server"]
    assert server["preset"] == "max-performance"
    assert server["device"] == "cuda"
    assert server["half"] is True
    assert server["t2s_backend"] == "triton"
    assert server["g2pw_backend"] == "cuda"
    assert server["g2pw_cuda_memory_limit_mb"] == 1536
    assert server["fallback_policy"] == "fail"


def test_config_store_preserves_comments_and_creates_backup(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("# operator note\nserver:\n  port: 9881\nweights: {}\n", encoding="utf-8")
    store = ConfigStore(path)
    document = store.load()
    document["server"]["port"] = 9988

    store.save(document)

    saved = path.read_text(encoding="utf-8")
    assert "# operator note" in saved
    assert "port: 9988" in saved
    assert Path(f"{path}.bak").read_text(encoding="utf-8").startswith("# operator note")


def test_new_config_uses_voice_profile_next_to_config(tmp_path):
    path = tmp_path / "config.yaml"

    document = new_config_document(path)

    assert document["weights"]["voice_profile"] == str(tmp_path / "voice-profile.pt")
