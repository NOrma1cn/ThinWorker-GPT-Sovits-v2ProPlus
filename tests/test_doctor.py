from thin_tts.diagnostics import inspect_environment


def test_doctor_reports_missing_config_without_installing_anything(tmp_path):
    checks = inspect_environment(tmp_path / "missing.yaml")

    config = next(check for check in checks if check.name == "configuration")
    assert config.status == "error"
    assert "does not exist" in config.detail
    assert config.suggestions
