from thin_tts.diagnostics import advise_failure


def test_missing_module_advice_never_claims_to_install_it():
    report = advise_failure("pipeline_load", "No module named 'torchmetrics'")

    assert report.reason == "Python module 'torchmetrics' is not installed."
    assert any("project documentation" in item for item in report.suggestions)
    assert all("Installing" not in item for item in report.suggestions)


def test_address_in_use_advice_points_to_port_configuration():
    report = advise_failure("server_listen", "[Errno 98] address already in use")

    assert "port is already in use" in report.reason.lower()
    assert any("port" in item.lower() for item in report.suggestions)
