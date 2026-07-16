import asyncio

from thin_tts.config_store import ConfigStore, new_config_document
from thin_tts.diagnostics import DiagnosticCheck, FailureReport
from thin_tts.service_manager import ServiceSnapshot
import thin_tts.tui as tui_module
from thin_tts.tui import ThinTTSApp


class FakeManager:
    def __init__(self, snapshot=None, start_error=None):
        self.snapshot = snapshot or ServiceSnapshot(state="stopped")
        self.start_error = start_error
        self.started = []
        self.stop_calls = 0

    def status(self):
        return self.snapshot

    def start(self, config_path):
        if self.start_error:
            raise self.start_error
        self.started.append(str(config_path))

    def stop(self):
        self.stop_calls += 1
        return True

    def events(self):
        return []

    def logs(self, limit=200):
        return ""


def test_tui_saves_configuration_and_starts_detached_service(tmp_path):
    config_path = tmp_path / "config.yaml"
    ConfigStore(config_path).save(new_config_document(config_path))
    manager = FakeManager()
    app = ThinTTSApp(config_path=config_path, manager=manager, auto_refresh=False)

    async def exercise():
        async with app.run_test(size=(120, 42)) as pilot:
            app.query_one("#port").value = "9988"
            app.query_one("#vits-text-cache").value = False
            await pilot.click("#save-config")
            await pilot.click("#start-service")
            await pilot.pause()
            app.exit()

    asyncio.run(exercise())

    document = ConfigStore(config_path).load()
    assert document["server"]["port"] == 9988
    assert document["server"]["cache_vits_encoded_text"] is False
    assert manager.started == [str(config_path)]
    assert manager.stop_calls == 0


def test_tui_renders_failure_reason_and_suggestions(tmp_path):
    config_path = tmp_path / "config.yaml"
    ConfigStore(config_path).save(new_config_document(config_path))
    failure = FailureReport(
        stage="g2pw",
        reason="CUDA provider unavailable.",
        suggestions=("Check CUDA visibility.", "Select the CPU backend."),
        technical_details="technical details",
    )
    manager = FakeManager(ServiceSnapshot(state="failed", failure=failure))
    app = ThinTTSApp(config_path=config_path, manager=manager, auto_refresh=False)

    rendered = ""

    async def exercise():
        nonlocal rendered
        async with app.run_test(size=(120, 42)):
            await app.refresh_status()
            rendered = str(app.query_one("#failure-summary").render())

    asyncio.run(exercise())

    assert "CUDA provider unavailable" in rendered
    assert "Select the CPU backend" in rendered


def test_tui_preserves_custom_preset_and_indexed_cuda_device(tmp_path):
    config_path = tmp_path / "config.yaml"
    document = new_config_document(config_path)
    document["server"]["preset"] = "custom"
    document["server"]["device"] = "cuda:1"
    ConfigStore(config_path).save(document)
    app = ThinTTSApp(config_path=config_path, manager=FakeManager(), auto_refresh=False)

    async def exercise():
        async with app.run_test(size=(120, 42)):
            assert app.query_one("#preset").value == "custom"
            assert app.query_one("#device").value == "cuda:1"

    asyncio.run(exercise())


def test_tui_runs_read_only_diagnostics(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    ConfigStore(config_path).save(new_config_document(config_path))
    checks = [
        DiagnosticCheck(
            name="package:triton",
            status="error",
            detail="Python distribution is missing: triton",
            suggestions=("Install the documented version in the service environment.",),
        )
    ]
    monkeypatch.setattr(tui_module, "inspect_environment", lambda path: checks)
    app = ThinTTSApp(config_path=config_path, manager=FakeManager(), auto_refresh=False)

    async def exercise():
        async with app.run_test(size=(120, 42)) as pilot:
            app.query_one("#tabs").active = "diagnostics"
            await pilot.pause()
            await pilot.click("#run-diagnostics")
            await pilot.pause()
            assert app.query_one("#diagnostics-table").row_count == 1
            assert "1 个错误" in str(app.query_one("#diagnostic-summary").render())

    asyncio.run(exercise())


def test_tui_keeps_launcher_failure_visible_after_refresh(tmp_path):
    config_path = tmp_path / "config.yaml"
    ConfigStore(config_path).save(new_config_document(config_path))
    manager = FakeManager(start_error=RuntimeError("Address already in use"))
    app = ThinTTSApp(config_path=config_path, manager=manager, auto_refresh=False)

    async def exercise():
        async with app.run_test(size=(120, 42)) as pilot:
            await pilot.click("#start-service")
            await pilot.pause()
            rendered = str(app.query_one("#failure-summary").render())
            assert "port is already in use" in rendered

    asyncio.run(exercise())
