import io
import json

import thin_tts.cli as cli


def test_tui_subcommand_dispatches_without_importing_server(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(cli, "launch_tui", lambda config: called.append(config))

    result = cli.main(["tui", "--config", str(tmp_path / "config.yaml")])

    assert result == 0
    assert called == [str(tmp_path / "config.yaml")]


def test_legacy_config_invocation_routes_to_serve(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "serve", lambda argv: called.append(argv) or 0)

    result = cli.main(["--config", "config.yaml"])

    assert result == 0
    assert called == [["--config", "config.yaml"]]


def test_no_arguments_open_tui_only_for_interactive_terminal(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "launch_tui", lambda config: called.append(config))

    assert cli.main([], interactive=True) == 0
    assert called == [None]
    assert cli.main([], interactive=False) == 2


def test_root_help_lists_commands_without_emitting_startup_events(capsys):
    result = cli.main(["--help"], interactive=False)

    output = capsys.readouterr().out
    assert result == 0
    assert "tui" in output
    assert "serve" in output
    assert "doctor" in output
    assert "thin_tts_startup" not in output


def test_status_command_reports_detached_service(monkeypatch, capsys):
    snapshot = type(
        "Snapshot",
        (),
        {"state": "ready", "pid": 42, "health": {"status": "ok"}, "process_alive": True},
    )()
    monkeypatch.setattr(cli, "service_status", lambda: snapshot)

    result = cli.main(["status"])

    assert result == 0
    assert '"state": "ready"' in capsys.readouterr().out


def test_invalid_configuration_emits_specific_failure_without_traceback(tmp_path, capsys, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("server:\n  preset: impossible\n", encoding="utf-8")
    event_output = io.StringIO()
    monkeypatch.setattr(cli.startup_events, "output", event_output)

    result = cli.serve(["--config", str(config_path)])

    captured = capsys.readouterr()
    events = [json.loads(line) for line in event_output.getvalue().splitlines()]
    assert result == 2
    assert events[-1]["event"] == "thin_tts_startup_failure"
    assert "Invalid preset" in events[-1]["reason"]
    assert "Invalid preset" in captured.err
    assert "Traceback" not in captured.err
