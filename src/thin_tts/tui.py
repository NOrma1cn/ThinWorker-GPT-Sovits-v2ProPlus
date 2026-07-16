"""Textual control panel for configuring and launching thin-tts-server."""

from __future__ import annotations

import asyncio
from pathlib import Path

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    DataTable,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)

from thin_tts.config_store import ConfigStore, PRESETS, apply_preset
from thin_tts.diagnostics import DiagnosticCheck, FailureReport, advise_failure, inspect_environment
from thin_tts.service_manager import ServiceManager, ServiceSnapshot


def _text(value) -> str:
    return "" if value is None else str(value)


class ThinTTSApp(App):
    """A launcher whose lifetime is deliberately independent from the service."""

    TITLE = "Thin TTS Server"
    SUB_TITLE = "配置与启动控制面板"
    BINDINGS = [Binding("q", "quit", "退出", show=False)]

    CSS = """
    Screen {
        background: #11161c;
        color: #e7edf3;
    }
    Header {
        background: #17212b;
        color: #e7edf3;
    }
    TabbedContent {
        height: 1fr;
    }
    TabPane {
        padding: 1 2;
    }
    .summary {
        height: 3;
        padding: 1 2;
        border: round #3d556c;
        background: #17212b;
    }
    .section-title {
        margin-top: 1;
        color: #7dd3fc;
        text-style: bold;
    }
    .form-row {
        height: 3;
        align-vertical: middle;
    }
    .form-row Label {
        width: 26;
        color: #a9b7c6;
    }
    .form-row Input, .form-row Select {
        width: 1fr;
    }
    .form-row Switch {
        width: 12;
    }
    #command-bar {
        dock: bottom;
        height: 4;
        padding: 0 2;
        background: #17212b;
        align-horizontal: right;
        align-vertical: middle;
    }
    #command-bar Button {
        margin-left: 1;
    }
    Button.-primary {
        background: #0f766e;
    }
    Button.-warning {
        background: #a16207;
    }
    Button.-danger {
        background: #9f1239;
    }
    #failure-summary {
        min-height: 7;
        padding: 1 2;
        border: round #9f1239;
    }
    #startup-table, #optimization-table {
        height: 1fr;
        min-height: 10;
    }
    #technical-log, #diagnostics-table {
        height: 1fr;
        min-height: 12;
        border: round #3d556c;
    }
    """

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        manager: ServiceManager | None = None,
        auto_refresh: bool = True,
    ) -> None:
        super().__init__()
        self.store = ConfigStore(config_path)
        self.document = self.store.load()
        self.manager = manager or ServiceManager()
        self.poll_enabled = auto_refresh
        self.launcher_failure: FailureReport | None = None

    def compose(self) -> ComposeResult:
        server = self.document.setdefault("server", {})
        weights = self.document.setdefault("weights", {})
        yield Header(show_clock=True)
        with TabbedContent(initial="dashboard", id="tabs"):
            with TabPane("概览", id="dashboard"):
                yield Static("服务状态：正在读取", id="status-summary", classes="summary")
                yield Static("请求配置与实际状态", classes="section-title")
                yield DataTable(id="optimization-table", cursor_type="none")
                yield Static("最近失败", classes="section-title")
                yield Static("无", id="failure-summary")
            with TabPane("配置", id="configuration"):
                with VerticalScroll():
                    yield Static(f"配置文件：{self.store.path}", classes="summary")
                    yield Static("运行策略", classes="section-title")
                    yield from self._select_row(
                        "性能预设",
                        "preset",
                        [
                            ("自定义", "custom"),
                            ("最大性能", "max-performance"),
                            ("均衡", "balanced"),
                            ("兼容", "compatible"),
                        ],
                        server.get("preset", "custom"),
                    )
                    yield Button("应用预设", id="apply-preset", variant="default")
                    yield from self._input_row("监听地址", "host", server.get("host", "0.0.0.0"))
                    yield from self._input_row("监听端口", "port", server.get("port", 9881), type="integer")
                    yield from self._input_row("计算设备", "device", server.get("device", "cuda"))
                    yield from self._switch_row("FP16", "half", bool(server.get("half", True)))
                    yield from self._select_row(
                        "T2S 后端",
                        "t2s-backend",
                        [("自动", "auto"), ("Triton", "triton"), ("SDPA", "sdpa")],
                        server.get("t2s_backend", "auto"),
                    )
                    yield from self._select_row(
                        "G2PW 后端",
                        "g2pw-backend",
                        [("自动", "auto"), ("CUDA", "cuda"), ("CPU", "cpu")],
                        server.get("g2pw_backend", "auto"),
                    )
                    yield from self._input_row(
                        "G2PW 显存上限 MiB",
                        "g2pw-memory",
                        server.get("g2pw_cuda_memory_limit_mb"),
                        type="integer",
                    )
                    yield from self._select_row(
                        "回退策略",
                        "fallback-policy",
                        [("失败即停", "fail"), ("警告", "warn"), ("允许", "allow")],
                        server.get("fallback_policy", "warn"),
                    )
                    yield from self._switch_row(
                        "请求级 RNG 隔离", "rng-isolation", bool(server.get("rng_isolation", True))
                    )
                    yield from self._switch_row(
                        "VITS 文本编码缓存",
                        "vits-text-cache",
                        bool(server.get("cache_vits_encoded_text", True)),
                    )
                    yield Static("模型与音色", classes="section-title")
                    for label, field in (
                        ("T2S 权重", "t2s_weights"),
                        ("VITS 权重", "vits_weights"),
                        ("VITS LoRA", "vits_lora"),
                        ("BERT 目录", "bert_path"),
                        ("HuBERT 目录", "hubert_path"),
                        ("说话人模型", "sv_path"),
                        ("参考音频", "ref_audio"),
                        ("参考文本", "ref_text"),
                        ("音色 Profile", "voice_profile"),
                    ):
                        yield from self._input_row(label, field.replace("_", "-"), weights.get(field, ""))
            with TabPane("启动过程", id="startup"):
                yield DataTable(id="startup-table", cursor_type="none")
                yield Static("技术日志", classes="section-title")
                yield RichLog(id="technical-log", highlight=True, markup=False, wrap=True)
            with TabPane("诊断", id="diagnostics"):
                yield Static(
                    "此面板只检查并解释环境，不会执行 pip、apt、驱动或 CUDA 安装。",
                    classes="summary",
                )
                yield Button("运行只读诊断", id="run-diagnostics", variant="default")
                yield Static("尚未运行诊断。", id="diagnostic-summary")
                yield DataTable(id="diagnostics-table", cursor_type="none")
        with Horizontal(id="command-bar"):
            yield Button("保存配置", id="save-config", variant="default")
            yield Button("▶ 启动", id="start-service", classes="-primary")
            yield Button("■ 停止", id="stop-service", classes="-danger")
            yield Button("↻ 重启", id="restart-service", classes="-warning")
            yield Button("刷新", id="refresh-service", variant="default")

    @staticmethod
    def _input_row(label: str, field_id: str, value, *, type: str = "text"):
        with Horizontal(classes="form-row"):
            yield Label(label)
            yield Input(value=_text(value), id=field_id, type=type)

    @staticmethod
    def _select_row(label: str, field_id: str, options, value):
        with Horizontal(classes="form-row"):
            yield Label(label)
            yield Select(options, value=value, allow_blank=False, id=field_id)

    @staticmethod
    def _switch_row(label: str, field_id: str, value: bool):
        with Horizontal(classes="form-row"):
            yield Label(label)
            yield Switch(value=value, id=field_id)

    async def on_mount(self) -> None:
        optimization = self.query_one("#optimization-table", DataTable)
        optimization.add_columns("项目", "请求", "实际", "状态")
        startup = self.query_one("#startup-table", DataTable)
        startup.add_columns("阶段", "状态", "耗时", "说明")
        diagnostics = self.query_one("#diagnostics-table", DataTable)
        diagnostics.add_columns("检查项", "状态", "详情", "可能的解决办法")
        await self.refresh_status()
        if self.poll_enabled:
            self.set_interval(1.0, self.refresh_status)

    async def refresh_status(self) -> None:
        snapshot = await asyncio.to_thread(self.manager.status)
        events = await asyncio.to_thread(self.manager.events)
        logs = await asyncio.to_thread(self.manager.logs, 100)
        self._render_snapshot(snapshot)
        self._render_events(events)
        log_widget = self.query_one("#technical-log", RichLog)
        log_widget.clear()
        if logs:
            log_widget.write(logs)

    def _render_snapshot(self, snapshot: ServiceSnapshot) -> None:
        labels = {
            "stopped": "已停止",
            "starting": "启动中",
            "ready": "运行中",
            "failed": "启动失败",
        }
        self.query_one("#status-summary", Static).update(
            f"服务状态：{labels.get(snapshot.state, snapshot.state)}"
            + (f"    PID：{snapshot.pid}" if snapshot.pid else "")
        )
        table = self.query_one("#optimization-table", DataTable)
        table.clear()
        server = self.document.get("server", {})
        health = snapshot.health or {}
        t2s = health.get("t2s_backend", {})
        g2pw = health.get("g2pw_backend", {})
        profile = health.get("voice_profile", {})
        table.add_row(
            "T2S",
            _text(server.get("t2s_backend", "auto")),
            _text(t2s.get("active", "-")),
            _text(t2s.get("state", snapshot.state)),
        )
        table.add_row(
            "G2PW",
            _text(server.get("g2pw_backend", "auto")),
            _text(g2pw.get("active", "-")),
            _text(g2pw.get("state", snapshot.state)),
        )
        table.add_row(
            "音色 Profile",
            "启用" if self.document.get("weights", {}).get("voice_profile") else "关闭",
            _text(profile.get("state", "-")),
            "编码器已卸载" if profile.get("voice_encoders_loaded") is False else "-",
        )
        table.add_row("FP16", "启用" if server.get("half", True) else "关闭", "-", "配置")
        self._render_failure(snapshot.failure or self.launcher_failure)

    def _render_failure(self, failure: FailureReport | None) -> None:
        widget = self.query_one("#failure-summary", Static)
        if failure is None:
            widget.update("无")
            return
        suggestions = "\n".join(f"{index}. {item}" for index, item in enumerate(failure.suggestions, 1))
        content = (
            f"阶段：{escape(failure.stage)}\n"
            f"原因：{escape(failure.reason)}\n\n"
            f"可能的解决办法：\n{escape(suggestions)}"
        )
        widget.update(content)

    def _render_diagnostics(self, checks: list[DiagnosticCheck]) -> None:
        table = self.query_one("#diagnostics-table", DataTable)
        table.clear()
        errors = sum(check.status == "error" for check in checks)
        warnings = sum(check.status == "warning" for check in checks)
        self.query_one("#diagnostic-summary", Static).update(
            f"诊断完成：{errors} 个错误，{warnings} 个警告。"
        )
        for check in checks:
            table.add_row(
                check.name,
                check.status,
                check.detail,
                "\n".join(check.suggestions),
            )

    def _render_events(self, events: list[dict]) -> None:
        table = self.query_one("#startup-table", DataTable)
        table.clear()
        latest = {}
        for event in events:
            if event.get("event") == "thin_tts_startup_stage":
                latest[event.get("stage", "unknown")] = event
            elif event.get("event") == "thin_tts_startup_failure":
                latest[event.get("stage", "unknown")] = event
        for stage, event in latest.items():
            elapsed = event.get("elapsed_ms")
            table.add_row(
                stage,
                _text(event.get("status", "-")),
                f"{elapsed} ms" if elapsed is not None else "-",
                _text(event.get("reason", "")),
            )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        action = event.button.id
        try:
            self.launcher_failure = None
            if action == "save-config":
                self._save_config()
                self.notify("配置已保存")
            elif action == "apply-preset":
                self._apply_selected_preset()
                self.notify("预设已应用，保存后生效")
            elif action == "start-service":
                self._save_config()
                await asyncio.to_thread(self.manager.start, self.store.path)
                self.notify("服务已在后台启动，关闭面板不会停止服务")
            elif action == "stop-service":
                stopped = await asyncio.to_thread(self.manager.stop)
                self.notify("服务已停止" if stopped else "没有正在运行的受管服务")
            elif action == "restart-service":
                await asyncio.to_thread(self.manager.stop)
                self._save_config()
                await asyncio.to_thread(self.manager.start, self.store.path)
                self.notify("服务已在后台重启")
            elif action == "refresh-service":
                pass
            elif action == "run-diagnostics":
                checks = await asyncio.to_thread(inspect_environment, self.store.path)
                self._render_diagnostics(checks)
        except Exception as exc:
            self.launcher_failure = advise_failure("launcher", str(exc))
        await self.refresh_status()

    def _apply_selected_preset(self) -> None:
        preset = self.query_one("#preset", Select).value
        if preset == "custom":
            return
        if preset not in PRESETS:
            raise ValueError("请选择有效的性能预设")
        apply_preset(self.document, str(preset))
        server = self.document["server"]
        self.query_one("#device", Input).value = server["device"]
        self.query_one("#half", Switch).value = server["half"]
        self.query_one("#t2s-backend", Select).value = server["t2s_backend"]
        self.query_one("#g2pw-backend", Select).value = server["g2pw_backend"]
        self.query_one("#g2pw-memory", Input).value = _text(server["g2pw_cuda_memory_limit_mb"])
        self.query_one("#fallback-policy", Select).value = server["fallback_policy"]
        self.query_one("#rng-isolation", Switch).value = server["rng_isolation"]
        self.query_one("#vits-text-cache", Switch).value = server["cache_vits_encoded_text"]

    def _save_config(self) -> None:
        server = self.document.setdefault("server", {})
        weights = self.document.setdefault("weights", {})
        server.update(
            {
                "preset": str(self.query_one("#preset", Select).value),
                "host": self.query_one("#host", Input).value.strip(),
                "port": int(self.query_one("#port", Input).value),
                "device": self.query_one("#device", Input).value.strip(),
                "half": self.query_one("#half", Switch).value,
                "t2s_backend": str(self.query_one("#t2s-backend", Select).value),
                "g2pw_backend": str(self.query_one("#g2pw-backend", Select).value),
                "g2pw_cuda_memory_limit_mb": self._optional_int("#g2pw-memory"),
                "fallback_policy": str(self.query_one("#fallback-policy", Select).value),
                "rng_isolation": self.query_one("#rng-isolation", Switch).value,
                "cache_vits_encoded_text": self.query_one("#vits-text-cache", Switch).value,
            }
        )
        for field in (
            "t2s_weights",
            "vits_weights",
            "vits_lora",
            "bert_path",
            "hubert_path",
            "sv_path",
            "ref_audio",
            "ref_text",
            "voice_profile",
        ):
            weights[field] = self.query_one(f"#{field.replace('_', '-')}", Input).value.strip()
        self.store.save(self.document)

    def _optional_int(self, selector: str):
        value = self.query_one(selector, Input).value.strip()
        return int(value) if value else None


def run_tui(config_path: str | Path | None = None) -> None:
    ThinTTSApp(config_path=config_path).run()
