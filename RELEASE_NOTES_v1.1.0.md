# thin-tts-server v1.1.0

v1.1.0 focuses on informed, adjustable startup and reliable access to the
validated performance path.

## Highlights

- Adds a Textual TUI for configuration, startup progress, logs, diagnostics,
  requested-versus-active backend status, and explicit service controls.
- The TUI is a launcher, not the service owner. Servers start in a detached
  process, remain running after the TUI closes, and can be rediscovered later.
- Adds `serve`, `tui`, `doctor`, `status`, and `stop` CLI commands while
  retaining the legacy `thin-tts-server --config ...` invocation.
- Adds max-performance, balanced, and compatible presets plus `fail`, `warn`,
  and `allow` fallback policies.
- Emits structured startup-stage and startup-failure events for monitoring.
- Diagnostics are read-only. They explain causes and possible remedies but do
  not install Python packages, CUDA components, drivers, or system packages.
- Adds the previously missing TorchMetrics dependency and constrains
  Transformers to the validated major version.

## Detached-service behavior

Closing the TUI never stops the server. Use the TUI Stop action or
`thin-tts-server stop` when termination is intended. State and log files are
stored in the platform user-state directory.
