# eventide-core

Default Eventide module installed by `install.sh`. Provides:

- **playback server** — the Rust MJPEG playback server on TCP port 8084.
- **master-record widget** — a sidebar UI component that aggregates every installed module's `recording`-type component into one RECORDING panel with per-source controls and a RECORD ALL / STOP ALL button.

This module exists so the base platform ships with no hardcoded components; future default services can be added here or in additional default modules.
