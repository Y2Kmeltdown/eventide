# eventide-core

Default Eventide module installed by `install.sh`. Provides:

- **playback server** — the Rust MJPEG playback server on TCP port 8084.
- **master-record widget** — a sidebar UI component that aggregates every installed module's `recording`-type component into one RECORDING panel with per-source controls and a RECORD ALL / STOP ALL button. RECORD ALL / STOP ALL calls the base backend's `POST /api/recording/trigger` (see `code/eventide.py`), which fans start/stop out to every recording source server-side — the same endpoint the scheduler below uses, so recordings can be triggered with no browser open.
- **scheduler** (`scheduler.py`, program `recording_scheduler`) — a small Flask service on its own `scheduler_api` socket that stores cron-triggered recording jobs and fires them in the background, POSTing to `/api/recording/trigger` when a job's schedule matches. Paired with two sidebar widgets:
  - **SCHEDULE RECORDING** (`schedule-form`) — create a job from a label, a standard 5-field cron expression, and a duration in seconds (0 = record until stopped).
  - **SCHEDULED RECORDINGS** (`schedule-table`) — lists jobs with their last-run result, an enable/disable checkbox, and a delete button.

This module exists so the base platform ships with no hardcoded components; future default services can be added here or in additional default modules.
