# Eventide Module System

Eventide payloads are built from a **minimal base platform** plus **modules**
installed from GitHub repositories. A module is any component that runs as one
or more **supervisord programs** — camera dataloggers, MJPEG streamers, the
gimbal controller, and so on.

This document covers:

- [Architecture](#architecture)
- [The module manifest (`eventide-module.json`)](#the-module-manifest)
- [Network locations (ports & proxying)](#network-locations-ports--proxying)
- [Dashboard UI components (`ui`)](#dashboard-ui-components-ui)
- [Install lifecycle & error handling](#install-lifecycle--error-handling)
- [How supervisor config is generated](#how-supervisor-config-is-generated)
- [Backend API reference](#backend-api-reference)
- [Authoring a module](#authoring-a-module)
- [Porting the existing components](#porting-the-existing-components)
- [Migrating from a pre-module install](#migrating-from-a-pre-module-install)
- [Troubleshooting](#troubleshooting)

---

## Architecture

```
┌──────────────────────────── payload (e.g. tripwire, 192.168.30.2) ─┐
│                                                                    │
│  base platform (install.sh)                                        │
│    eventide.py  ──► /api/modules/*  (module manager, runs as root)│
│    supervisord   ──► /etc/supervisor/conf.d/                       │
│                        00-eventide-base.conf   (inet_http_server)  │
│                        playback.conf           (in-repo component) │
│                        module-<name>.conf      (one per module)    │
│    /usr/local/eventide/                                            │
│      modules.json            ← installed-modules registry          │
│      packages/<name>/        ← cloned module repos                 │
│      packages/<name>/.venv/  ← per-module Python virtualenv        │
│      code/                   ← copied module artifacts (binaries)  │
│      config/                 ← copied module config files          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
            ▲ cross-origin /api calls (CORS open)
┌───────────┴──────────── ground station ────────────────────────────┐
│  frontend.py serves dashboard.html → MODULES tab                   │
└────────────────────────────────────────────────────────────────────┘
```

The **base install** (`install.sh`) contains only what the platform needs to
boot and manage modules: OS configuration, Python + Flask, supervisord,
nginx, the dashboard backend, the watchdog/RTC/MAVProxy services, the playback
server (its source ships in this repo), and the Rust toolchain so Rust modules
can build on-device. Everything else is a module.

The **backend** (`eventide.py`) performs installs and reports status; the
**frontend** (MODULES tab) only displays what the backend tells it.

---

## The module manifest

Every module repository contains `eventide-module.json` at its root. It is the
single source of truth for what the module is, what it needs, and how it runs.

### Minimal example

```json
{
  "name": "hello-module",
  "version": "1.0.0",
  "description": "Prints a heartbeat every few seconds.",
  "programs": [
    {
      "name": "hello_module",
      "command": "/usr/bin/python3 {module_dir}/hello_module.py"
    }
  ]
}
```

A complete, working example ships in this repo under
[`module-template/`](../module-template/) — fork it to start a new module.

### Field reference

#### Top level

| Field         | Type   | Required | Description |
| ------------- | ------ | -------- | ----------- |
| `name`        | string | yes      | Module identifier. Must match `^[a-z0-9][a-z0-9-]*$` and be unique per payload. Used for the clone directory and the `module-<name>.conf` filename. |
| `version`     | string | yes      | Free-form version string, shown in the dashboard. |
| `description` | string | yes      | Short human-readable summary, shown in the dashboard. |
| `author`      | string | no       | Author/owner, shown in the dashboard. |

#### `dependencies` (optional)

Installed **before** any build steps, in the order `apt` → venv creation →
`requirements` → `pip` → `commands`.

| Field       | Type     | Description |
| ----------- | -------- | ----------- |
| `apt`       | string[] | Debian packages, installed system-wide with `apt-get install -y`. |
| `requirements` | string | Path (relative to the repo root) to a pip requirements file, installed into the module's venv with `pip install -r`. If omitted, `requirements.txt` at the repo root is used automatically when present. |
| `pip`       | string[] | Extra PyPI packages/specifiers, installed into the module's venv after `requirements`. |
| `system_site_packages` | boolean | Create the venv with `--system-site-packages` (default `true`) so apt-provided Python libraries such as `python3-picamera2` stay importable. Set to `false` for full isolation. |
| `commands`  | string[] | Arbitrary shell commands run (via `bash -c`) in the cloned repo root — after the venv exists, and with placeholders expanded, so they can use `{venv_python}`/`{venv_dir}`. Use for udev rules, firmware setup, etc. |

##### Python virtual environments

Every module that uses Python gets its **own virtual environment** at
`/usr/local/eventide/packages/<name>/.venv`, created at install time. A venv
is created whenever the module has a requirements file, declares
`dependencies.pip`, or references a `{venv_...}` placeholder in a program
command. The venv is removed together with the module directory on uninstall.

Supervisor programs must run the venv interpreter explicitly — use the
`{venv_python}` placeholder:

```json
"command": "{venv_python} {module_dir}/camera_app.py --output-dir {recordings_subdir}"
```

Because the venv is per module, two modules can pin conflicting versions of
the same package without breaking each other — and without touching the base
system's Python (no more `pip install --break-system-packages` for modules).

#### `install` (optional)

| Field       | Type   | Description |
| ----------- | ------ | ----------- |
| `commands`  | string[] | Build commands run in the repo root (e.g. `"cargo build --release"`). Each has a 30-minute timeout. |
| `artifacts` | object | Map of `source path (relative to repo root)` → `destination path`. Destinations must be absolute after expansion and support the same placeholders as program commands (e.g. `{install_dir}/my_binary`). Every source is verified to exist after the build; the copy preserves the file mode (`cp -p`). Parent directories of destinations are created automatically. |

#### `recordings_subdir` (optional)

String. If present, `<recordings_dir>/<recordings_subdir>` is created at
install time and the module appears as a **recording source**: the dashboard's
PLAYBACK tab gets an inner tab for it and `/api/recordings/<subdir>` lists its
files. Must be unique across installed modules. Program commands should
reference the directory through the `{recordings_subdir}` placeholder (below)
rather than hardcoding the path.

#### `arguments` (optional)

Command-line argument metadata — the module's configurable surface. The
dashboard displays these; defaults are substituted into program commands via
`{arg:<name>}` at install time.

| Field       | Type    | Required | Description |
| ----------- | ------- | -------- | ----------- |
| `name`      | string  | yes      | Argument identifier (`^[a-z0-9_]+$`), referenced as `{arg:name}`. |
| `flag`      | string  | yes      | CLI flag, e.g. `"--port"`. Informational. |
| `type`      | string  | yes      | `"str"`, `"int"`, or `"float"` (defaults are validated against it). |
| `default`   | any     | yes      | Value substituted for `{arg:name}` during install. |
| `required`  | boolean | no       | Informational — whether the program needs the flag. |
| `description` | string | no      | Shown in the dashboard. |

#### `sockets` (optional)

Network/IPC endpoints the module exposes. The backend uses them to wire up
network locations: any HTTP service behind a TCP socket is reachable through
the backend's generic proxy at `/proxy/<module>/<socket>/<upstream path>`,
and program commands reference them with the `{socket:<name>}` placeholder.

| Field         | Type   | Required | Description |
| ------------- | ------ | -------- | ----------- |
| `name`        | string | yes      | Socket identifier, unique within the module. |
| `type`        | string | yes      | `"tcp"` or `"unix"`. |
| `port`        | int    | no       | TCP port (1–65535). **Normally omitted — eventide allocates a free port from its pool (`--port-pool`, default 8100–8199) at install time.** Only set this to request a specific port; it must not collide with a socket of another installed module. |
| `path`        | string | unix only | Filesystem path of the UNIX socket. Its parent directory is created at install time. |
| `description` | string | no       | What the socket is for (e.g. `"MJPEG live stream"`). |

The allocated (or declared) port is stored in the registry and shown in the
MODULES tab. Because programs get the value through `{socket:<name>}`, a
module never needs to know the number in advance. The dashboard resolves
camera streams by convention: the module whose `recordings_subdir` matches
the camera key, preferring a TCP socket named `mjpeg`; the gimbal API is the
first TCP socket of the `gimbal-controller` module (falling back to any
module exposing the legacy port 5001).

#### `programs` (required, ≥1)

One entry per supervisord program the module provides. A camera module, for
example, typically declares two: a datalogger and an MJPEG server.

| Field          | Type    | Default                     | Description |
| -------------- | ------- | --------------------------- | ----------- |
| `name`         | string  | —                           | supervisord program name. Must match `^[a-z0-9][a-z0-9_-]*$` and be unique across **all** installed modules. |
| `command`      | string  | —                           | Full command line; supports placeholders (below). |
| `directory`    | string  | `{install_dir}`             | Working directory; supports placeholders. |
| `autostart`    | boolean | `true`                      | Start when supervisord starts. |
| `autorestart`  | boolean | `true`                      | Restart if the process exits. |
| `startretries` | int     | `10000`                     | Start attempts before giving up. |
| `priority`     | int     | `10`                        | supervisord start/stop ordering. |
| `user`         | string  | `"root"`                    | User the program runs as. |

### Command placeholders

Placeholders are expanded at install time — in program commands and
`directory` fields (when the module's supervisor config is generated) and in
`install.artifacts` destinations (when artifacts are copied):

| Placeholder          | Expands to                                   |
| -------------------- | -------------------------------------------- |
| `{install_dir}`      | `/usr/local/eventide/code`                   |
| `{config_dir}`       | `/usr/local/eventide/config`                 |
| `{module_dir}`       | `/usr/local/eventide/packages/<name>`        |
| `{recordings_dir}`   | The payload's recordings directory           |
| `{recordings_subdir}`| `{recordings_dir}/<recordings_subdir>` — requires the `recordings_subdir` field |
| `{venv_dir}`         | `/usr/local/eventide/packages/<name>/.venv`  |
| `{venv_python}`      | `{venv_dir}/bin/python3`                     |
| `{arg:<name>}`       | The argument's default value (string form)   |
| `{socket:<name>}`    | The socket's port (tcp) or path (unix)       |

---

## Network locations (ports & proxying)

Nothing about a module's network presence is hardcoded outside its manifest:

- **Ports are allocated by eventide.** A TCP socket without an explicit
  `port` gets one from the backend's pool (`--port-pool`, default
  `8100-8199`) at install time — the lowest port not used by another
  installed module and not already bound on the device. The chosen port is
  persisted in the registry entry's manifest and shown in the MODULES tab.
  An explicit `port` is still honoured when free (checked against other
  installed modules), so older manifests keep working.
- **nginx has no per-module locations.** `config/eventide.nginx` only fronts
  `eventide.py` (`location /`) and the base playback server (`/playback/`).
  Every HTTP service a module exposes is proxied by the backend itself:

  ```
  /proxy/<module>/<socket>/<upstream path>  →  http://127.0.0.1:<port>/<upstream path>
  ```

  Responses are streamed, so MJPEG works through it. Examples: a camera
  module's live stream is `/proxy/evk-datalogger/mjpeg/stream`, its settings
  API `/proxy/evk-datalogger/mjpeg/api/settings`, the gimbal API
  `/proxy/gimbal-controller/api/target`.
- **The dashboard resolves URLs from `/api/modules`.** Camera cells map the
  cam key (`evk`, `picam`, `ircam`) to the module whose `recordings_subdir`
  (or name) matches, preferring a TCP socket named `mjpeg`; the gimbal panel
  looks for the `gimbal-controller` module (falling back to any module
  exposing the legacy port 5001) and uses its first TCP socket. Follow those
  naming conventions and new modules light up the UI automatically.

---

## Dashboard UI components (`ui`)

The MAIN tab of the dashboard is a **modular workspace**: a left sidebar, a
right sidebar, and a tabbed centre workspace. Modules advertise the panels
they offer in an optional top-level `ui` array; the user places components
from the palette (＋ COMPONENTS button), drags them between regions, and the
layout persists in the browser per backend host. Components with
`"default": true` are placed automatically when the module is installed.

Every component's traffic goes through the backend proxy —
`/proxy/<module>/<socket>/<path>` — so `socket` in a `ui` config is always a
**socket name from the module's own `sockets` list** (tcp), never a port.

### Common fields

| Field    | Type    | Required | Description |
| -------- | ------- | -------- | ----------- |
| `id`     | string  | yes      | Component id, `^[a-z0-9][a-z0-9_-]*$`, unique within the module. |
| `type`   | string  | yes      | Widget type (below). |
| `title`  | string  | no       | Panel header text. |
| `region` | string  | no       | `"sidebar"` (default) or `"center"` — where `default` placement puts it. |
| `default`| boolean | no       | Auto-place on install (default `false`); otherwise palette-only. |

### Widget types

| Type       | Region  | Config (in addition to `socket`) |
| ---------- | ------- | -------------------------------- |
| `mjpeg`    | center  | `path` — MJPEG stream path, e.g. `"/stream"`. Renders with offline/retry handling. |
| `form`     | sidebar | `get`, `put`, `submit_label?`, `fields[]`. GET populates, PUT applies. Field: `{key, label?, kind: number\|slider\|toggle\|text\|select, min?, max?, step?, get?, put?, options?}` — per-field `get`/`put` overrides let one form span several endpoints. |
| `telemetry`| sidebar | `get`, `interval?` (ms), `rows[]` — polled readout. Row: `{label, path, fmt?}`; `path` is a dot-path into the JSON (`buffer.bytes`). |
| `joystick` | sidebar | `put`, `telemetry_get?`, `paths?: {x, y}`, `fields?: {x, y, frame}` — two-axis RC pad seeded from a telemetry poll. |
| `table`    | sidebar | `get`, `interval?`, `columns[]` (`{label, path, fmt?}`), `row_action?: {label, method, path, key}`, `stop_action?: {label, method, path}` — polled table with a per-row action button (e.g. ADS-B track/stop). |
| `map`      | center  | `track?: {socket, get, interval?, lat, lon, heading?, gimbal?, frame?}`, `adsb?: {socket, get, interval?, lat, lon, label?, key?}` — Leaflet map with optional device/track markers. Without bindings it's a plain map. |

`fmt` is one of the dashboard's named formatters: `int`, `f1`, `f2`, `f6`
(decimal places), `m_km` (metres → m/km).

### Example (evk-datalogger)

```json
"ui": [
  { "id": "live", "type": "mjpeg", "title": "EVK4 LIVE", "region": "center",
    "default": true, "socket": "mjpeg", "path": "/stream" },
  { "id": "stream-settings", "type": "form", "title": "EVK4 STREAM",
    "region": "sidebar", "default": true, "socket": "mjpeg",
    "get": "/api/settings", "put": "/api/settings",
    "fields": [
      {"key": "quality", "kind": "slider", "min": 1, "max": 100},
      {"key": "out_width", "kind": "number", "min": 1},
      {"key": "out_height", "kind": "number", "min": 1},
      {"key": "streaming", "kind": "toggle",
       "get": "/api/streaming", "put": "/api/streaming"}
    ] },
  { "id": "biases", "type": "form", "title": "EVK4 BIASES",
    "region": "sidebar", "default": true, "socket": "http_api",
    "get": "/api/biases", "put": "/api/biases",
    "fields": [
      {"key": "diff_on", "kind": "number", "min": 0, "max": 255},
      {"key": "diff_off", "kind": "number", "min": 0, "max": 255}
    ] }
]
```

---

## Install lifecycle & error handling

Installing is a **background job** on the backend. The job moves through these
statuses:

```
pending → cloning|extracting → deps → building → artifacts → configuring → verifying → done
                                                                              ↘ failed
```

| Status        | What happens |
| ------------- | ------------ |
| `cloning`     | Repo cloned to a staging dir (`packages/.staging-<job>`). HTTPS is tried first; for `github.com` URLs a SSH (`git@github.com:…`) retry follows automatically. The manifest is read and fully validated here — including name/program/socket conflicts with already-installed modules. |
| `extracting`  | Zip installs only: the uploaded zip is stored under `packages/.uploads/` and extracted into staging (zip-slip paths are rejected). The manifest must sit at the zip root or in a single top-level folder (as GitHub's "Download ZIP" produces). Validation then proceeds exactly as for `cloning`. |
| `deps`        | `dependencies.apt`, then the module venv is created and `requirements`/`pip` are installed into it, then `dependencies.commands`. |
| `building`    | `install.commands` run in the repo root. |
| `artifacts`   | Every `install.artifacts` source is checked for existence, then copied to its destination. `recordings_subdir` is created. |
| `configuring` | `/etc/supervisor/conf.d/module-<name>.conf` is rendered and written; `supervisorctl reread && supervisorctl update` is run. |
| `verifying`   | `supervisorctl status` is polled (up to ~10 s) for the module's programs. |
| `done`        | The module is recorded in `/usr/local/eventide/modules.json`. |
| `failed`      | See rollback below. |

**Rollback.** Any hard failure (clone error, invalid manifest, dependency or
build command exiting non-zero, missing artifact, supervisor apply error)
removes everything the job created: copied artifacts, the conf.d file
(followed by `reread`/`update`), the staging/clone directory, and any partial
registry entry. The job ends `failed` with the failing command's output in its
log.

**Runtime state vs install state.** A program that installs cleanly but then
fails to start (e.g. its camera is not connected on a bench install) does
**not** fail the install — it is reported in the job's `warnings` list and is
visible in the dashboard's SUPERVISOR tab. Hard errors in the install steps
themselves always fail the job.

**Concurrency.** One install at a time; concurrent requests are rejected with
`409 Conflict`.

**Updating a module** is uninstall + reinstall (no in-place upgrade yet).

---

## How supervisor config is generated

The base config `/etc/supervisor/conf.d/00-eventide-base.conf` (installed by
`install.sh`) holds only `[supervisord]` and `[inet_http_server]`. Each module
gets its own generated file, `/etc/supervisor/conf.d/module-<name>.conf`:

```ini
; Generated by eventide module manager from <repo url> — do not edit by hand.
[program:pi_camera_datalogger]
command=/usr/bin/python3 /usr/local/eventide/code/camera_app.py --output-dir /home/tripwire/recordings/picam/ --config /usr/local/eventide/config/camera_config.json
directory=/usr/local/eventide/code
autostart=true
autorestart=true
startretries=10000
priority=10
user=root
stdout_logfile=/var/log/supervisor/%(program_name)s.log
```

The file is written atomically (temp file + rename), then applied with
`supervisorctl reread && supervisorctl update`. Uninstalling a module simply
deletes its conf file and re-applies — no other module's config is touched,
and a broken module can never corrupt the rest of the system.

---

## Backend API reference

All endpoints are served by `eventide.py` under `/api/modules` (through nginx
on the payload, like the rest of `/api`). Errors return
`{"error": "<message>"}` with a 4xx/5xx status.

### `GET /api/modules`

List installed modules, merged with live supervisor status.

```json
{
  "modules": [
    {
      "name": "hello-module",
      "version": "1.0.0",
      "description": "…",
      "author": "…",
      "repo_url": "https://github.com/you/hello-module",
      "installed_at": "2026-07-24T12:00:00+00:00",
      "recordings_subdir": null,
      "arguments": [ … ],
      "sockets":   [ … ],
      "programs": [
        {"name": "hello_module", "status": "RUNNING", "description": "pid 1234, uptime 0:03:12"}
      ]
    }
  ]
}
```

### `GET /api/modules/<name>`

Full detail for one installed module (registry entry including the raw
manifest). `404` if not installed.

### `POST /api/modules/install`

Body: `{"repo_url": "https://github.com/you/module", "ref": "main"}` (`ref`
optional — branch or tag). Starts a background install job.

- `202 Accepted` → `{"job_id": "…", "status": "pending"}`
- `400` missing/invalid `repo_url` · `409` another install is already running

### `POST /api/modules/install-upload`

Installs a module from an uploaded zip file instead of a git clone. The
request is `multipart/form-data` with the archive in the `file` field. The
zip must contain `eventide-module.json` at its root or inside a single
top-level folder (what GitHub's **Download ZIP** produces). Maximum upload
size: 100 MB.

- `202 Accepted` → `{"job_id": "…", "status": "pending"}` — poll the job as
  usual; the first status is `extracting` instead of `cloning`
- `400` no file / not a `.zip` · `409` another install is already running ·
  `413` zip too large

### `GET /api/modules/jobs/<job_id>`

Job status for polling:

```json
{
  "id": "…", "status": "building",
  "repo_url": "…", "module": null,
  "created_at": "…", "finished_at": null,
  "log": ["$ git clone …", "…"],
  "error": null, "warnings": []
}
```

`status` ∈ `pending, cloning, deps, building, artifacts, configuring,
verifying, done, failed`. When `done`, `module` is the module name and
`warnings` lists any programs not yet RUNNING. When `failed`, `error` says why.

### `POST /api/modules/<name>/uninstall`

Stops the module's programs (best-effort), removes its conf file, re-applies
supervisor, deletes copied artifacts and the cloned repo, and drops the
registry entry. `404` if not installed.

### `GET /api/recordings`

Lists the **recording sources** — one per installed module that declares
`recordings_subdir`:

```json
{"sources": [{"name": "evk", "module": "evk-datalogger"},
             {"name": "picam", "module": "picam-datalogger"}]}
```

`GET /api/recordings/<source>` lists that source's files;
`GET /api/recordings/<source>/<file>/download` downloads one. Sources no
installed module declares return `404` — the dashboard's PLAYBACK tab builds
its inner tabs from exactly this list.

### `/proxy/<module>/<socket>/…`

Generic proxy to the HTTP service behind a module's TCP socket:
`/proxy/<module>/<socket>/<upstream path>` is forwarded (streamed) to
`http://127.0.0.1:<allocated port>/<upstream path>`. `404` when the module or
socket isn't installed, `502` when the module's server is down. This is how
the dashboard reaches camera streams, per-camera settings, and the gimbal
API — nginx carries no per-module locations.

---

## Authoring a module

1. Start from [`module-template/`](../module-template/).
2. Write the manifest. Validate it locally:
   `python3 -m json.tool eventide-module.json > /dev/null`.
3. Keep programs **foreground** processes — supervisord manages daemonisation,
   restarts, and logging. Log to stdout/stderr; it lands in
   `/var/log/supervisor/<program>.log`.
4. Put everything the program needs at runtime either in `install.artifacts`
   (copied to a stable location) or reference it inside `{module_dir}` — the
   clone is not removed after install.
5. Declare every TCP/UNIX socket you bind in `sockets` — reference them from
   program commands with `{socket:<name>}` and let eventide allocate the TCP
   ports (omit `port` unless you genuinely need a fixed one).
6. Push to GitHub and install from the dashboard's MODULES tab — either by
   repo URL, or by uploading/dragging a zip of the repository (GitHub's
   "Download ZIP" layout works as-is).

---

## Porting the existing components

The pre-module `install.sh` cloned and installed four repositories. Each maps
to a module as follows (the removed install.sh lines become manifest fields):

### `Y2Kmeltdown/evk_datalogger` → `evk-datalogger` ✅ converted

The repo ships `eventide-module.json` — a two-service module:

- `dependencies.apt`: `build-essential`, `pkg-config`, `libusb-1.0-0-dev`
  (the crate links `rusb`/`libusb1-sys`)
- `requirements.txt` (into the venv): `neuromorphic_drivers==0.17.0`,
  `faery==0.7.0`
- `dependencies.commands`: udev rules, from inside the venv —
  `{venv_dir}/bin/neuromorphic-drivers-install-udev-rules` and
  `{venv_python} {venv_dir}/lib/python3.*/site-packages/neuromorphic_drivers/udev.py`
- `install.commands`: `cargo build --release`
- `install.artifacts`: `target/release/evk_datalogger` and
  `target/release/viewfinder` → `/usr/local/eventide/code/`
- `recordings_subdir`: `evk`
- `programs`: `event_based_camera` (priority 1) and `evk_mjpeg_server`
  (priority 2, viewfinder bound via `--bind 0.0.0.0:{socket:mjpeg}`)
- `sockets`: unix `/tmp/evk4_events.sock` + `/tmp/evk4_triggers.sock` and
  tcp `mjpeg` (MJPEG live stream — no `port`; eventide allocates one)

### `Y2Kmeltdown/picam_datalogger` → `picam-datalogger` ✅ converted

The repo ships `eventide-module.json` — a two-service module:

- `dependencies.apt`: `python3-picamera2`, `ffmpeg`
- keep `dependencies.system_site_packages` at its default `true` so the
  apt-provided `picamera2` stays importable from the module venv
- `requirements.txt` (into the venv): `aiohttp`, `numpy`,
  `opencv-python-headless`
- `recordings_subdir`: `picam`
- `programs`: `pi_camera_datalogger` (priority 1) and `pi_mjpeg_server`
  (priority 2), both run straight from `{module_dir}` with `{venv_python}` —
  no `install.artifacts` needed for pure-Python modules
- `sockets`: unix `/tmp/picam_frames.sock` (inter-service frame socket —
  referenced as `{socket:frames}`) + tcp `mjpeg` (MJPEG; port allocated)

### `ericltb15/aravis-ir` → `ircam-datalogger` ✅ converted

The repo ships `eventide-module.json` — a two-service module:

- `dependencies.apt`: `libaravis-dev` (plus its private pkg-config
  requirements `libusb-1.0-0-dev`, `libxml2-dev`, `zlib1g-dev`, which the
  distro package does not pull in), the meson/ninja toolchain, and the
  GStreamer dev files + runtime plugins
- `requirements.txt` (into the venv): `aiohttp`, `numpy`,
  `opencv-python-headless`
- `install.commands`: `meson setup build --buildtype=release`,
  `ninja -C build`
- `install.artifacts`: `build/ircam` and `scripts/ir_mjpeg.py` →
  `{install_dir}`
- `recordings_subdir`: `ircam`
- `programs`: `infrared_camera` (priority 1 — records segmented MP4 to
  `{recordings_subdir}`, serves raw 16-bit frames on `{socket:frames}`) and
  `ir_mjpeg_server` (priority 2 — `{venv_python} {install_dir}/ir_mjpeg.py`)
- `sockets`: unix `/tmp/irstream.sock` (inter-service frame socket) + tcp
  `mjpeg` (MJPEG; port allocated)

### `j-vanarsdale/tripwire-gimbal-point` → `gimbal-controller`

- `dependencies.pip`: `telnetlib3`, `pymavlink==2.4.49`, `pyserial==3.5`
- `install.artifacts`: `GIMBAL_POINT_API.py`, `adsb.py` → `{install_dir}`
- `recordings_subdir`: `telemetry`
- `programs`: `gimbal_controller` — port the full command line from the old
  `config/supervisor.conf` git history, with `{recordings_subdir}` in place of
  the old `SEDPLACEHOLDER`
- `sockets`: tcp `api` at explicit `port: 5001` (gimbal API — kept fixed for
  external clients; the dashboard also finds it as the first tcp socket of
  `gimbal-controller`)

The old pinned versions in `config/requirements.txt` (`opencv-python-headless`,
`smbus2`, `spidev`, `gpiozero`, …) belonged to these components — move them
into the corresponding module manifests, not the base install.

---

## Migrating from a pre-module install

On a payload that already runs the old monolithic setup:

1. Remove the stale supervisor configs:
   `sudo rm /etc/supervisor/conf.d/supervisor.conf /etc/supervisor/conf.d/supervisord.conf`
   (whichever exists), then `sudo supervisorctl reread && sudo supervisorctl update`.
2. Run the new `install.sh` (safe to re-run; it installs the base only).
3. Reinstall each component as a module from the dashboard's MODULES tab.

Fresh installs need no migration — `install.sh` removes the stale files
itself.

---

## Troubleshooting

- **Job failed during `cloning`** — check the repo URL is correct and public;
  private repos need working SSH keys on the payload (the installer retries
  `git@github.com:…` automatically).
- **Job failed during `building`** — read the job log in the MODULES tab; the
  failing command's stdout/stderr is captured there.
- **Install `done` but a program is not RUNNING** — check the job `warnings`
  and the SUPERVISOR tab; hardware-dependent programs legitimately fail on
  bench installs without their devices attached.
- **"socket port 8081 already used by module X"** — two modules explicitly
  request the same TCP port; drop the `port` field from one manifest and let
  eventide allocate a free one from its pool.
- **"no free port in pool 8100-8199 for socket …"** — the allocation pool is
  exhausted (or everything in it is bound); widen it with the backend's
  `--port-pool start-end` flag.
- **Supervisor didn't pick up a change** — `sudo supervisorctl reread &&
  sudo supervisorctl update`, then check `/etc/supervisor/conf.d/` for the
  generated `module-<name>.conf`.
- **Full install log** — the job log is kept in memory by `eventide.py`;
  journald has the backend's own output: `journalctl -u eventide.service`.
