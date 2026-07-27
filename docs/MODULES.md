# Eventide Module System

Eventide payloads are built from a **minimal base platform** plus **modules**
installed from GitHub repositories. A module is any component that runs as one
or more **supervisord programs** — camera dataloggers, MJPEG streamers, the
gimbal controller, and so on.

This document covers:

- [Architecture](#architecture)
- [The module manifest (`eventide-module.json`)](#the-module-manifest)
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
│    dashboard.py  ──► /api/modules/*  (module manager, runs as root)│
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

The **backend** (`dashboard.py`) performs installs and reports status; the
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
"command": "{venv_python} {module_dir}/camera_app.py --output-dir {recordings_dir}/picam/"
```

Because the venv is per module, two modules can pin conflicting versions of
the same package without breaking each other — and without touching the base
system's Python (no more `pip install --break-system-packages` for modules).

#### `install` (optional)

| Field       | Type   | Description |
| ----------- | ------ | ----------- |
| `commands`  | string[] | Build commands run in the repo root (e.g. `"cargo build --release"`). Each has a 10-minute timeout. |
| `artifacts` | object | Map of `source path (relative to repo root)` → `absolute destination path`. Every source is verified to exist after the build; the copy preserves the file mode (`cp -p`). Parent directories of destinations are created automatically. |

#### `recordings_subdir` (optional)

String. If present, `<recordings_dir>/<recordings_subdir>` is created at
install time. Use it for modules that write recordings.

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

Network/IPC endpoints the module exposes — advertised to the dashboard so it
knows the module's capabilities (e.g. which port carries an MJPEG stream).

| Field         | Type   | Required | Description |
| ------------- | ------ | -------- | ----------- |
| `name`        | string | yes      | Socket identifier, unique within the module. |
| `type`        | string | yes      | `"tcp"` or `"unix"`. |
| `port`        | int    | tcp only | TCP port (1–65535). Must not collide with a socket of another installed module. |
| `path`        | string | unix only | Filesystem path of the UNIX socket. |
| `description` | string | no       | What the socket is for (e.g. `"MJPEG live stream"`). |

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

Placeholders are expanded when the module's supervisor config is generated
(install time):

| Placeholder        | Expands to                                   |
| ------------------ | -------------------------------------------- |
| `{install_dir}`    | `/usr/local/eventide/code`                   |
| `{config_dir}`     | `/usr/local/eventide/config`                 |
| `{module_dir}`     | `/usr/local/eventide/packages/<name>`        |
| `{recordings_dir}` | The payload's recordings directory           |
| `{venv_dir}`       | `/usr/local/eventide/packages/<name>/.venv`  |
| `{venv_python}`    | `{venv_dir}/bin/python3`                     |
| `{arg:<name>}`     | The argument's default value (string form)   |

---

## Install lifecycle & error handling

Installing is a **background job** on the backend. The job moves through these
statuses:

```
pending → cloning → deps → building → artifacts → configuring → verifying → done
                                                                              ↘ failed
```

| Status        | What happens |
| ------------- | ------------ |
| `cloning`     | Repo cloned to a staging dir (`packages/.staging-<job>`). HTTPS is tried first; for `github.com` URLs a SSH (`git@github.com:…`) retry follows automatically. The manifest is read and fully validated here — including name/program/socket conflicts with already-installed modules. |
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

All endpoints are served by `dashboard.py` under `/api/modules` (through nginx
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
5. Declare every TCP port you bind in `sockets` so the installer can catch
   port conflicts between modules.
6. Push to GitHub and install from the dashboard's MODULES tab.

---

## Porting the existing components

The pre-module `install.sh` cloned and installed four repositories. Each maps
to a module as follows (the removed install.sh lines become manifest fields):

### `Y2Kmeltdown/evk_datalogger` → `evk-datalogger`

- `dependencies.pip`: `neuromorphic_drivers==0.17.0`, `faery==0.7.0`
- `dependencies.commands`: udev rules, now from inside the venv —
  `{venv_dir}/bin/neuromorphic-drivers-install-udev-rules` and
  `{venv_python} {venv_dir}/lib/python3.*/site-packages/neuromorphic_drivers/udev.py`
  (adjust the `python3.*` glob to the payload's Python version)
- `install.commands`: `cargo build --release`
- `install.artifacts`: `target/release/evk_datalogger` and
  `target/release/viewfinder` → `/usr/local/eventide/code/`
- `recordings_subdir`: `evk`
- `programs`: `event_based_camera` (`{install_dir}/evk_datalogger --output-dir
  {recordings_dir}/evk/`) and `evk_mjpeg_server` (`{install_dir}/viewfinder
  --bind 0.0.0.0:8081 --quality 15`)
- `sockets`: tcp `8081` (MJPEG live stream)

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
- `sockets`: unix `/tmp/picam_frames.sock` (inter-service frame socket) +
  tcp `8082` (MJPEG)

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
  `{recordings_dir}/ircam/`, serves raw 16-bit frames on
  `/tmp/irstream.sock`) and `ir_mjpeg_server` (priority 2 —
  `{venv_python} {install_dir}/ir_mjpeg.py`)
- `sockets`: unix `/tmp/irstream.sock` (inter-service frame socket) + tcp
  `8083` (MJPEG)

### `j-vanarsdale/tripwire-gimbal-point` → `gimbal-controller`

- `dependencies.pip`: `telnetlib3`, `pymavlink==2.4.49`, `pyserial==3.5`
- `install.artifacts`: `GIMBAL_POINT_API.py`, `adsb.py` → `{install_dir}`
- `recordings_subdir`: `telemetry`
- `programs`: `gimbal_controller` — port the full command line from the old
  `config/supervisor.conf` git history, with `{recordings_dir}` in place of
  the old `SEDPLACEHOLDER`
- `sockets`: tcp `5001` (gimbal API)

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
- **"socket port 8081 already used by module X"** — two modules declare the
  same TCP port; change one module's manifest.
- **Supervisor didn't pick up a change** — `sudo supervisorctl reread &&
  sudo supervisorctl update`, then check `/etc/supervisor/conf.d/` for the
  generated `module-<name>.conf`.
- **Full install log** — the job log is kept in memory by `dashboard.py`;
  journald has the backend's own output: `journalctl -u dashboard.service`.
