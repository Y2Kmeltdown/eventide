# Eventide Module Template

This directory is a **complete, valid, installable Eventide module**. Fork or
copy it as the starting point for your own module.

An Eventide module is a GitHub repository containing an
[`eventide-module.json`](eventide-module.json) manifest at its root. The
Eventide dashboard (MODULES tab) installs the module onto the payload by
cloning the repo, running its declared dependency/build steps, copying its
artifacts into place, and registering its programs with supervisord.

Full specification: see `docs/MODULES.md` in the main eventide repository.

## What this example does

`hello_module.py` prints a timestamped heartbeat line every few seconds. It is
installed as a supervisord program named `hello_module`, and its output shows
up in `/var/log/supervisor/hello_module.log` (viewable from the dashboard's
SUPERVISOR tab).

## Repository layout

```
eventide-module.json   ← required manifest (module metadata + supervisor programs)
requirements.txt       ← Python deps, installed into the module's own venv
hello_module.py        ← the program itself (any language/binary works)
README.md              ← this file
```

## Python virtual environments

If the repo has a `requirements.txt` (or the manifest declares
`dependencies.requirements` / `dependencies.pip`, or a program command
references `{venv_python}`), the installer creates a dedicated virtual
environment at `/usr/local/eventide/packages/<name>/.venv` and installs the
packages there — never system-wide. Run your program with the venv's
interpreter via the `{venv_python}` placeholder, as this template does. The
venv is created with `--system-site-packages` by default so apt-provided
Python libraries (e.g. `python3-picamera2`) remain visible; set
`dependencies.system_site_packages` to `false` for full isolation.

## Using the template

1. Copy this directory into a new GitHub repository (or fork it).
2. Rename the module: edit `name` in `eventide-module.json`
   (`^[a-z0-9][a-z0-9-]*$`, must be unique per payload).
3. Replace `hello_module.py` with your own program(s) and update the
   `programs` list — one entry per supervisord service.
4. Declare any system dependencies (`dependencies.apt` / `dependencies.pip`),
   build steps (`install.commands`, e.g. `cargo build --release`), and files to
   copy out of the repo (`install.artifacts`).
5. Document the command-line arguments your program accepts in `arguments` and
   any TCP/UNIX sockets it exposes in `sockets`, so the dashboard knows the
   module's capabilities.
6. Push, then install from the dashboard: **MODULES → enter the repo URL →
   INSTALL** — or zip the folder and use **ZIP FILE** (drag & drop works too).

## Placeholders available in commands and artifact destinations

| Placeholder          | Expands to                                    |
| -------------------- | --------------------------------------------- |
| `{install_dir}`      | `/usr/local/eventide/code`                    |
| `{config_dir}`       | `/usr/local/eventide/config`                  |
| `{module_dir}`       | `/usr/local/eventide/packages/<module name>`  |
| `{recordings_dir}`   | The payload recordings directory              |
| `{recordings_subdir}`| `{recordings_dir}/<recordings_subdir>` (needs the manifest field) |
| `{venv_dir}`         | `/usr/local/eventide/packages/<module name>/.venv` |
| `{venv_python}`      | `{venv_dir}/bin/python3`                      |
| `{arg:<name>}`       | Default value of the declared argument        |
| `{socket:<name>}`    | The socket's port (tcp) or path (unix)        |

## Sockets, ports and recordings

Declare every TCP/UNIX endpoint your programs bind in `sockets`, and point
your commands at them with `{socket:<name>}` — never hardcode a port or
socket path in a command. **Omit `port` for TCP sockets**: eventide
allocates a free one at install time (from `--port-pool`, default
`8100-8199`), stores it in the registry, and shows it in the MODULES tab.
Any HTTP service behind a TCP socket is reachable through the backend at
`/proxy/<module>/<socket>/<upstream path>` — no nginx changes are ever
needed for a module.

If your module writes recordings, set `recordings_subdir` and use
`{recordings_subdir}` in commands; the installer creates the directory, the
PLAYBACK tab gets an inner tab for it, and `/api/recordings/<subdir>` lists
its files:

```json
"recordings_subdir": "mycam",
"sockets": [
  {"name": "mjpeg", "type": "tcp", "description": "MJPEG live stream"}
],
"programs": [
  {
    "name": "mycam_mjpeg_server",
    "command": "{venv_python} {module_dir}/mjpeg.py --bind 0.0.0.0:{socket:mjpeg}"
  },
  {
    "name": "mycam_datalogger",
    "command": "{venv_python} {module_dir}/record.py --output-dir {recordings_subdir}"
  }
]
```

(The hello template itself binds no sockets and writes no recordings, so its
manifest keeps `"sockets": []` and no `recordings_subdir`.)

## Instanceable modules

If your module wraps hardware that can exist more than once on a payload — a
serial device, a second camera — declare an `instance` key and eventide runs
**one copy of the module's programs per instance** the operator creates in the
MODULES tab (an ADD INSTANCE row appears on the module card; the first
instance is auto-created at install from the argument's `default`):

```json
"instance": { "argument": "port", "label": "Serial port" },
"arguments": [
  { "name": "port", "flag": "--port", "type": "str", "default": "/dev/ttyS1" }
]
```

`instance.argument` names the CLI argument (declared in `arguments`, type
`str` or `int`) that identifies a copy. Per instance, eventide renders the
programs with that instance's argument values (`{arg:port}` → `/dev/ttyS2`),
allocates fresh TCP ports from its pool, generates unix socket paths
(`/tmp/eventide-<name>-<iid>-<socket>.sock`), and — when `recordings_subdir`
is declared — gives the instance its own recordings directory
(`<subdir>-<iid>`). Supervisor programs are named `<program>-<iid>`; each
instance shows up in the MODULES tab with its own services, EDIT and REMOVE
buttons. Rules: don't pin `path` on unix sockets or `port` on tcp sockets
(the copies would collide) — always reference them via `{socket:<name>}`.

Modules without an `instance` key are not instanceable and behave exactly as
before. Full details and migration steps: `docs/MODULES.md` in the main
eventide repository ("Instanceable modules", "Migrating a manifest to
instanceable").

## Dashboard UI components

Modules can advertise panels for the dashboard's MAIN tab in an optional
`ui` array. The dashboard shows them in the component palette; components
with `"default": true` are placed automatically on install. Widget types:
`mjpeg` (centre stream view), `form` (sidebar settings form), `telemetry`
(polled readout), `joystick` (RC pad), `table` (polled table with row
actions), `map` (Leaflet map). All widget traffic goes through the backend
proxy, so configs reference sockets **by name**, never by port:

```json
"ui": [
  { "id": "live", "type": "mjpeg", "title": "MYCAM LIVE", "region": "center",
    "default": true, "socket": "mjpeg", "path": "/stream" },
  { "id": "stream-settings", "type": "form", "title": "MYCAM STREAM",
    "region": "sidebar", "default": true, "socket": "mjpeg",
    "get": "/api/settings", "put": "/api/settings",
    "fields": [
      {"key": "quality", "kind": "slider", "min": 1, "max": 100},
      {"key": "streaming", "kind": "toggle",
       "get": "/api/streaming", "put": "/api/streaming"}
    ] }
]
```

Full schema and per-type config reference: `docs/MODULES.md` in the main
eventide repository ("Dashboard UI components (`ui`)").

## Local sanity check

```bash
python3 -m json.tool eventide-module.json > /dev/null && echo "manifest OK"
python3 hello_module.py --message test --interval 1   # Ctrl-C to stop
```
