# Eventide Module Template

This directory is a **complete, valid, installable Eventide module**. Fork or
copy it as the starting point for your own module.

An Eventide module is a GitHub repository containing an
[`eventide-module.json`](eventide-module.json) manifest at its root. The
Eventide dashboard's MODULES tab installs the module onto the payload by
cloning the repo, running its declared dependency/build steps, and copying
its artifacts into place — which makes its `programs[]` available as node
types in the GRAPH tab's palette. Nothing actually runs under supervisord
until those programs are placed as nodes and the graph is submitted.

Full specification: see `docs/MODULES.md` in the main eventide repository.

> **This template targets the graph-based supervisor.** `arguments`,
> `sockets`, and `ui` are declared **per program**, not at the top of the
> manifest — see the examples below. Full design rationale:
> `docs/GRAPH_SUPERVISOR_PLAN.md` in the main eventide repository (the
> authoring reference, `docs/GRAPH_CONNECTIONS.md`, is still being written).

## What this example does

`hello_module.py` prints a timestamped heartbeat line every few seconds.
Installing this module makes `hello_module` available as a node type in the
GRAPH tab's palette; placing it and submitting the graph runs it as a
supervisord program, with its output landing in
`/var/log/supervisor/hello_module.log` (viewable from the dashboard's
GRAPH tab).

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
   `programs` list — one entry per node type you want in the GRAPH tab's
   palette.
4. Declare any system dependencies (`dependencies.apt` / `dependencies.pip`),
   build steps (`install.commands`, e.g. `cargo build --release`), and files to
   copy out of the repo (`install.artifacts`). These stay module-level.
5. Document the command-line arguments each program accepts in that
   program's own `arguments`, and any unix/tcp/http sockets it exposes in
   that program's own `sockets` — tagged `direction`/`transport`/`pattern`/
   `stream_kind` so the GRAPH tab knows what it can be wired to (see
   `docs/GRAPH_SUPERVISOR_PLAN.md` §6).
6. Push, then install from the dashboard: **MODULES → enter the repo URL →
   INSTALL** — or zip the folder and use **ZIP FILE** (drag & drop works
   too). Installing only makes the module's programs available in the GRAPH
   tab's palette; nothing runs until they're placed as nodes and the graph
   is submitted.

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

Declare every unix/tcp/http endpoint a program binds in **that program's
own** `sockets` list, and point its command at them with `{socket:<name>}`
— never hardcode a port or socket path in a command. Every socket needs
`direction` (`"output"` if this program owns/binds it, `"input"` if it
attaches to another node's output) and `transport` (`"unix"`, `"tcp"`, or
`"http"`); `unix`/`tcp` sockets also need `pattern` (`"stream"` or
`"request-reply"`), and a `"stream"` socket needs `stream_kind`
(`"irregular"`, `"regular"`, or `"framed"` — see
`docs/GRAPH_SUPERVISOR_PLAN.md` §6 for what each means and why they don't
mix). **Omit `port`/`path`**: eventide allocates a free TCP port or
generates a unix path when the graph is submitted and the node's edges are
resolved. Any `http` socket is reachable through the backend at
`/proxy/node/<node-id>/<socket>/<upstream path>` once its node is placed and
running — no nginx changes are ever needed for a module.

If your module writes recordings, set `recordings_subdir` and use
`{recordings_subdir}` in commands; the installer creates the directory, the
PLAYBACK tab gets an inner tab for it, and `/api/recordings/<subdir>` lists
its files:

```json
"recordings_subdir": "mycam",
"programs": [
  {
    "name": "mycam_mjpeg_server",
    "command": "{venv_python} {module_dir}/mjpeg.py --bind 0.0.0.0:{socket:mjpeg}",
    "arguments": [],
    "sockets": [
      { "name": "mjpeg", "direction": "output", "transport": "http",
        "pattern": "request-reply", "description": "MJPEG live stream" }
    ]
  },
  {
    "name": "mycam_datalogger",
    "command": "{venv_python} {module_dir}/record.py --output-dir {recordings_subdir}",
    "arguments": [],
    "sockets": []
  }
]
```

(The hello template itself binds no sockets and writes no recordings, so its
one program keeps `"sockets": []` and the manifest has no
`recordings_subdir`.)

## Running more than one instance

There's no manifest field for this any more: placing **multiple nodes** of
the same program template on the GRAPH tab's canvas *is* how you run more
than one instance (of a serial daemon, a second camera, …) — each node gets
its own argument values, its own allocated ports/unix paths, and (if the
program declares `recordings_subdir`) its own recordings directory. See
`docs/GRAPH_SUPERVISOR_PLAN.md` §5.

## Dashboard UI components

A program can advertise panels for the dashboard's MAIN tab in its own
optional `ui` array (also per-program, like `arguments`/`sockets`). The MAIN
tab's palette is built from the programs currently placed as nodes in the
active graph — installing the module alone doesn't add anything to it.
Components with `"default": true` are placed automatically when a node
using them is added to the graph. Widget types: `mjpeg` (centre stream
view), `form` (sidebar settings form), `telemetry` (polled readout),
`joystick` (RC pad), `table` (polled table with row actions), `map`
(Leaflet map). All widget traffic goes through the backend proxy, so
configs reference **that program's own `http` sockets, by name**, never by
port:

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
eventide repository ("Dashboard UI components (`ui`)") — the field shapes
are unchanged from before, only their placement (per-program) and the
palette's source (the active graph) are new.

## Local sanity check

```bash
python3 -m json.tool eventide-module.json > /dev/null && echo "manifest OK"
python3 hello_module.py --message test --interval 1   # Ctrl-C to stop
```
