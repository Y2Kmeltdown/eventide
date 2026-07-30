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

## Local sanity check

```bash
python3 -m json.tool eventide-module.json > /dev/null && echo "manifest OK"
python3 hello_module.py --message test --interval 1   # Ctrl-C to stop
```
