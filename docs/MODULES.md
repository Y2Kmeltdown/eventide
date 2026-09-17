# Eventide Module System

Eventide payloads are built from a **minimal base platform** plus **modules**
installed from GitHub repositories, plus a **graph** you build in the
dashboard's GRAPH tab that decides what actually runs. A module is a
GitHub repo (or a zip upload) that declares one or more **program
templates** — camera dataloggers, MJPEG streamers, the gimbal controller,
and so on — which become node types you can place on the graph. Installing
a module makes its programs available; it doesn't start anything by
itself.

This document covers:

- [Architecture](#architecture)
- [The module manifest (`eventide-module.json`)](#the-module-manifest)
- [Command placeholders](#command-placeholders)
- [The graph: nodes, edges, and how supervisor config is generated](#the-graph-nodes-edges-and-how-supervisor-config-is-generated)
- [Network locations (ports & proxying)](#network-locations-ports--proxying)
- [Dashboard UI components (`ui`)](#dashboard-ui-components-ui)
- [Install lifecycle & error handling](#install-lifecycle--error-handling)
- [Backend API reference](#backend-api-reference)
- [Authoring a module](#authoring-a-module)
- [Porting the existing components](#porting-the-existing-components)
- [Migrating from a pre-module install](#migrating-from-a-pre-module-install)
- [Troubleshooting](#troubleshooting)

For the full reasoning behind the socket tag taxonomy
(`direction`/`transport`/`pattern`/`stream_kind`/`capped`/`count_arg`) and a
worked example, see
[docs/GRAPH_CONNECTIONS.md](GRAPH_CONNECTIONS.md) — this document covers
the rest of the manifest and the platform around it. For the full design
rationale behind the graph itself, see
[docs/GRAPH_SUPERVISOR_PLAN.md](GRAPH_SUPERVISOR_PLAN.md).

---

## Architecture

```
┌──────────────────────────── payload (e.g. tripwire, 192.168.30.2) ─┐
│                                                                    │
│  base platform (install.sh)                                        │
│    eventide.py  ──► /api/modules/*  (module manager, runs as root)│
│               ──► /api/graph/*   (graph compiler)                 │
│    supervisord   ──► /etc/supervisor/conf.d/                       │
│                        00-eventide-base.conf   (inet_http_server)  │
│                        eventide-graph.conf     (generated on       │
│                                                  every graph submit)│
│    /usr/local/eventide/                                            │
│      modules.json            ← installed-modules registry          │
│      graph.json               ← active/draft graph + resolved addrs│
│      packages/<name>/        ← cloned module repos                 │
│      packages/<name>/.venv/  ← per-module Python virtualenv        │
│      code/                   ← copied module artifacts (binaries)  │
│      config/                 ← copied module config files          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
            ▲ cross-origin /api calls (CORS open)
┌───────────┴──────────── ground station ────────────────────────────┐
│  frontend.py serves dashboard.html → MODULES + GRAPH tabs           │
└────────────────────────────────────────────────────────────────────┘
```

The **base install** (`install.sh`) contains only what the platform needs to
boot and manage modules and graphs: OS configuration, Python + Flask,
supervisord, nginx, the dashboard backend, the watchdog/RTC/MAVProxy
services, the playback server (its source ships in this repo, run as a
graph node provided by the built-in `eventide-core` module), and the Rust
toolchain so Rust modules can build on-device. Everything else is a
module, and nothing a module provides runs until it's placed as a node in
the graph and the graph is submitted.

The **backend** (`eventide.py`) performs installs, compiles submitted
graphs into a running supervisor config, and reports status; the
**frontend** (MODULES + GRAPH tabs) only displays what the backend tells
it.

---

## The module manifest

Every module repository contains `eventide-module.json` at its root. It is
the single source of truth for what the module is, what it needs, and what
program templates it contributes to the GRAPH tab's palette.

### Minimal example

```json
{
  "name": "hello-module",
  "version": "1.0.0",
  "description": "Prints a heartbeat every few seconds.",
  "programs": [
    {
      "name": "hello_module",
      "command": "/usr/bin/python3 {module_dir}/hello_module.py",
      "arguments": [],
      "sockets": []
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
| `name`        | string | yes      | Module identifier. Must match `^[a-z0-9][a-z0-9-]*$` and be unique per payload. Used for the clone directory. |
| `version`     | string | yes      | Free-form version string, shown in the dashboard. |
| `description` | string | yes      | Short human-readable summary, shown in the dashboard. |
| `author`      | string | no       | Author/owner, shown in the dashboard. |

#### `dependencies` (optional)

Installed **before** any build steps, in the order `apt` → venv creation →
`requirements` → `pip` → `commands`. Unchanged from before — this is
module-level, install-time behaviour, not affected by the graph.

| Field       | Type     | Description |
| ----------- | -------- | ----------- |
| `apt`       | string[] | Debian packages, installed system-wide with `apt-get install -y`. |
| `requirements` | string | Path (relative to the repo root) to a pip requirements file, installed into the module's venv with `pip install -r`. If omitted, `requirements.txt` at the repo root is used automatically when present. |
| `pip`       | string[] | Extra PyPI packages/specifiers, installed into the module's venv after `requirements`. |
| `system_site_packages` | boolean | Create the venv with `--system-site-packages` (default `true`) so apt-provided Python libraries such as `python3-picamera2` stay importable. Set to `false` for full isolation. |
| `commands`  | string[] | Arbitrary shell commands run (via `bash -c`) in the cloned repo root — after the venv exists, and with placeholders expanded, so they can use `{venv_python}`/`{venv_dir}`. Use for udev rules, firmware setup, etc. |

##### Python virtual environments

Every module that uses Python gets its **own virtual environment** at
`/usr/local/eventide/packages/<name>/.venv`, created at install time — same
as before. A venv is created whenever the module has a requirements file,
declares `dependencies.pip`, or references a `{venv_...}` placeholder in
any program's command. The venv is shared by every node placed from that
module, however many there are, and is removed together with the module
directory on uninstall.

Program commands must run the venv interpreter explicitly — use the
`{venv_python}` placeholder:

```json
"command": "{venv_python} {module_dir}/camera_app.py --output-dir {recordings_subdir}"
```

#### `install` (optional)

| Field       | Type   | Description |
| ----------- | ------ | ----------- |
| `commands`  | string[] | Build commands run in the repo root (e.g. `"cargo build --release"`), with the same placeholder expansion as program commands (`{venv_dir}`, `{venv_python}`, …). Each has a 30-minute timeout. |
| `artifacts` | object | Map of `source path (relative to repo root)` → `destination path`. Destinations must be absolute after expansion and support the same placeholders as program commands (e.g. `{install_dir}/my_binary`). Every source is verified to exist after the build; the copy preserves the file mode (`cp -p`). Parent directories of destinations are created automatically. |

#### `recordings_subdir` (optional)

String, module-level (unchanged in shape from before). If present, a
program that references `{recordings_subdir}` in its command records to
`<recordings_dir>/<recordings_subdir>-<node id>` — **suffixed with the
placing node's id**, one directory per node, created when the graph is
submitted (not at install time — nothing has a node id yet at install
time). The dashboard's PLAYBACK tab gets one inner tab per such node,
automatically, for whatever's in the *active* graph; installing the module
alone adds nothing to PLAYBACK.

#### `programs` (required, ≥1)

One entry per node type the module contributes to the GRAPH tab's palette.
A camera module, for example, typically declares two: a datalogger and an
MJPEG server.

| Field          | Type    | Default                     | Description |
| -------------- | ------- | --------------------------- | ----------- |
| `name`         | string  | —                           | Program template identifier. Must match `^[a-z0-9][a-z0-9_-]*$` and be unique within the module (not across modules — see below). |
| `command`      | string  | —                           | Full command line; supports placeholders (below), resolved per graph node. |
| `directory`    | string  | `{install_dir}`             | Working directory; supports placeholders. |
| `autostart`    | boolean | `true`                      | Start when supervisord starts. |
| `autorestart`  | boolean | `true`                      | Restart if the process exits. |
| `startretries` | int     | `10000`                     | Start attempts before giving up. |
| `priority`     | int     | `10`                        | supervisord start/stop ordering. |
| `user`         | string  | `"root"`                    | User the program runs as. |
| `arguments`    | array   | `[]`                        | This program's own configurable surface — see below. |
| `sockets`      | array   | `[]`                        | This program's own socket declarations — see [docs/GRAPH_CONNECTIONS.md](GRAPH_CONNECTIONS.md). |
| `ui`           | array   | `[]`                        | This program's dashboard MAIN tab components — see [Dashboard UI components](#dashboard-ui-components-ui). |

Program names no longer need to be globally unique across every installed
module: what actually runs is one supervisor program per **graph node**,
named after that node's id (a slug of a user-editable label), not the
program template's name. Two different modules can both declare a program
called `server` with no conflict.

##### `arguments` (per program)

Command-line argument metadata — a program's configurable surface. The
GRAPH tab shows these as widgets on every node placed from this program
template, independently editable per node; defaults are substituted into
the program's command via `{arg:<name>}` when the graph is submitted.

| Field       | Type    | Required | Description |
| ----------- | ------- | -------- | ----------- |
| `name`      | string  | yes      | Argument identifier (`^[a-z0-9_]+$`), referenced as `{arg:name}`. |
| `flag`      | string  | yes      | CLI flag, e.g. `"--port"`. Informational. |
| `type`      | string  | yes      | `"str"`, `"int"`, or `"float"` (defaults are validated against it). |
| `default`   | any     | yes      | Value substituted for `{arg:name}` when no node overrides it. |
| `required`  | boolean | no       | Informational — whether the program needs the flag. |
| `description` | string | no      | Shown in the dashboard. |

##### `sockets` (per program)

Network/IPC endpoints this program's placed nodes expose or connect to.
Full field reference and the reasoning behind each tag:
[docs/GRAPH_CONNECTIONS.md](GRAPH_CONNECTIONS.md#socket-tags). Summary:

| Field         | Type   | Required | Description |
| ------------- | ------ | -------- | ----------- |
| `name`        | string | yes      | Socket identifier, unique within the program. |
| `direction`   | string | yes      | `"output"` (this program owns the address) or `"input"` (attaches to another node's output). |
| `transport`   | string | yes      | `"unix"`, `"tcp"`, or `"http"`. |
| `pattern`     | string | `unix`/`tcp` only | `"stream"` or `"request-reply"` (omit for `http` — always request-reply). |
| `stream_kind` | string | `stream` pattern only | `"irregular"`, `"regular"`, or `"framed"`. |
| `capped`      | boolean | no      | Only meaningful on a `stream`, non-`http` output — asserts it's safe to leave unconnected. |
| `port`        | int    | no       | TCP/HTTP only. Omit to let eventide pool-allocate one at graph-submit time. |
| `path`        | string | no       | Unix only. Omit to let eventide generate one at graph-submit time. |
| `count_arg`   | string | no       | Names an `int` argument that turns this into a variable-count socket template — see the linked doc. |
| `description` | string | no       | What the socket is for, shown in the dashboard. |

##### `ui` (per program)

Optional dashboard MAIN tab components this program's placed nodes offer —
see [Dashboard UI components](#dashboard-ui-components-ui) below for the
full widget reference (unchanged from before; only its placement in the
manifest, and what populates the palette, has moved).

### Command placeholders

Placeholders are expanded when a graph is submitted (for program commands
and `directory` fields) or at install time (for `dependencies.commands`,
`install.commands`, and `install.artifacts` destinations, which are
module-level and run once, before any node exists):

| Placeholder          | Expands to                                   |
| -------------------- | --------------------------------------------- |
| `{install_dir}`      | `/usr/local/eventide/code`                   |
| `{config_dir}`       | `/usr/local/eventide/config`                 |
| `{module_dir}`       | `/usr/local/eventide/packages/<name>`        |
| `{recordings_dir}`   | The payload's recordings directory           |
| `{recordings_subdir}`| `{recordings_dir}/<recordings_subdir>-<node id>` (program commands only — requires the manifest field, and a node id, so it never resolves for `dependencies.commands`/`install.*`) |
| `{venv_dir}`         | `/usr/local/eventide/packages/<name>/.venv`  |
| `{venv_python}`      | `{venv_dir}/bin/python3`                     |
| `{arg:<name>}`       | The node's value for that argument (its own override, or the declared default) |
| `{socket:<name>}`    | The address resolved for that socket by the graph compiler — a port, a path, or (for a `count_arg` socket) every numbered slot's address joined with commas |

For copies of copyable programs — **there's no such thing any more**. What
used to be a per-copy re-resolution of these placeholders is now just what
happens for every node: `{arg:...}`/`{socket:...}`/`{recordings_subdir}`
all resolve **per node**, always.

---

## The graph: nodes, edges, and how supervisor config is generated

A **node** is one placed instance of a program template; an **edge** is a
wire between one node's output socket and another's input socket. The
dashboard's GRAPH tab is a litegraph.js canvas: install a module to add its
programs to the palette, click a palette entry to place a node, drag from
an output dot to an input dot to wire two nodes together, edit a node's
argument widgets, then **SUBMIT**.

Submitting a graph is a **full stop-and-regenerate**, every time — there's
no incremental/diff-based apply:

1. The submitted graph is validated: every node's module/program must
   exist, every edge must connect a compatible output/input pair (matching
   `transport`, `pattern`, and — for streams — `stream_kind` exactly), and
   cardinality is enforced (a stream output accepts at most one edge; any
   input accepts at most one). This is the server-side re-check of exactly
   the rules in
   [docs/GRAPH_CONNECTIONS.md](GRAPH_CONNECTIONS.md#connection-compatibility-rules-summarised)
   — the editor's own checks are a convenience, this is the authority.
2. **Addresses are allocated**: one per (node, output socket) — a
   pool-allocated TCP port, or a generated unix path — regardless of how
   many edges use it (0 for an unconnected stub, 1 for a stream socket, any
   number for a `request-reply`/`http` one).
3. Every node's command is rendered, substituting `{arg:...}` from that
   node's own values and `{socket:...}` from the resolved address (an
   input socket's `{socket:...}` resolves to whatever output it's wired
   to; unconnected resolves to nothing).
4. A single generated conf, `/etc/supervisor/conf.d/eventide-graph.conf`,
   is written — one `[program:<node id>]` block per node — replacing
   whatever was there before, atomically, then applied with
   `supervisorctl reread && update` (which starts/stops/restarts programs
   to match the new file on its own — added nodes start, removed ones
   stop, changed ones restart).
5. The submitted graph becomes the new **active** graph.

**Unconnected stream sockets** don't fail validation — see `capped` in
[docs/GRAPH_CONNECTIONS.md](GRAPH_CONNECTIONS.md#capped-promising-you-wont-block-on-a-missing-reader).
The GRAPH tab shows a persistent warning badge on any such socket until
it's wired or the manifest marks it `capped`.

**Uninstalling a module in use by the active graph is allowed**, not
rejected: every node using one of its programs is dropped from the graph,
which is then revalidated and, if that removal introduces no new hard
error, resubmitted automatically (this is what actually stops the removed
nodes' supervisor programs). If it does introduce an error, the module is
still removed, but the previous graph is left running unchanged and the
error is reported — never a silently-applied broken config.

**Recordings and playback** are graph-node-driven the same way: PLAYBACK's
inner tabs come from the *active graph's* nodes, not installed modules —
see `recordings_subdir` above.

**The MAIN tab's ＋ COMPONENTS palette** is built from the active graph's
nodes too, per program's `ui[]` — see
[Dashboard UI components](#dashboard-ui-components-ui).

---

## Network locations (ports & proxying)

Nothing about a node's network presence is hardcoded outside the graph:

- **Addresses are allocated at graph-submit time**, not module-install
  time (see above) — a TCP port from the backend's pool (`--port-pool`,
  default `8100-8199`), or a generated unix path, one per (node, output
  socket). An explicit `port`/`path` in the manifest is still honoured
  (checked for collisions against other nodes' explicit addresses at
  submit time).
- **nginx has no per-node locations.** `config/eventide.nginx` only fronts
  `eventide.py` (`location /`) and the base playback server (`/playback/`).
  Every HTTP service a node exposes is proxied by the backend itself:

  ```
  /proxy/node/<node id>/<socket>/<upstream path>   →   http://127.0.0.1:<port>/<upstream path>
  ```

  Responses are streamed, so MJPEG works through it. Resolved from the
  *active graph's* resolved addresses at request time — `404` if that
  node/socket isn't part of the active graph, `502` if the node's server
  is down.
- **The dashboard resolves everything from `/api/modules` + `/api/graph`.**
  A `ui[]` entry's `socket` field names one of its owning program's own
  sockets; the MAIN tab cross-references the active graph's nodes against
  their program's `ui[]` to build the palette and proxy URLs — see below.

---

## Dashboard UI components (`ui`)

The MAIN tab of the dashboard is a **modular workspace**: a left sidebar, a
right sidebar, and a tabbed centre workspace. A program advertises the
panels its nodes offer in its own optional `ui` array (per-program, like
`arguments`/`sockets` — see above); the palette is built from the
**programs currently placed as nodes in the active graph**, not from
installed modules — every node contributes its own copy of its program's
`ui[]` entries, so placing a second node of the same program just gets its
own independent set of components, keyed by node id. The user places
components from the palette (＋ COMPONENTS button), drags them between
regions, and the layout persists in the browser per backend host.
Components with `"default": true` are placed automatically when a node
using them appears in the active graph.

Every component's traffic goes through the backend's node-scoped proxy —
`/proxy/node/<node id>/<socket>/<path>` — so `socket` in a `ui` config is
always an **http socket name from that same program's own `sockets`
list**, resolved for whichever specific node the component was placed
from.

### Common fields

| Field    | Type    | Required | Description |
| -------- | ------- | -------- | ----------- |
| `id`     | string  | yes      | Component id, `^[a-z0-9][a-z0-9_-]*$`, unique within the program. |
| `type`   | string  | yes      | Widget type (below). |
| `title`  | string  | no       | Panel header text. Gets a `· <node label>` suffix automatically when more than one node shares this program, so the two don't look identical. |
| `region` | string  | no       | `"sidebar"` (default) or `"center"` — where `default` placement puts it. |
| `default`| boolean | no       | Auto-place when a node using this program appears in the active graph (default `false`); otherwise palette-only. |

### Widget types

These field shapes are unchanged from the pre-graph manifest — only their
placement (per-program) and what populates the palette (the active graph,
not the module registry) are new.

| Type       | Region  | Config (in addition to `socket`) |
| ---------- | ------- | -------------------------------- |
| `mjpeg`    | center  | `path` — MJPEG stream path, e.g. `"/stream"`. Renders with offline/retry handling. |
| `form`     | sidebar | `get`, `put`, `method?` (`"PUT"` default, or `"POST"`), `submit_label?`, `fields[]`. GET populates, PUT/POST applies. Field: `{key, label?, kind: number\|slider\|toggle\|text\|select\|nudge, min?, max?, step?, get?, put?, options?}` — per-field `get`/`put` overrides let one form span several endpoints. A `nudge` field is a number input with `-step` / `+step` buttons underneath: a click adjusts the value by `step` (default 1, clamped to `min`/`max`) and immediately submits the form; the box is editable like a number field and its value is never refreshed from the server on its own — it changes only via a nudge click or a manual edit. |
| `telemetry`| sidebar | `get`, `interval?` (ms), `rows[]` — polled readout. Row: `{label, path, fmt?}`; `path` is a dot-path into the JSON (`buffer.bytes`). |
| `features` | sidebar | `get` (capabilities-tree endpoint, e.g. `/api/camera/features`), `filter?` — adaptive GenICam feature form. Fetches the tree once (`{categories: [{name, display_name, features: [...]}]}`) and renders each feature as a collapsible-category control: slider/number for `Integer`/`Float` (with `min`/`max`/`inc`/`unit`), select for `Enumeration` (`enum_entries`), toggle for `Boolean`, button for `Command`, read-only text for `access: RO`. `locked`/`available` disable/hide controls; `filter?: beginner\|expert\|guru\|all` (default `expert`) sets the visibility ceiling (`Invisible` never renders). Live values are fetched lazily per expanded category via `GET <get>/<name>`; edits `PUT <get>/<name>` with `{"value": ...}` and the control shows the re-read effective value. |
| `recording`| sidebar | `get`, `put`, `interval?` (ms), `key?`, `control_key?`, `file_key?`, `duration?` — recording start/stop control with live status. Polls `get` (expects `{"recording": bool, "recording_control": bool, "current_file": string|null}`; the keys are the defaults of `key`/`control_key`/`file_key`) and the button PUTs `{"recording": bool}` to `put`. When the source reports `recording_control: false` (e.g. the recorder runs without its toggle flag) the button locks and the state is shown read-only. `duration?: {default?, min?, max?, key?, label?}` adds a clip-length input: the start PUT then also carries `{"<key>": seconds}` (default key `duration_seconds`) so high-bandwidth sources can auto-stop; sources that don't know the key ignore it and record until stopped. |
| `joystick` | sidebar | `put`, `telemetry_get?`, `paths?: {x, y}`, `fields?: {x, y, frame}`, `gamepad?: {axes?, deadzone?}` — two-axis RC pad seeded from a telemetry poll; the poll is skipped while a pan/tilt box has focus so manual edits aren't overwritten. Any browser-connected gamepad (Xbox/XInput pads included, via the Gamepad API) also drives the pad: the stick sets a deflection-proportional slew rate, mirrored on the knob. `axes` picks the stick (default `[0, 1]`, the left stick; `[2, 3]` for right), `deadzone` is radial (default `0.15`). The pad only appears after its first button press (browser requirement); pointer drags take priority while active. |
| `table`    | sidebar | `get`, `interval?`, `columns[]` (`{label, path, fmt?}`), `row_action?: {label, method, path, key}`, `stop_action?: {label, method, path}` — polled table with a per-row action button (e.g. ADS-B track/stop). |
| `map`      | center  | `track?: {socket, get, interval?, lat, lon, heading?, gimbal?, frame?}`, `adsb?: {socket, get, interval?, lat, lon, label?, key?}` — Leaflet map with optional device/track markers. User map clicks are shown as a separate marker from the system target reported by the track endpoint (`target: {lat, lon}`). Without bindings it's a plain map. |
| `orientation3d` | center | `get`, `interval?` (ms), `paths?: {pan, tilt, roll}` (telemetry keys; default `pan`/`tilt`/`roll`), `warn_delta?` (deg, default 45), `model?` — 3D attitude indicator; drag to orbit the view. `model` names an STL file relative to the module root (served read-only at `/api/modules/<name>/files/<path>` — module-scoped, since a model file is a static asset shared by every node, not something graph-scoped): it is drawn flat-shaded at the stage centre, auto-centred and auto-scaled, and rotated by the telemetry. Model axes: **+Z forward (the pointer direction), +Y up, +X right** — keep meshes small (≲ 10k triangles). If the file is missing or unparseable the widget falls back to its built-in fin shape. |

`fmt` is one of the dashboard's named formatters: `int`, `f1`, `f2`, `f6`
(decimal places), `m_km` (metres → m/km).

### Special component types

| Type          | Region   | Description |
| ------------- | -------- | ----------- |
| `master-record` | `sidebar` | Aggregates every `recording`-type component currently placed **by the active graph** into one RECORDING panel with per-source rows and a RECORD ALL / STOP ALL button. RECORD ALL / STOP ALL calls the base backend's `POST /api/recording/trigger` (body `{"recording": bool, "duration_seconds"?: number}`), which fans start/stop out to every such node server-side (with server-side auto-stop after `duration_seconds`), so it also works with no browser open — that's what the `eventide-core` scheduler below uses. Provided by the built-in `eventide-core` module; no other module should declare it. |
| `schedule-table` | `sidebar` | Lists cron-triggered recording jobs from a scheduler's CRUD API (`get` → `{"items": [{id, label, cron, duration_seconds, enabled, last_run, last_result}]}`), with a per-row enable/disable checkbox (`PATCH <get>/<id>` with `{"enabled": bool}`) and a delete button (`DELETE <get>/<id>`). A dedicated type rather than `table`, since `table`'s `row_action`/`stop_action` assumes a single globally-active row rather than N independently-toggleable, server-tracked ones. Paired with a plain `form` widget (`put` → `POST` to the same API, fields `label`/`cron`/`duration_seconds`) for creating jobs. Provided by the built-in `eventide-core` module's scheduler service. |

### Example (a camera module's `mjpeg_server` program)

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
  { "id": "record-control", "type": "recording", "title": "EVK4 RECORD",
    "region": "sidebar", "default": true, "socket": "mjpeg",
    "get": "/api/recording", "put": "/api/recording" }
]
```

(`socket` here names a socket declared on the **same program**, e.g. one
tagged `transport: "http"`.)

---

## Install lifecycle & error handling

Installing is a **background job** on the backend. The job moves through these
statuses:

```
pending → cloning|extracting → deps → building → artifacts → configuring → done
                                                                              ↘ failed
```

| Status        | What happens |
| ------------- | ------------ |
| `cloning`     | Repo cloned to a staging dir (`packages/.staging-<job>`). HTTPS is tried first; for `github.com` URLs a SSH (`git@github.com:…`) retry follows automatically. The manifest is read and fully validated here — including name/`recordings_subdir`/explicit-port conflicts with already-installed modules. |
| `extracting`  | Zip installs only: the uploaded zip is stored under `packages/.uploads/` and extracted into staging (zip-slip paths are rejected). The manifest must sit at the zip root or in a single top-level folder (as GitHub's "Download ZIP" produces). Validation then proceeds exactly as for `cloning`. |
| `deps`        | `dependencies.apt`, then the module venv is created and `requirements`/`pip` are installed into it, then `dependencies.commands`. |
| `building`    | `install.commands` run in the repo root. |
| `artifacts`   | Every `install.artifacts` source is checked for existence, then copied to its destination. |
| `configuring` | The module is recorded in `/usr/local/eventide/modules.json`. Nothing is rendered into supervisor conf or started here any more — that's the graph compiler's job, once a node using one of this module's programs is placed and submitted. |
| `done`        | Install finished; the module's programs are now available in the GRAPH tab's palette. |
| `failed`      | See rollback below. |

**Rollback.** Any hard failure (clone error, invalid manifest, dependency or
build command exiting non-zero, missing artifact) removes everything the
job created: copied artifacts, the staging/clone directory, and any
partial registry entry. The job ends `failed` with the failing command's
output in its log.

**There's no more install-time "verifying" step**, since install doesn't
start anything to verify — a program's actual runtime health only exists
once it's a node in a submitted graph; see `GET /api/graph/status`.

**Concurrency.** One install at a time; concurrent requests are rejected with
`409 Conflict`. A graph import (below) that needs to install several
modules does them one at a time, holding this same lock per install.

**Updating a module** is uninstall + reinstall (no in-place upgrade yet).

---

## Backend API reference

All endpoints are served by `eventide.py` under `/api/modules` and
`/api/graph` (through nginx on the payload, like the rest of `/api`).
Errors return `{"error": "<message>"}` with a 4xx/5xx status.

### Module endpoints

#### `GET /api/modules`

List installed modules — the palette the GRAPH tab's node types are built
from.

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
      "programs": [
        {
          "name": "hello_module",
          "command": "…",
          "arguments": [ … ],
          "sockets": [ … ],
          "ui": [ … ],
          "autostart": true, "autorestart": true,
          "startretries": 10000, "priority": 10,
          "active_node_count": 2
        }
      ]
    }
  ]
}
```

A program has no live status of its own — `arguments`/`sockets`/`ui` are
templates, not running things. `active_node_count` is a convenience count
of how many nodes in the **active** graph currently use this program,
mainly so the dashboard can warn before an uninstall. See
`GET /api/graph/status` for what's actually executing.

#### `GET /api/modules/<name>`

Full detail for one installed module (registry entry including the raw
manifest). `404` if not installed.

#### `POST /api/modules/install`

Body: `{"repo_url": "https://github.com/you/module", "ref": "main"}` (`ref`
optional — branch or tag). Starts a background install job.

- `202 Accepted` → `{"job_id": "…", "status": "pending"}`
- `400` missing/invalid `repo_url` · `409` another install is already running

#### `POST /api/modules/install-upload`

Installs a module from an uploaded zip file instead of a git clone. The
request is `multipart/form-data` with the archive in the `file` field. The
zip must contain `eventide-module.json` at its root or inside a single
top-level folder (what GitHub's **Download ZIP** produces). Maximum upload
size: 100 MB. A zip-sourced module can't be re-cloned by graph export/import
(below) — it goes in the exported bundle's `preinstalled_modules` list
instead.

- `202 Accepted` → `{"job_id": "…", "status": "pending"}` — poll the job as
  usual; the first status is `extracting` instead of `cloning`
- `400` no file / not a `.zip` · `409` another install is already running ·
  `413` zip too large

#### `GET /api/modules/jobs/<job_id>`

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

`status` ∈ `pending, cloning, deps, building, artifacts, configuring, done,
failed`. When `done`, `module` is the module name. When `failed`, `error`
says why.

#### `POST /api/modules/<name>/uninstall`

Removes the module's files and registry entry. **Allowed even if the
module is in use by the active graph** — see
[The graph](#the-graph-nodes-edges-and-how-supervisor-config-is-generated)
above for the cascade behaviour. `404` if not installed.

```json
{
  "ok": true, "removed": "hello-module",
  "graph_removed_nodes": ["cam1", "cam2"],
  "graph_regenerated": true,
  "graph_error": null
}
```

`graph_removed_nodes` lists which active-graph nodes used this module (and
were dropped); `graph_regenerated` says whether the trimmed graph was
successfully resubmitted; `graph_error` is set (and the *previous* graph
left running unchanged) if it couldn't be.

#### `/proxy/node/<node id>/<socket>/…`

Generic proxy to the HTTP service behind one active-graph node's socket:
forwarded (streamed) to `http://127.0.0.1:<resolved port>/<upstream path>`.
`404` when that node/socket isn't part of the active graph, `502` when the
node's server is down.

### Graph endpoints

#### `GET /api/graph`

```json
{ "active": { "nodes": [...], "edges": [...] } | null,
  "draft":  { "nodes": [...], "edges": [...] } | null }
```

A node: `{"id", "label", "module", "program", "args": {...}, "pos": [x, y]}`.
An edge: `{"id", "from": {"node", "socket"}, "to": {"node", "socket"}}`.

#### `PUT /api/graph/draft`

Body: a graph document (`{"nodes": [...], "edges": [...]}`). Autosaved by
the GRAPH tab as you edit; no validation beyond basic shape, and no effect
on the running system until submitted.

#### `POST /api/graph/submit`

Body: a graph document (or omit it to submit the current draft). Validates
and, on success, applies it as the new active graph — see
[The graph](#the-graph-nodes-edges-and-how-supervisor-config-is-generated).

- `200` → `{"ok": true, "warnings": [{"node", "socket", "message"}, ...]}`
  — `warnings` are the persistent unconnected-capped-stream-socket kind,
  not submission blockers.
- `400` → `{"error": "...", "errors": ["...", ...]}` — every validation
  problem, nothing applied.

#### `GET /api/graph/status`

```json
{ "nodes": [{"id", "status", "status_detail"}, ...],
  "warnings": [{"node", "socket", "message"}, ...] }
```

Live per-node supervisor status for the active graph, plus the same
persistent warnings `submit` returns. Per-node start/stop/restart and log
tailing don't need a dedicated REST endpoint — a node is just a supervisor
program named after its id, so the existing `/supervisor/` XML-RPC proxy
(`supervisor.startProcess`/`stopProcess`/`tailProcessStdoutLog`/etc., the
same one the dashboard already uses) works unchanged, keyed by node id.

### Graph export / import

Full design: `docs/GRAPH_SUPERVISOR_PLAN.md` §12.

#### `GET /api/graph/export?source=active|draft`

Bundles that graph (default `active`) with, for every module it
references, enough to reinstall it identically elsewhere:

```json
{
  "eventide_export_version": 1,
  "exported_at": "…",
  "source": "active",
  "graph": { "nodes": [...], "edges": [...] },
  "modules": [
    {"name": "cam-mod", "repo_url": "https://github.com/you/cam-mod", "ref": "main", "commit": "abc123..."}
  ],
  "preinstalled_modules": ["eventide-core"]
}
```

`modules` lists git-sourced modules with everything needed to re-clone
them at the *exact* commit that was running. `preinstalled_modules` lists
modules with no usable `repo_url` (zip-sourced, or one of the base
platform's own `--install-local` modules like `eventide-core`) by name
only — the destination is expected to already have these; import checks
for that up front. `400` if the graph references a module that isn't
currently installed.

#### `POST /api/graph/import`

Body: an export bundle (as above). Starts a background job that: checks
every `preinstalled_modules` entry is already present (failing immediately
and clearly if not); installs each `modules` entry not already present,
cloning at `ref` and checking out the exact `commit`; then loads `graph`
as the **draft** — never active — so the operator reviews and explicitly
submits it on the destination.

- `202 Accepted` → `{"job_id": "…", "status": "pending"}`
- `400` malformed bundle (missing/wrong-shaped `graph`)

#### `GET /api/graph/import/jobs/<job_id>`

```json
{
  "id": "…", "status": "installing",
  "log": ["module 'cam-mod' already installed — skipping", "…"],
  "error": null, "warnings": [],
  "modules_installed": ["…"], "modules_skipped": ["cam-mod"],
  "created_at": "…", "finished_at": null
}
```

`status` ∈ `pending, installing, validating, done, failed`.

---

## Authoring a module

1. Start from [`module-template/`](../module-template/).
2. Write the manifest. Validate it locally:
   `python3 -m json.tool eventide-module.json > /dev/null`.
3. Keep programs **foreground** processes — supervisord manages daemonisation,
   restarts, and logging. Log to stdout/stderr; it lands in
   `/var/log/supervisor/<node id>.log` once placed and submitted.
4. Put everything a program needs at runtime either in `install.artifacts`
   (copied to a stable location) or reference it inside `{module_dir}` — the
   clone is not removed after install.
5. Declare each program's own `arguments` and every socket it binds or
   attaches to in that same program's `sockets` — tagged
   `direction`/`transport`/`pattern`/`stream_kind` (see
   [docs/GRAPH_CONNECTIONS.md](GRAPH_CONNECTIONS.md)) so the GRAPH tab
   knows what it can be wired to. Reference them from the program's
   command with `{arg:<name>}`/`{socket:<name>}`, and let eventide
   allocate ports/paths (omit `port`/`path` unless you genuinely need a
   fixed one).
6. Push to GitHub and install from the dashboard: **MODULES → enter the repo URL →
   INSTALL** — or zip the folder and use **ZIP FILE** (drag & drop works too).
   Then switch to the **GRAPH** tab, place a node for each program you want
   running, wire them up, and **SUBMIT**.

---

## Porting the existing components

The pre-module `install.sh` cloned and installed four repositories. Each
maps to a module, written against the **pre-graph** manifest shape
(top-level `arguments`/`sockets`/`ui`, no socket tags) — migrating them to
the per-program, tagged-socket shape this document describes is tracked
separately (`docs/GRAPH_SUPERVISOR_PLAN.md` §2, §8, workstream 8) and
hasn't been done yet, so installing one of these today needs that
migration first.

### `Y2Kmeltdown/evk_datalogger` → `evk-datalogger`

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
- sockets to tag: two unix frame/trigger sockets (pick `stream_kind` per
  what the format actually is) and an `mjpeg` socket that should become
  `transport: "http"` (it's an MJPEG-over-HTTP stream, not a raw `tcp`
  socket)

### `Y2Kmeltdown/picam_datalogger` → `picam-datalogger`

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
- sockets to tag: a unix frame socket (`transport: "unix"`,
  `stream_kind` per the frame format) and `mjpeg` → `transport: "http"`

### `ericltb15/aravis-ir` → `ircam-datalogger`

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
- sockets to tag: `frames` (unix, `stream_kind` per the raw frame format)
  and `mjpeg` → `transport: "http"`

### `j-vanarsdale/tripwire-gimbal-point` → `gimbal-controller`

- `dependencies.pip`: `telnetlib3`, `pymavlink==2.4.49`, `pyserial==3.5`
- `install.artifacts`: `GIMBAL_POINT_API.py`, `adsb.py` → `{install_dir}`
- `recordings_subdir`: `telemetry`
- `programs`: `gimbal_controller` — port the full command line from the old
  `config/supervisor.conf` git history, with `{recordings_subdir}` in place of
  the old `SEDPLACEHOLDER`
- sockets to tag: `api`, kept at explicit `port: 5001` (external clients
  depend on it) → `transport: "http"` (it's the gimbal's HTTP API)

The old pinned versions in `config/requirements.txt` (`opencv-python-headless`,
`smbus2`, `spidev`, `gpiozero`, …) belonged to these components — move them
into the corresponding module manifests, not the base install.

---

## Migrating from a pre-module install

On a payload that already runs the old monolithic (pre-module) setup:

1. Remove the stale supervisor configs:
   `sudo rm /etc/supervisor/conf.d/supervisor.conf /etc/supervisor/conf.d/supervisord.conf`
   (whichever exists), then `sudo supervisorctl reread && sudo supervisorctl update`.
2. Run the new `install.sh` (safe to re-run; it installs the base only).
3. Reinstall each component as a module from the dashboard's MODULES tab,
   then place and wire its programs as nodes in the GRAPH tab and SUBMIT.

Fresh installs need no migration — `install.sh` removes the stale files
itself.

---

## Troubleshooting

- **Job failed during `cloning`** — check the repo URL is correct and public;
  private repos need working SSH keys on the payload (the installer retries
  `git@github.com:…` automatically).
- **Job failed during `building`** — read the job log in the MODULES tab; the
  failing command's stdout/stderr is captured there.
- **A module installed fine but nothing's running** — this is normal now:
  install only adds programs to the GRAPH tab's palette. Place a node for
  each program you want running, wire it up, and SUBMIT.
- **A node is not RUNNING after SUBMIT** — check `GET /api/graph/status` /
  the GRAPH tab's per-node status badge; hardware-dependent programs
  legitimately fail without their devices attached, and a persistent
  amber warning badge means an unconnected, non-`capped` stream socket
  (wire it up, or mark it `capped` in the manifest if the producer is
  already safe to leave unconnected).
- **"socket port 8081 already used by module X"** — two program templates
  explicitly request the same TCP/HTTP port; drop the `port` field from one
  socket and let eventide pool-allocate it.
- **"no free port in pool 8100-8199 for socket …"** — the allocation pool is
  exhausted (or everything in it is bound); widen it with the backend's
  `--port-pool start-end` flag.
- **Submitting a graph fails validation** — the error lists every problem
  (`errors: [...]`); the most common are a transport/pattern/stream_kind
  mismatch on an edge, or a stream output with more than one connected
  edge (insert a Stream Fan-out node instead — see
  [docs/GRAPH_CONNECTIONS.md](GRAPH_CONNECTIONS.md#the-stream-fan-out-node)).
- **Supervisor didn't pick up a submitted graph** — `sudo supervisorctl
  reread && sudo supervisorctl update`, then check
  `/etc/supervisor/conf.d/eventide-graph.conf` was actually rewritten.
- **Full install/import log** — job logs are kept in memory by
  `eventide.py`; journald has the backend's own output:
  `journalctl -u eventide.service`.
