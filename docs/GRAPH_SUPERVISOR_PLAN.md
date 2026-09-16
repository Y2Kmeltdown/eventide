# Graph-defined supervisor: design & implementation plan

Status: **draft for review** — sections marked **PROPOSED** are this plan's
best design given the answers gathered so far; they are not yet implemented
and should be corrected before work starts if they're wrong. Sections marked
**OPEN QUESTION** / **OPEN SUB-QUESTION** are decisions this plan deliberately
leaves for the user because they weren't resolved in the scoping conversation.

## 1. Goal

Replace the current model — *install a module → its manifest's `programs[]`
run automatically, wired together by convention (fixed socket names, the
generic HTTP proxy)* — with one where **the running set of supervisor
programs, and the connections between them, are defined by a graph the user
builds in a litegraph.js editor**. A node is a placed instance of one
program from an installed module; a wire between two node sockets is a real
socket/HTTP connection materialised on the device when the graph is
submitted. This makes the data flow between sensors, algorithms, and
effectors an explicit, visible, user-authored thing instead of an implicit
consequence of which modules happen to be installed.

## 2. Scope, confirmed with the user

**Round 1:**

| Question | Decision |
| -------- | -------- |
| How does a graph edge become a real connection? | **Direct address injection** — eventide generates the concrete address (TCP port / unix path) for a connection and substitutes it into both endpoints' rendered commands, the same way `{socket:name}` works today, just scoped to a specific wire instead of a whole module. No broker/proxy process sits in the data path for raw socket streams. |
| Fan-out / fan-in cardinality | Unix (and, by extension, raw TCP) stream sockets are **strictly one-to-one**. A built-in **Stream Fan-out** node type takes one stream input and duplicates it to N outputs, for when the user needs to split a stream. HTTP sockets fan out natively (a server can serve any number of callers) and need no special node. Fan-*in* (merging multiple upstream outputs into one input) is **not supported** in this plan — see §9. |
| Migration of existing hardware modules | Out of scope for this plan. Only the base platform, `module-template/`, and the in-repo `eventide-core` module are migrated to the new manifest schema. `evk-datalogger`, `picam-datalogger`, `ircam-datalogger`, and `gimbal-controller` (separate repos) keep their current manifests and **will not install or run correctly against the new backend** until someone migrates them — this plan documents what that migration involves (§8) but doesn't do it. |
| Editing a running graph | **Full stop-and-regenerate on every submit.** Submitting a graph tears down and re-renders the entire generated supervisor config, exactly like today's "reinstall = full re-render" model. No diffing, no partial hot-reload (uninstall-triggered regeneration in §7 is the one scripted exception, and it reuses this same full-regenerate mechanism). |
| Is `eventide-core` inside or outside the graph? | **Inside.** The playback server, the recording scheduler, and (per §9) the new stream-fanout utility are themselves graph nodes, placed and wired like any other module's programs — not fixed always-on infrastructure. |

**Round 2:**

| Question | Decision |
| -------- | -------- |
| §6 `stream_kind` granularity | Collapses to **three** mutually-incompatible groups — `irregular`, `regular`, `framed` — not the original four. Framing subsumes the regular/irregular distinction (a `framed` consumer handles both regular- and irregular-sized framed data identically, since the frame header always says how much is coming); unframed data keeps the split because regularity is the only thing that makes unframed data parseable at all without a frame header. See revised §6. |
| §7 unconnected required stream sockets | Not a hard validation error. The compiler still allocates a real "stub" address so the program starts normally, and the GRAPH tab shows a **persistent** warning on that socket — "no consumer connected; may block/crash if the producer isn't capped" — until it's wired up or the manifest marks the socket `capped` (an assertion that the producer checks for a live connection and skips writes when there is none). Not needed for `http` sockets, where an unconnected server is harmless. See revised §7. |
| §7 uninstalling a module in use by the active graph | **Allowed.** The affected nodes are removed from the graph automatically, the graph is regenerated (full stop-and-regenerate) if that removal introduces no new hard errors, and the user is shown a confirmation warning listing the affected nodes before the operation proceeds. See revised §7. |
| §9.2 (old) MAIN tab scope | **In scope for this pass**, but simpler than first proposed and then walked back: `ui[]` **stays** in `eventide-module.json` (not removed, not turned into graph nodes) — it just moves from module top level to per-program, the same relocation as `arguments`/`sockets`. What changes is only where the MAIN tab's palette comes from: the programs currently placed in the active graph, not installed modules. See revised §10. |
| Node instance identity | Confirmed as originally proposed: auto-slug from a user-editable per-node label, unique within the graph. |

## 3. Terminology

- **Module** — an installable unit (unchanged concept): a git repo or zip
  with an `eventide-module.json`, built/installed once (deps, venv, build,
  artifacts). A module contributes one or more **program templates** to the
  litegraph **palette** — it no longer starts any supervisor program by
  itself.
- **Program (template)** — one entry in a module's `programs[]`. Defines a
  command, its arguments, and its sockets. This is what becomes one
  litegraph **node type**.
- **Node** — one placed instance of a program template on the canvas. Multiple
  nodes of the same program template are allowed and fully replace today's
  `instance`/"copies" mechanism — there is no more base-program-vs-copy
  distinction; every node is just a node.
- **Socket (manifest)** — a named input or output on a program template,
  tagged with transport/pattern/stream-kind (§6). Becomes a litegraph input
  or output slot on that node type.
- **Edge** — a wire in the graph, connecting one node's output socket to
  another node's input socket. Materialises as a real allocated TCP
  port/unix path or, for HTTP, a call against the server node's existing
  address.
- **Graph** — the full node+edge document the user builds and submits.

## 4. What changes vs. today, per the user's 7 requirements

1. **New GRAPH tab** — litegraph.js canvas. Palette = program templates from
   installed modules, plus built-in utility node types like Stream Fan-out
   (§9). Node params = that program's `arguments`. Node sockets = that
   program's `sockets`, split by `direction` into litegraph input/output
   slots.
2. **New backend: graph compiler** (§7) — turns a submitted graph into a
   rendered supervisor conf: one `[program:...]` block per process node,
   addresses allocated per edge/socket, then `supervisorctl reread &&
   update`.
3. **MODULES tab loses supervisor duties.** It becomes purely an install /
   uninstall / palette-management screen (install job flow unchanged); the
   live program list, start/stop/restart, and status/log viewing move to the
   GRAPH tab (§4.4), operating on graph nodes instead of module programs.
4. **GRAPH tab, once submitted:** each node shows live status
   (RUNNING/STOPPED/FATAL/…), START/STOP/RESTART buttons, and OUT/ERR buttons
   that open that node's log in a right-hand panel — the same interaction
   MODULES has today, just keyed by node instance instead of module program.
5. **`arguments` moves from module top level to per-program**, and so does
   **`ui`** (§10). Each entry in `programs[]` gets its own `arguments[]` and
   `ui[]` arrays (same field shapes as today). This is required because two
   programs in the same module can now be placed as independently-configured
   nodes with no shared "module args"/"module ui" concept — every node
   instance owns its own argument values and contributes its own dashboard
   panels.
6. **New README**: `docs/GRAPH_CONNECTIONS.md` (§11) documenting the socket
   tag taxonomy, direction/binding semantics, the stream-kind distinctions,
   and how to author a module against this system.
7. **`sockets[]` gets tags** — `direction`, `transport`, `pattern`, and
   (when applicable) `stream_kind` and `capped`. Full taxonomy in §6.

## 5. Palette & node model

- `GET /api/modules` (already exists) is extended so each program entry
  carries its own `arguments` and `sockets` (moved out of the module-level
  fields). The GRAPH tab's palette builds one litegraph node type per
  `(module, program)` pair via `LiteGraph.registerNodeType`, generated
  dynamically from that metadata — no per-module frontend code, matching the
  existing "manifest is the single source of truth" philosophy.
- Node widgets are generated from `arguments` the same way MODULES' edit
  form fields are today (str/int/float, defaults editable per node).
- Node slots are generated from `sockets`, filtered by `direction` into
  litegraph inputs (this program consumes from here) and outputs (this
  program exposes/owns this address). Slot *type* strings (litegraph's
  mechanism for colour-coding and restricting valid connections) are derived
  from `transport` + `pattern` + `stream_kind`, e.g. `unix/stream/framed` —
  the editor refuses to connect two slots whose types aren't compatible
  (§6 defines compatibility precisely).
- **PROPOSED**: placing a node auto-generates a short instance id (slug of a
  user-editable label, unique within the graph — same slugification rule
  `instance`/copies uses today) used for supervisor program naming, log file
  naming, and (if the program declares `recordings_subdir`) its own
  recordings subdirectory, exactly as today's `<program>-<cid>` /
  `<subdir>-<cid>` scheme, just derived from graph placement instead of an
  instance-argument value.

## 6. Socket tag taxonomy — PROPOSED (revised)

Every entry in a program's `sockets[]` gains:

| Field | Values | Meaning |
| ----- | ------ | ------- |
| `direction` | `"output"` \| `"input"` | **`output`** = this program *hosts/owns* the address — it binds a unix path, listens on a TCP port, or runs the HTTP server. **`input`** = this program *attaches to* an address owned by whatever output socket it's wired to. This is a binding/ownership direction, not necessarily a byte-flow direction — for a `request-reply` socket the "output" side is the server and the "input" side is the client that calls it. |
| `transport` | `"unix"` \| `"tcp"` \| `"http"` | Physical mechanism. `http` is modelled distinctly from raw `tcp` because it's always `request-reply` and fans out natively (§2). |
| `pattern` | `"stream"` \| `"request-reply"` | Captures "free-flowing" vs "programs want to request data" from the original brief. `http` sockets are always `request-reply`. `unix`/`tcp` sockets declare one or the other explicitly. |
| `stream_kind` | `"irregular"` \| `"regular"` \| `"framed"` | **Required** when `transport` is `unix` or `tcp` and `pattern` is `stream`; omitted otherwise. Three mutually-incompatible groups, **not** a four-way product of framed/unframed × regular/irregular: `framed` covers both fixed- and variable-sized framed data (the frame header always tells the reader how much is coming, so regularity stops mattering once there's a frame); `regular` and `irregular` are unframed data, where regularity is the only thing that makes it parseable at all — so those two stay distinct from each other and from `framed`. |
| `capped` | boolean, default `false` | Only meaningful on a `stream`-pattern, non-`http` **output** socket. Asserts that the producer checks for a live connection before writing and simply skips output when nothing's attached — i.e. it is *known safe* to leave unconnected. Suppresses the persistent "no consumer" warning described in §7. |

**Connection compatibility** (enforced both client-side in the litegraph
editor via slot type strings, and re-validated server-side on submit — never
trust the browser alone):

- `transport` must match exactly.
- `pattern` must match exactly.
- For `stream` sockets, `stream_kind` must match exactly between `irregular`,
  `regular`, and `framed` — there is no partial compatibility between them.
  **Resolved**: an input socket declares exactly **one** `stream_kind`, not
  a set of acceptable kinds — for current simplicity. A node that genuinely
  wants to accept more than one kind needs a separate input socket (and
  program-side handling) per kind for now; broadening this to an
  accepted-set later is a compatible schema change if it turns out to be
  needed.
- **Cardinality**: an `output` socket with `transport ∈ {unix, tcp}` and
  `pattern: stream` accepts **at most one** connected edge (§2) — enforced
  in the editor (no second wire allowed) and re-checked on submit. `http`
  outputs and any `request-reply` socket accept any number of edges. There
  is no fan-in in this plan: an `input` socket, regardless of type, accepts
  at most one incoming edge. A stream output left with **zero** edges is
  *not* a validation error — see §7's stub/warning behaviour.

## 7. Backend: the graph compiler

New endpoints, replacing the module-conf-per-install flow for anything that
was previously auto-started:

- `GET /api/graph` — the currently active (last-submitted) graph, plus a
  separately-persisted **draft** the user is still editing (`PUT
  /api/graph/draft`, autosaved, has no effect on the running system).
- `POST /api/graph/submit` — validates the draft (§6 compatibility rules,
  unknown module/program references, duplicate instance ids — see below for
  the unconnected-socket case, which is a warning, not a validation failure)
  and, if valid:
  1. Stops everything currently running from the previous graph.
  2. Allocates addresses: a fresh TCP port from the existing pool
     (`--port-pool`) for every `tcp`/`http` output socket that doesn't
     already have one; a fresh generated unix path
     (`/tmp/eventide-<node-id>-<socket>.sock`) for every `unix` output
     socket; **one per edge** for `stream`-pattern sockets (since they're
     1:1, "the edge's address" and "the output socket's address" are the
     same thing and can be allocated once, when the edge is drawn — and
     allocated as a **stub**, with the identical shape, even when there are
     zero edges, so the placeholder always resolves); **one per node** (not
     per edge) for `request-reply`/`http` output sockets, reused across
     every client edge — this is exactly today's `{socket:name}` allocation,
     just scoped to a node instead of a module.
  3. Renders every node's command, substituting `{arg:name}` from that
     node's argument values and `{socket:name}` from the address resolved
     in step 2 — reusing the existing placeholder engine unchanged.
  4. Writes a single generated conf,
     `/etc/supervisor/conf.d/eventide-graph.conf` (replacing the current
     per-module `module-<name>.conf` files — there's no more "module conf",
     only "the graph's conf"), atomically, then `supervisorctl reread &&
     update`.
  5. Persists the submitted graph as the new active graph.
  - `400` with every validation problem listed (mirrors
    `/api/modules/<name>/args`'s error style) if step 1 fails validation —
    nothing is torn down or touched.

**Unconnected stream sockets.** A `stream`-pattern, non-`http` output socket
with zero connected edges is not a compile error: step 2 above allocates it a
stub address exactly as if it were connected, so the producer program starts
normally and its `{socket:name}` placeholder always has a value. After a
successful submit, the GRAPH tab shows a **persistent** warning badge on that
socket — not a one-time toast — reading approximately "no consumer connected;
if this program isn't capped it may block or crash when nothing connects."
The warning stays visible for as long as the socket is unconnected, and is
suppressed entirely when the socket's manifest entry sets `"capped": true`
(§6). `http` output sockets never trigger this warning, connected or not — a
server with no current callers is normal and not analogous to a stream
producer blocked on a write with no reader.

**Uninstalling a module in use by the active graph** is allowed rather than
rejected: every node in the active graph whose program template belongs to
the module being uninstalled is removed from the graph, the resulting graph
is re-validated, and — if that removal introduces no new hard validation
error — resubmitted (full stop-and-regenerate, per §2) automatically as part
of the uninstall. The dashboard shows a confirmation dialog before proceeding
("Uninstalling `<module>` will remove N node(s) from the running graph and
regenerate the system — continue?"), listing the affected nodes. If removal
does introduce a hard error, the module and its nodes are still removed from
the *draft*, but the previous graph is left running unchanged and the error
is reported, rather than applying a broken config.

Per-node control, replacing the module program endpoints for anything
graph-driven: `POST /api/graph/nodes/<node_id>/start|stop|restart`, log tail
endpoints mirroring whatever MODULES already exposes for stdout/stderr
(reuse verbatim — logs are still just
`/var/log/supervisor/<program_name>.log`, `program_name` is now derived from
the node id). `GET /api/graph/status` gives live supervisor status per node
(thin wrapper around the existing `supervisor_statuses()`), polled by the
GRAPH tab the same way MODULES polls today.

**Module install** keeps doing exactly what it does now (clone/extract →
deps → build → artifacts) but **stops at "configuring"** — it no longer
renders or starts any supervisor program. A module is "installed" the moment
its program templates are available in the palette; nothing runs until a
graph referencing them is submitted.

## 8. Manifest schema changes

```jsonc
{
  "name": "example-module",
  "version": "1.0.0",
  "description": "…",
  // top-level "arguments", "sockets", and "ui" are REMOVED — each moves
  // under its owning program below
  "programs": [
    {
      "name": "producer",
      "command": "{venv_python} {module_dir}/producer.py --rate {arg:rate} --out {socket:frames}",
      "arguments": [
        { "name": "rate", "flag": "--rate", "type": "int", "default": 30 }
      ],
      "sockets": [
        { "name": "frames", "direction": "output", "transport": "unix",
          "pattern": "stream", "stream_kind": "framed", "capped": false,
          "description": "Length-prefixed frames." },
        { "name": "http_api", "direction": "output", "transport": "http",
          "pattern": "request-reply" }
      ],
      "ui": [
        { "id": "rate-form", "type": "form", "title": "PRODUCER RATE",
          "region": "sidebar", "default": true, "socket": "http_api",
          "get": "/api/rate", "put": "/api/rate",
          "fields": [ {"key": "rate", "kind": "number", "min": 1} ] }
      ]
    },
    {
      "name": "consumer",
      "command": "{venv_python} {module_dir}/consumer.py --in {socket:frames}",
      "arguments": [],
      "sockets": [
        { "name": "frames", "direction": "input", "transport": "unix",
          "pattern": "stream", "stream_kind": "framed" }
      ]
    }
  ]
}
```

Other manifest fields (`dependencies`, `install`, `recordings_subdir`) are
unchanged in shape but move where necessary from module-level thinking to
node-instance-level thinking — see §9. `ui` is **not** removed — it moves
from module top level to per-program alongside `arguments`/`sockets`, and
keeps its existing field shapes unchanged (§10).

**Migration path for module authors** (documented in full in
`docs/GRAPH_CONNECTIONS.md`, §11): move each program's slice of the old
top-level `arguments` down into that program's own `arguments[]`; move each
program's slice of the old top-level `ui[]` down into that program's own
`ui[]`; add `direction`/`transport`/`pattern`/`stream_kind`/`capped` to
every existing socket based on what it already is today (e.g. today's MJPEG
`tcp` sockets are wrong to tag as raw `tcp` — MJPEG-over-HTTP becomes
`transport: http, pattern: request-reply`; today's raw frame unix sockets
like `evk4_events.sock` become `transport: unix, pattern: stream,
stream_kind: <pick the right one>`). This is exactly the work explicitly
deferred for
`evk-datalogger`/`picam-datalogger`/`ircam-datalogger`/`gimbal-controller`
per §2.

## 9. Consequential changes: recordings & playback

Recording programs being graph nodes — possibly several nodes of the same
recording program template — breaks today's "one PLAYBACK inner tab per
module (or per copy)" assumption. **PROPOSED**: PLAYBACK's inner tabs are
built from the *active graph's* nodes (one tab per node whose program
declares `recordings_subdir`, recording to
`<recordings_dir>/<subdir>-<node-id>`), not from installed modules.

**Stream Fan-out**, the built-in node from §2, is proposed to ship as a new
program template inside `eventide-core` (fitting the "inside the graph"
decision in §2) rather than as frontend-only sugar: a small program with one
`input` stream socket of *any* `stream_kind` (pass-through) and a
configurable number of `output` stream sockets, all producing an identical
copy of every chunk read from the input. This needs one new small manifest
capability — a **socket count driven by an argument** (e.g. `"count_arg":
"fanout_count"` on a socket definition, generating that many numbered output
slots) — which doesn't exist in the schema at all today and would need to be
designed and added as part of this work.

## 10. MAIN tab UI components — stays manifest-defined, now per-program

The first pass at this (UI widgets becoming their own graph nodes) was
walked back as more complexity than the project needs right now. The
simpler design: **`ui[]` stays in `eventide-module.json`**, using exactly
the widget types and field shapes already documented in `docs/MODULES.md`'s
"Dashboard UI components" table (`mjpeg`, `form`, `telemetry`, `features`,
`recording`, `joystick`, `table`, `map`, `orientation3d`, plus the special
`master-record`/`schedule-table` types) — no schema changes to the widget
config itself, and no new node type, no second wiring layer, no second
compiler artifact.

The one change is where `ui[]` lives and what populates the palette:

- **`ui[]` moves from module top level to per-program**, the same
  relocation as `arguments`/`sockets` (§8). A program's `ui[]` entries keep
  referencing `socket` by name, exactly as today — it just now names one of
  that *program's own* sockets rather than reaching across to any socket
  anywhere in the module.
- **The MAIN tab's ＋ COMPONENTS palette is built from the programs
  currently placed as nodes in the active graph, not from installed
  modules.** Since every node is a concrete instance with its own resolved
  socket addresses, each node contributes its own copy of its program's
  `ui[]` entries to the palette — this is exactly today's per-copy panel
  duplication logic (see "Program copies" in `docs/MODULES.md`), just
  driven by "which nodes exist in the graph" instead of "which copies exist
  for an instanceable program." `default: true` components auto-place when
  a node is added to the running graph (mirroring today's
  auto-place-on-install); removing a node hides its widgets, and re-adding
  it restores them in place, same as a removed/re-added copy today.
- Component traffic still goes through the backend proxy, just re-scoped to
  the node instance rather than the module + copy pair it's keyed off
  today: `/proxy/node/<node-id>/<socket>/<upstream path>` replaces both
  `/proxy/<module>/<socket>/...` and its copy variant
  `/proxy/<module>/copy/<cid>/<socket>/...`, which stop making sense once
  sockets are addressed per graph node rather than per module/copy.
- Per-browser placement/visibility (today's localStorage-per-host layout
  persistence) is otherwise **unaffected** — only the palette's contents
  change (installed modules → active graph nodes); which of those available
  components a given browser has actually placed, and where, keeps working
  exactly as it does today.
- `master-record` still aggregates every currently-placed `recording`-type
  component, just resolved from the active graph's nodes instead of the
  module registry.

## 11. New documentation

`docs/GRAPH_CONNECTIONS.md` — written for module authors, covering:

- The three `stream_kind` groups, with concrete framing/regularity examples
  and why `framed` absorbs the regular/irregular split while unframed data
  doesn't.
- `direction` as a binding/ownership concept, not a naive byte-flow arrow,
  explained with the request-reply case (server = output, client = input)
  since that's the one place the naming is non-obvious.
- Why `http` is its own `transport` rather than "tcp + request-reply".
- The 1:1 cardinality rule for stream sockets, the Stream Fan-out node, and
  the `capped` tag / persistent unconnected-socket warning.
- Per-program `arguments`/`sockets`/`ui[]` placement (replacing the old
  top-level fields) with a before/after example.
- A migration checklist for existing manifests, including moving `ui[]`
  entries down into the specific program(s) they belong to (§10).

`docs/MODULES.md` gets trimmed of everything the GRAPH tab now owns (install
lifecycle stays; "how supervisor config is generated", program copies, and
network-locations proxy sections are rewritten to reflect §7. The dashboard
UI components section stays largely as-is — same widget types and fields —
amended only to describe per-program placement and the graph-node-driven
palette (§10).

## 12. Rough workstreams (not a schedule — for sequencing sanity only)

1. Manifest schema + validation: move `arguments`/`sockets`/`ui[]` under
   `programs[]`, add the socket tags (§6), add fanout socket-count support.
   Migrate `module-template/` and `eventide-core` to the new shape.
2. Backend graph compiler (§7): draft/active graph storage, validation,
   stub/warning handling for unconnected stream sockets, address allocation
   (per-edge for streams, per-node for request-reply), command rendering
   reusing the existing placeholder engine, conf generation/apply, per-node
   start/stop/restart/log endpoints, uninstall-cascade regeneration.
3. `eventide-core`'s new Stream Fan-out program.
4. Frontend GRAPH tab: litegraph.js integration, dynamic palette from
   `/api/modules`, typed slots from the tag taxonomy, draft
   autosave/submit, per-node status/controls/log panel, persistent
   unconnected-socket warning badges.
5. MODULES tab: strip supervisor/status UI down to install/uninstall +
   palette listing.
6. PLAYBACK tab: rebuild inner-tab enumeration from active-graph nodes
   instead of installed modules (§9).
7. MAIN tab (§10): rebuild the ＋ COMPONENTS palette from the active
   graph's nodes instead of installed modules, re-scope component proxying
   to `/proxy/node/<node-id>/<socket>/...`, keep per-browser placement
   logic as-is.
8. Write a migration guide for existing modules — a concrete, step-by-step
   walkthrough (separate from the general authoring docs in workstream 9)
   for taking an already-published manifest through the moves described in
   §8's migration path: relocating `arguments`/`sockets`/`ui[]` under each
   `programs[]` entry, and tagging every existing socket with
   `direction`/`transport`/`pattern`/`stream_kind`/`capped` per §6. Written
   so `evk-datalogger`, `picam-datalogger`, `ircam-datalogger`, and
   `gimbal-controller` (out of scope for this plan to migrate itself, per
   §2) can each be updated later by following it without re-deriving any
   design decisions.
9. Documentation: `docs/GRAPH_CONNECTIONS.md`, `docs/MODULES.md` rewrite,
   root `README.md` updates.

## 13. Open questions summary

All open questions from earlier revisions are now resolved:

- **§6** `stream_kind` on an input socket is explicitly a single value, not
  a set — see the resolved note there.
- (§10's earlier open sub-questions — multi-socket UI nodes, per-browser
  layout loss, proxy path scoping for UI-as-nodes — no longer apply now that
  `ui[]` stays manifest-defined instead of becoming graph nodes.)

Nothing outstanding is blocking the start of workstream 1 (§12).
