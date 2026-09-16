# Graph-defined supervisor: design & implementation plan

Status: **draft for review** — sections marked **PROPOSED** are this plan's
best design given the answers gathered so far; they are not yet implemented
and should be corrected before work starts if they're wrong. Sections marked
**OPEN QUESTION** are decisions this plan deliberately leaves for the user
because they weren't resolved in the scoping conversation.

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

These were resolved via clarifying questions before this plan was written:

| Question | Decision |
| -------- | -------- |
| How does a graph edge become a real connection? | **Direct address injection** — eventide generates the concrete address (TCP port / unix path) for a connection and substitutes it into both endpoints' rendered commands, the same way `{socket:name}` works today, just scoped to a specific wire instead of a whole module. No broker/proxy process sits in the data path for raw socket streams. |
| Fan-out / fan-in cardinality | Unix (and, by extension, raw TCP) stream sockets are **strictly one-to-one**. A built-in **Stream Fan-out** node type takes one stream input and duplicates it to N outputs, for when the user needs to split a stream. HTTP sockets fan out natively (a server can serve any number of callers) and need no special node. Fan-*in* (merging multiple upstream outputs into one input) is **not supported** in this plan — see §9. |
| Migration of existing hardware modules | Out of scope for this plan. Only the base platform, `module-template/`, and the in-repo `eventide-core` module are migrated to the new manifest schema. `evk-datalogger`, `picam-datalogger`, `ircam-datalogger`, and `gimbal-controller` (separate repos) keep their current manifests and **will not install or run correctly against the new backend** until someone migrates them — this plan documents what that migration involves (§8) but doesn't do it. |
| Editing a running graph | **Full stop-and-regenerate on every submit.** Submitting a graph tears down and re-renders the entire generated supervisor config, exactly like today's "reinstall = full re-render" model. No diffing, no partial hot-reload. |
| Is `eventide-core` inside or outside the graph? | **Inside.** The playback server, the recording scheduler, and (per §9) the new stream-fanout utility are themselves graph nodes, placed and wired like any other module's programs — not fixed always-on infrastructure. |

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
   installed modules (§5). Node params = that program's `arguments`. Node
   sockets = that program's `sockets`, split by `direction` into litegraph
   input/output slots.
2. **New backend: graph compiler** (§7) — turns a submitted graph into a
   rendered supervisor conf: one `[program:...]` block per node, addresses
   allocated per edge/socket, then `supervisorctl reread && update`.
3. **MODULES tab loses supervisor duties.** It becomes purely an install /
   uninstall / palette-management screen (install job flow unchanged); the
   live program list, start/stop/restart, and status/log viewing move to the
   GRAPH tab (§4.4), operating on graph nodes instead of module programs.
4. **GRAPH tab, once submitted:** each node shows live status
   (RUNNING/STOPPED/FATAL/…), START/STOP/RESTART buttons, and OUT/ERR buttons
   that open that node's log in a right-hand panel — the same interaction
   MODULES has today, just keyed by node instance instead of module program.
5. **`arguments` moves from module top level to per-program.** Each entry in
   `programs[]` gets its own `arguments[]` array (same field shape as
   today). This is required because two programs in the same module can now
   be placed as independently-configured nodes with no shared "module args"
   concept — every node instance owns its own argument values.
6. **New README**: `docs/GRAPH_CONNECTIONS.md` (§10) documenting the socket
   tag taxonomy, direction/binding semantics, the stream-kind distinctions,
   and how to author a module against this system.
7. **`sockets[]` gets tags** — `direction`, `transport`, `pattern`, and
   (when applicable) `stream_kind`. Full taxonomy in §6.

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
  from `transport` + `pattern` + `stream_kind`, e.g.
  `unix/stream/framed-regular` — the editor refuses to connect two slots
  whose types aren't compatible (§6 defines compatibility precisely).
- **PROPOSED**: placing a node auto-generates a short instance id (slug of a
  user-editable label, unique within the graph — same slugification rule
  `instance`/copies uses today) used for supervisor program naming, log file
  naming, and (if the program declares `recordings_subdir`) its own
  recordings subdirectory, exactly as today's `<program>-<cid>` /
  `<subdir>-<cid>` scheme, just derived from graph placement instead of an
  instance-argument value.

## 6. Socket tag taxonomy — PROPOSED

Every entry in a program's `sockets[]` gains:

| Field | Values | Meaning |
| ----- | ------ | ------- |
| `direction` | `"output"` \| `"input"` | **`output`** = this program *hosts/owns* the address — it binds a unix path, listens on a TCP port, or runs the HTTP server. **`input`** = this program *attaches to* an address owned by whatever output socket it's wired to. This is a binding/ownership direction, not necessarily a byte-flow direction — for a `request-reply` socket the "output" side is the server and the "input" side is the client that calls it. |
| `transport` | `"unix"` \| `"tcp"` \| `"http"` | Physical mechanism. `http` is modelled distinctly from raw `tcp` because it's always `request-reply` and fans out natively (§2). |
| `pattern` | `"stream"` \| `"request-reply"` | Captures "free-flowing" vs "programs want to request data" from the original brief. `http` sockets are always `request-reply`. `unix`/`tcp` sockets declare one or the other explicitly. |
| `stream_kind` | `"unframed-irregular"` \| `"framed-irregular"` \| `"unframed-regular"` \| `"framed-regular"` | **Required** when `transport` is `unix` or `tcp` and `pattern` is `stream`; omitted otherwise. Exactly the four cases from the brief: whether a length-prefix frame precedes each chunk, and whether chunks are a fixed size. |

**Connection compatibility** (enforced both client-side in the litegraph
editor via slot type strings, and re-validated server-side on submit — never
trust the browser alone):

- `transport` must match exactly.
- `pattern` must match exactly.
- For `stream` sockets, the input side's `stream_kind` must be able to
  process what the output side produces. **OPEN QUESTION**: should an input
  socket declare exactly one accepted `stream_kind`, or a set (e.g. a
  generic recorder that accepts any `*-regular` stream)? This plan defaults
  to **exact match only** (simplest, matches "these distinctions define what
  a downstream program is capable of processing" read literally) but a
  small change makes it a set if you want more flexible nodes later.
- **Cardinality**: an `output` socket with `transport ∈ {unix, tcp}` and
  `pattern: stream` accepts **at most one** connected edge (§2) — enforced
  in the editor (no second wire allowed) and re-checked on submit. `http`
  outputs and any `request-reply` socket accept any number of edges. There
  is no fan-in in this plan: an `input` socket, regardless of type, accepts
  at most one incoming edge.

## 7. Backend: the graph compiler

New endpoints, replacing the module-conf-per-install flow for anything that
was previously auto-started:

- `GET /api/graph` — the currently active (last-submitted) graph, plus a
  separately-persisted **draft** the user is still editing (`PUT
  /api/graph/draft`, autosaved, has no effect on the running system).
- `POST /api/graph/submit` — validates the draft (§6 compatibility rules,
  unknown module/program references, required-but-unconnected sockets — see
  open question below, duplicate instance ids) and, if valid:
  1. Stops everything currently running from the previous graph.
  2. Allocates addresses: a fresh TCP port from the existing pool
     (`--port-pool`) for every `tcp`/`http` output socket that doesn't
     already have one; a fresh generated unix path
     (`/tmp/eventide-<node-id>-<socket>.sock`) for every `unix` output
     socket; **one per edge** for `stream`-pattern sockets (since they're
     1:1, "the edge's address" and "the output socket's address" are the
     same thing and can be allocated once, when the edge is drawn); **one
     per node** (not per edge) for `request-reply`/`http` output sockets,
     reused across every client edge — this is exactly today's
     `{socket:name}` allocation, just scoped to a node instead of a module.
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
- Per-node control, replacing the module program endpoints for anything
  graph-driven: `POST /api/graph/nodes/<node_id>/start|stop|restart`, log
  tail endpoints mirroring whatever MODULES already exposes for
  stdout/stderr (reuse verbatim — logs are still just
  `/var/log/supervisor/<program_name>.log`, `program_name` is now derived
  from the node id).
- `GET /api/graph/status` — live supervisor status per node (thin wrapper
  around the existing `supervisor_statuses()`), polled by the GRAPH tab the
  same way MODULES polls today.

**Module install** keeps doing exactly what it does now (clone/extract →
deps → build → artifacts) but **stops at "configuring"** — it no longer
renders or starts any supervisor program. A module is "installed" the moment
its program templates are available in the palette; nothing runs until a
graph referencing them is submitted.

**PROPOSED**: uninstalling a module whose program templates are used by
nodes in the *active* graph is rejected with a clear error (`400`, "module
X is used by N node(s) in the running graph — remove them and resubmit
first") rather than silently orphaning running programs. **OPEN QUESTION**:
should uninstall instead be allowed to also drop those nodes from the graph
and resubmit automatically? This plan defaults to the conservative "refuse"
behaviour.

## 8. Manifest schema changes

```jsonc
{
  "name": "example-module",
  "version": "1.0.0",
  "description": "…",
  // top-level "arguments" and "sockets" are REMOVED
  "programs": [
    {
      "name": "producer",
      "command": "{venv_python} {module_dir}/producer.py --rate {arg:rate} --out {socket:frames}",
      "arguments": [
        { "name": "rate", "flag": "--rate", "type": "int", "default": 30 }
      ],
      "sockets": [
        { "name": "frames", "direction": "output", "transport": "unix",
          "pattern": "stream", "stream_kind": "framed-regular",
          "description": "Fixed-size frames, length-prefixed." }
      ]
    },
    {
      "name": "consumer",
      "command": "{venv_python} {module_dir}/consumer.py --in {socket:frames}",
      "arguments": [],
      "sockets": [
        { "name": "frames", "direction": "input", "transport": "unix",
          "pattern": "stream", "stream_kind": "framed-regular" }
      ]
    }
  ]
}
```

Other manifest fields (`dependencies`, `install`, `recordings_subdir`, `ui`)
are unchanged in shape but move where necessary from module-level thinking
to node-instance-level thinking — see §9 for the two that actually need
behavioural changes, not just relocation.

**Migration path for module authors** (documented in full in
`docs/GRAPH_CONNECTIONS.md`, §10): move each program's slice of the old
top-level `arguments` down into that program's own `arguments[]`; add
`direction`/`transport`/`pattern`/`stream_kind` to every existing socket
based on what it already is today (e.g. today's MJPEG `tcp` sockets become
`transport: tcp` is wrong — MJPEG-over-HTTP is `transport: http,
pattern: request-reply`; today's raw frame unix sockets like
`evk4_events.sock` become `transport: unix, pattern: stream, stream_kind:
<pick the right one>`). This is exactly the work explicitly deferred for
`evk-datalogger`/`picam-datalogger`/`ircam-datalogger`/`gimbal-controller`
per §2.

## 9. Consequential changes not explicitly in the 7 requirements

Two existing features are built on "module install = fixed running
programs" and break under the graph model unless adapted. Flagging both so
they're decided deliberately rather than discovered mid-implementation:

1. **`recordings_subdir` / PLAYBACK tab.** Today one PLAYBACK inner tab
   exists per module (or per copy). Under the graph model, recording
   programs are graph nodes — possibly several nodes of the same recording
   program template. **PROPOSED**: PLAYBACK's inner tabs are built from the
   *active graph's* nodes (one tab per node whose program declares
   `recordings_subdir`, recording to `<recordings_dir>/<subdir>-<node-id>`),
   not from installed modules.
2. **`ui` dashboard components (MAIN tab).** Today's per-copy panel
   duplication logic (§ "Program copies" in `docs/MODULES.md`) needs the
   same treatment: a `ui` component keyed to a socket is placed once per
   *graph node* that owns that socket, not once per module. **This wasn't
   one of the 7 listed requirements** — it's flagged here because the MAIN
   tab will otherwise not know which running node a component's `socket`
   field refers to once there can be several nodes of the same program.
   **OPEN QUESTION**: is wiring MAIN tab components to graph nodes in scope
   for this project's first pass, or should MAIN tab support be dropped
   entirely (no live component palette until a follow-up), leaving GRAPH +
   MODULES + PLAYBACK as the only tabs touched initially?

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

## 10. New documentation

`docs/GRAPH_CONNECTIONS.md` — written for module authors, covering:

- The four `stream_kind` values, with concrete framing/regularity examples.
- `direction` as a binding/ownership concept, not a naive byte-flow arrow,
  explained with the request-reply case (server = output, client = input)
  since that's the one place the naming is non-obvious.
- Why `http` is its own `transport` rather than "tcp + request-reply".
- The 1:1 cardinality rule for stream sockets and when to reach for the
  Stream Fan-out node instead of trying to wire one output to many inputs.
- Per-program `arguments`/`sockets` placement (replacing the old top-level
  fields) with a before/after example.
- A migration checklist for existing manifests (the bullet list at the end
  of §8).

`docs/MODULES.md` gets trimmed of everything the GRAPH tab now owns (install
lifecycle stays; "how supervisor config is generated", program copies, and
the network-locations proxy sections are rewritten or moved to reflect §7).

## 11. Rough workstreams (not a schedule — for sequencing sanity only)

1. Manifest schema + validation: move `arguments`/`sockets` under
   `programs[]`, add the four socket tags, add fanout socket-count support.
   Migrate `module-template/` and `eventide-core` to the new shape.
2. Backend graph compiler (§7): draft/active graph storage, validation,
   address allocation (per-edge for streams, per-node for request-reply),
   command rendering reusing the existing placeholder engine, conf
   generation/apply, per-node start/stop/restart/log endpoints.
3. `eventide-core`'s new Stream Fan-out program.
4. Frontend GRAPH tab: litegraph.js integration, dynamic palette from
   `/api/modules`, typed slots from the tag taxonomy, draft
   autosave/submit, per-node status/controls/log panel.
5. MODULES tab: strip supervisor/status UI down to install/uninstall +
   palette listing.
6. PLAYBACK tab: rebuild inner-tab enumeration from active-graph nodes
   instead of installed modules (§9.1).
7. Documentation: `docs/GRAPH_CONNECTIONS.md`, `docs/MODULES.md` rewrite,
   root `README.md` updates.
8. Decide and implement §9.2 (MAIN tab `ui` components) — scope depends on
   the open question there.

## 12. Open questions summary

Collected from above, for a single pass of sign-off:

1. **§6** — does an input socket's `stream_kind` accept exactly one kind, or
   a set of acceptable kinds?
2. **§7** — is submitting a graph with a required socket left unconnected a
   hard validation error, or is it allowed (the node just won't get that
   `{socket:name}` value and may fail at runtime)?
3. **§7** — should uninstalling a module in use by the active graph be
   refused outright, or allowed with automatic removal of the affected
   nodes + resubmit?
4. **§9.2** — is MAIN tab `ui` component support (re-keyed to graph nodes
   instead of modules) in scope for this project's first pass, or deferred?
5. Node instance identity: auto-slug from a user-editable label per node
   (as proposed in §5), or some other scheme (e.g. numbered purely by
   placement order)?
