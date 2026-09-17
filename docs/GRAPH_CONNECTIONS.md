# Eventide Graph Connections

This is the authoring reference for a module's **sockets** and **arguments**
under the graph-based supervisor: what each socket tag means, why they're
shaped the way they are, and how to declare them. For the platform-level
design (why any of this exists, how the graph compiler turns a submitted
graph into a running system, the backend API) see
[docs/GRAPH_SUPERVISOR_PLAN.md](GRAPH_SUPERVISOR_PLAN.md). For the rest of
the manifest (`dependencies`, `install`, `recordings_subdir`, the module
lifecycle, the backend API) see [docs/MODULES.md](MODULES.md). This doc
covers:

- [Why a graph, in one paragraph](#why-a-graph-in-one-paragraph)
- [Arguments and sockets are per-program now](#arguments-and-sockets-are-per-program-now)
- [Socket tags](#socket-tags)
  - [`direction`: ownership, not byte-flow](#direction-ownership-not-byte-flow)
  - [`transport`: why `http` isn't just `tcp`](#transport-why-http-isnt-just-tcp)
  - [`pattern`: stream vs. request-reply](#pattern-stream-vs-request-reply)
  - [`stream_kind`: three groups, not four](#stream_kind-three-groups-not-four)
  - [`capped`: promising you won't block on a missing reader](#capped-promising-you-wont-block-on-a-missing-reader)
  - [`port` / `path`](#port--path)
  - [`count_arg`: a variable number of slots](#count_arg-a-variable-number-of-slots)
- [Putting it together: a worked example](#putting-it-together-a-worked-example)
- [The Stream Fan-out node](#the-stream-fan-out-node)
- [Connection compatibility rules, summarised](#connection-compatibility-rules-summarised)
- [Migration checklist for an existing manifest](#migration-checklist-for-an-existing-manifest)

---

## Why a graph, in one paragraph

The set of supervisor programs actually running on a payload, and the
connections between them, is defined by a graph the user builds in the
dashboard's **GRAPH** tab: a **node** is one placed instance of a program
from an installed module, and a **wire** between two nodes' sockets is a
real connection (a unix socket, a TCP port, or an HTTP call) materialised
when the graph is submitted. Installing a module no longer starts
anything by itself — it just adds that module's programs to the GRAPH
tab's palette as node types. What you're declaring in a program's
`arguments`/`sockets` is the shape of that node type: what knobs it has,
and what it can be wired to.

## Arguments and sockets are per-program now

Every program in `programs[]` declares its **own** `arguments` and
`sockets` — there's no more module-level `arguments`/`sockets` array
shared across programs. This matters because two programs in the same
module can now be placed as independent nodes with independently-edited
argument values, and a program's command can only ever reference its own
declarations:

```jsonc
{
  "name": "example-module",
  "version": "1.0.0",
  "description": "…",
  "programs": [
    {
      "name": "producer",
      "command": "{venv_python} {module_dir}/producer.py --rate {arg:rate} --out {socket:frames}",
      "arguments": [
        { "name": "rate", "flag": "--rate", "type": "int", "default": 30 }
      ],
      "sockets": [
        { "name": "frames", "direction": "output", "transport": "unix",
          "pattern": "stream", "stream_kind": "framed",
          "description": "Length-prefixed frames." }
      ]
    },
    {
      "name": "consumer",
      "command": "{venv_python} {module_dir}/consumer.py --in {socket:frames}",
      "sockets": [
        { "name": "frames", "direction": "input", "transport": "unix",
          "pattern": "stream", "stream_kind": "framed" }
      ]
    }
  ]
}
```

`{arg:<name>}` and `{socket:<name>}` in a program's `command` (or
`directory`) resolve against **that program's own** `arguments`/`sockets`
— referencing a name declared on a sibling program is a validation error,
same as referencing one that doesn't exist at all. `ui[]` is per-program
too — see `docs/MODULES.md`'s "Dashboard UI components" for its field
reference, which is unchanged; only where it's declared has moved.

There's no more `instance` field, and no more "copies" of a program:
placing a **second node** of the same program template on the canvas *is*
how you run a second instance (a second serial device, a second camera —
each node gets its own argument values, its own allocated addresses, and,
if the program uses `{recordings_subdir}`, its own recordings directory
named `<subdir>-<node id>`).

## Socket tags

Every entry in a program's `sockets[]` needs:

| Field | Values | Required when |
| ----- | ------ | ------------- |
| `name` | any string, unique within the program | always |
| `direction` | `"output"` \| `"input"` | always |
| `transport` | `"unix"` \| `"tcp"` \| `"http"` | always |
| `pattern` | `"stream"` \| `"request-reply"` | `transport` is `unix` or `tcp` (omit for `http` — see below) |
| `stream_kind` | `"irregular"` \| `"regular"` \| `"framed"` | `pattern` is `"stream"` (omit otherwise) |
| `capped` | `true` \| `false` | optional; only meaningful on a `stream`, non-`http` **output** socket |
| `port` | 1–65535 | optional, `tcp`/`http` only — omit to let eventide pool-allocate one |
| `path` | absolute path | optional, `unix` only — omit to let eventide generate one |
| `count_arg` | name of a declared `int` argument | optional — see [below](#count_arg-a-variable-number-of-slots) |

### `direction`: ownership, not byte-flow

`direction` says which side **owns the address** — binds the unix path,
listens on the TCP port, or runs the HTTP server — not which way bytes
happen to flow:

- **`output`** = this program hosts the address. When the graph is
  submitted, eventide allocates (or uses the declared `port`/`path` for)
  this socket, and every node wired to it as an input receives that same
  address.
- **`input`** = this program attaches to an address some other node's
  output socket owns.

For a `stream` socket this lines up with intuition (the producer/writer is
usually the `output`). For a `request-reply` socket it can look backwards
at first: the **server** is the `output` (it's the side that binds/owns
the listening address), and every **client** calling it is an `input` —
even though, per call, the client is the one "sending" a request. Think of
it as "who owns the address," not "who's talking first."

### `transport`: why `http` isn't just `tcp`

`http` is its own transport rather than "`tcp` + `pattern: request-reply`"
for two reasons that both stem from the same fact — an HTTP output
naturally serves any number of simultaneous callers:

- It's always `request-reply` — omit `pattern` on an `http` socket (it's
  set for you); setting it to anything but `request-reply` is a
  validation error.
- Cardinality is unrestricted in both directions: any number of edges can
  connect to an `http` output, unlike a `stream` output's strict 1:1 (see
  below).

A raw `tcp` socket can still be `pattern: "request-reply"` too (e.g. a
custom binary RPC protocol) — it just doesn't get HTTP's implicit fan-out
for free; declare it the same way a `stream` socket would be for
cardinality purposes if that matters to your protocol.

### `pattern`: stream vs. request-reply

This is the "free-flowing vs. request data" distinction from the original
design brief:

- **`stream`** — a continuous, one-directional byte stream with no
  natural start/stop per exchange (camera frames, telemetry, anything you'd
  otherwise pipe). Strictly **1:1**: an output socket accepts at most one
  connected edge.
- **`request-reply`** — discrete calls with a response each time (a
  settings API, a query). `http` sockets are always this; a raw `tcp`
  socket can be too.

### `stream_kind`: three groups, not four

Required on every `stream`-pattern `unix`/`tcp` socket, and it's **not**
a four-way product of "framed or not" × "regular or not" — it's three
mutually exclusive, mutually incompatible groups:

| Value | Meaning |
| ----- | ------- |
| `irregular` | Unframed, variable-sized chunks. There is no header — the consumer only knows a chunk's boundary by however your protocol otherwise implies it (e.g. it's a message-oriented transport, or the consumer parses the payload itself to find the end). |
| `regular` | Unframed, **fixed**-sized chunks. No header either, but the consumer can safely `read()` exactly N bytes at a time because every chunk is the same size — this is what actually makes unframed data parseable at all when the size isn't self-describing. |
| `framed` | Every chunk is preceded by a small header that says how many bytes follow. Covers both fixed- and variable-sized payloads underneath — once there's a frame header, the consumer never needs to care whether the framed sizes happen to be regular, because the header tells it exactly how much to read either way. |

This is *why* it collapses to three groups instead of four: framing
already subsumes the regular/irregular distinction (a framed consumer
handles both identically), so there's no `framed-regular` vs.
`framed-irregular` — just `framed`. Unframed data keeps the split because,
without a header, regularity is the only thing that makes it parseable at
all.

An input socket declares **exactly one** `stream_kind`, not a set of
acceptable ones — a node that genuinely needs to accept more than one kind
needs a separate input socket (and separate handling) per kind it accepts.

### `capped`: promising you won't block on a missing reader

A `stream`-pattern, non-`http` **output** socket left with no connected
edge is not a validation error — the graph still compiles, and the
producer starts normally with a real (if unused) address. But if your
program does a plain blocking write to that socket with nothing reading
it, it can hang or crash. The GRAPH tab shows a **persistent** warning
badge on any such socket until either it's wired up, or the manifest
tells eventide the producer is already safe:

```jsonc
{ "name": "out", "direction": "output", "transport": "unix",
  "pattern": "stream", "stream_kind": "irregular", "capped": true,
  "description": "Duplicate output — drops writes when nothing's connected." }
```

Set `capped: true` only when your program actually behaves that way —
i.e. it checks for a live connection before writing and simply skips the
write (not blocks, not errors) when there is none. `modules/eventide-core/stream_fanout.py`'s `OutputListener.write()` is a
worked example: it holds at most one client per output and just returns
immediately if nothing's currently connected.

### `port` / `path`

Both are optional and, for almost every module, should stay that way:
eventide allocates a free TCP port from its pool, or generates a unix
path, for every output socket when the graph is submitted. Set an
explicit `port` only when something outside the graph needs a fixed,
predictable address (e.g. a legacy client that isn't graph-aware); set an
explicit `path` only for the equivalent unix case. An explicit `port`
that collides with another node's explicit `port` fails validation at
submit time, not silently.

### `count_arg`: a variable number of slots

A socket can name an `int` argument as its `count_arg`, turning it into a
**template** for N independent, numbered sockets rather than one fixed
socket:

```jsonc
"arguments": [
  { "name": "fanout_count", "flag": "--fanout-count", "type": "int", "default": 2 }
],
"sockets": [
  { "name": "out", "direction": "output", "transport": "unix",
    "pattern": "stream", "stream_kind": "framed", "capped": true,
    "count_arg": "fanout_count" }
]
```

Placing this node with `fanout_count: 3` gives it three independent
output slots — `out1`, `out2`, `out3` — each with its own address and its
own 1:1 cardinality rule; a graph edge targets one specific numbered slot
(`{"node": "fan1", "socket": "out2"}`). Inside the program's own command,
`{socket:out}` (the **base** name, not a numbered one) resolves to every
numbered slot's address joined with commas — `/tmp/…/out1.sock,/tmp/…/out2.sock,/tmp/…/out3.sock`
— since a fixed command string can't otherwise reference a
variable-length list of placeholders. Your program parses that list
itself; see `stream_fanout.py`'s `--out` handling for the reference
implementation.

## Putting it together: a worked example

A camera module with a datalogger (producing raw frames) and an MJPEG
server (consuming them, plus exposing its own settings API):

```jsonc
{
  "name": "mycam",
  "version": "1.0.0",
  "description": "Example camera module.",
  "recordings_subdir": "mycam",
  "programs": [
    {
      "name": "datalogger",
      "command": "{venv_python} {module_dir}/record.py --frames-out {socket:frames} --dir {recordings_subdir}",
      "arguments": [],
      "sockets": [
        { "name": "frames", "direction": "output", "transport": "unix",
          "pattern": "stream", "stream_kind": "regular",
          "description": "Fixed-size raw frames." }
      ]
    },
    {
      "name": "mjpeg_server",
      "command": "{venv_python} {module_dir}/mjpeg.py --frames-in {socket:frames} --bind 0.0.0.0:{socket:api}",
      "arguments": [
        { "name": "quality", "flag": "--quality", "type": "int", "default": 80 }
      ],
      "sockets": [
        { "name": "frames", "direction": "input", "transport": "unix",
          "pattern": "stream", "stream_kind": "regular" },
        { "name": "api", "direction": "output", "transport": "http" }
      ],
      "ui": [
        { "id": "live", "type": "mjpeg", "title": "MYCAM LIVE", "region": "center",
          "default": true, "socket": "api", "path": "/stream" }
      ]
    }
  ]
}
```

On the canvas: place one `datalogger` node and one `mjpeg_server` node,
wire `datalogger.frames` → `mjpeg_server.frames` (both `regular`, both
`unix`/`stream` — a valid edge), and submit. `mjpeg_server`'s `api` socket
needs no wire from anything to work — it's `http`, so the MAIN tab's
`mjpeg` widget (which proxies through it, per the `ui[]` entry) is just
another client calling in, same as any other.

## The Stream Fan-out node

Because stream sockets are strictly 1:1, splitting one stream to several
consumers means inserting a fan-out node between the producer and however
many consumers you need — it's not something an edge can do on its own.
`eventide-core` ships one program per `stream_kind`
(`stream_fanout_irregular`, `stream_fanout_regular`, `stream_fanout_framed`
— three variants rather than one "any-kind" node, since `stream_kind`
matching stays exact-value), all wrapping the same script
(`modules/eventide-core/stream_fanout.py`): one `in` input, a `count_arg`
`out` output producing an identical byte-for-byte copy of every chunk it
reads to each connected consumer. Use whichever variant matches the
stream you're splitting.

## Connection compatibility rules, summarised

An edge from output socket A to input socket B is valid when, and only
when:

1. `A.transport == B.transport`.
2. Their effective `pattern` matches (`http` is always treated as
   `request-reply` here, even though the field is normally omitted on an
   `http` socket).
3. If that pattern is `stream`, `A.stream_kind == B.stream_kind` exactly
   (`irregular`/`regular`/`framed` never mix).
4. Cardinality: if `A` is a `stream` output, it can have at most one
   connected edge in total. `B` (any input, regardless of transport) can
   have at most one connected edge.

The GRAPH tab's node types encode this as litegraph slot "type" strings
(`transport/pattern/stream_kind`), so most mismatches are simply
impossible to wire in the editor — but the backend re-checks all four
rules again on submit regardless, since the editor's checks are a
convenience, not the authority.

## Migration checklist for an existing manifest

If you're updating a module written against the pre-graph manifest shape:

1. Move the top-level `arguments` array down into the `arguments` of
   whichever program(s) actually use each argument (via `{arg:<name>}` in
   their `command`). An argument used by more than one program needs to be
   declared separately on each.
2. Move the top-level `sockets` array down the same way, matching each
   socket to the program that binds/uses it via `{socket:<name>}`.
3. Move the top-level `ui` array down to the program that owns the
   `socket` each entry references.
4. Tag every socket:
   - Was it a `"type": "tcp"` socket fronting an HTTP server (an MJPEG
     stream, a settings API)? → `"transport": "http"` (not `"tcp"` —
     `http` is its own transport, see above), no `pattern` needed.
   - Was it a `"type": "tcp"` socket used for something else? →
     `"transport": "tcp"` plus a `"pattern"`.
   - Was it a `"type": "unix"` socket? → `"transport": "unix"` plus a
     `"pattern"`, plus a `"stream_kind"` if the pattern is `"stream"`
     (pick whichever of `irregular`/`regular`/`framed` actually matches
     what the socket carries).
   - Add `"direction"` to every socket: `"output"` if this program binds
     it, `"input"` if it attaches to another program's output.
5. Remove any `"instance"` key from a program entry — placing multiple
   nodes of that program in the GRAPH tab replaces it. If operators need
   to add instances at runtime the way they could add "copies" before,
   that's now done from the GRAPH tab, not the manifest.
6. Validate: `python3 -m json.tool eventide-module.json > /dev/null`,
   then install and check the GRAPH tab's palette shows the program(s)
   with the arguments/sockets you expect.

This is the exact work still pending for the pre-graph hardware modules
this project doesn't own (`evk-datalogger`, `picam-datalogger`,
`ircam-datalogger`, `gimbal-controller`) — see
`docs/GRAPH_SUPERVISOR_PLAN.md` §2 and workstream 8.
