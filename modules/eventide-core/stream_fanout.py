#!/usr/bin/env python3
"""
stream_fanout.py — eventide-core Stream Fan-out node.

Reads a byte stream from one unix socket and duplicates every chunk it
reads, verbatim, to any number of other unix sockets. It never inspects the
bytes it copies, so the same script backs all three of the manifest's
stream_fanout_<stream_kind> program entries (irregular/regular/framed,
docs/GRAPH_SUPERVISOR_PLAN.md §9) — they only differ in which fixed
stream_kind their sockets declare for the graph's type system.

This is the platform's way around the 1:1 cardinality rule for unix stream
sockets (docs/GRAPH_SUPERVISOR_PLAN.md §2/§6): a stream output can only ever
feed one downstream consumer, so splitting one stream to several consumers
means inserting one of these nodes between the producer and however many
consumers are needed.

The socket this program's *input* is wired to is owned by whatever produces
the stream, so this program is the client there: it dials in, retrying
until the producer is listening. Each *output* socket is instead owned by
this program (a normal unix stream socket server): this program accepts at
most one connected consumer per output — matching that each is itself an
ordinary 1:1 stream socket, one per numbered slot the manifest's
`count_arg` mechanism generates — and re-accepts if that consumer
disconnects. Each output's manifest entry is declared "capped": writing to
an output with no consumer currently connected is a silent no-op rather
than a blocking/erroring write, which is exactly what this program does.

Usage:
    stream_fanout.py --in /tmp/upstream.sock --out /tmp/a.sock,/tmp/b.sock
"""

import argparse
import os
import socket
import threading
import time


def log(msg: str) -> None:
    print(f"[stream_fanout] {msg}", flush=True)


class OutputListener:
    """Owns one output unix socket. Binds and listens immediately; accepts
    connections in a background thread, keeping at most one at a time (a
    fresh connection replaces whatever was there). write() fans out to
    whatever consumer is currently connected and is a no-op when there is
    none — the "capped" behaviour this socket's manifest entry declares."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn: socket.socket | None = None
        self._server = self._bind(path)
        threading.Thread(target=self._accept_loop, daemon=True).start()

    @staticmethod
    def _bind(path: str) -> socket.socket:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(path)
        srv.listen(1)
        return srv

    def _accept_loop(self) -> None:
        while True:
            conn, _ = self._server.accept()
            log(f"consumer connected: {self.path}")
            with self._lock:
                old, self._conn = self._conn, conn
            if old is not None:
                old.close()

    def write(self, chunk: bytes) -> None:
        with self._lock:
            conn = self._conn
        if conn is None:
            return  # capped: nothing connected — drop, don't block
        try:
            conn.sendall(chunk)
        except OSError:
            log(f"consumer on {self.path} went away")
            with self._lock:
                if self._conn is conn:
                    self._conn = None
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()


def connect_input(path: str) -> socket.socket:
    """Block, retrying, until the upstream producer's socket accepts a
    connection — it may not have bound yet when this program starts, and
    supervisor's own restart policy is what recovers from the producer
    disappearing later (see the retry loop in main())."""
    while True:
        sock_ = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock_.connect(path)
        except OSError as exc:
            sock_.close()
            log(f"waiting for input {path} ({exc}); retrying in 1s")
            time.sleep(1)
            continue
        log(f"connected to input {path}")
        return sock_


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True,
                        help="Unix socket path to read the source stream from")
    parser.add_argument("--out", dest="output_paths", required=True,
                        help="Comma-separated unix socket paths to duplicate the stream to")
    parser.add_argument("--chunk-size", type=int, default=65536)
    args = parser.parse_args()

    out_paths = [p for p in args.output_paths.split(",") if p]
    if not out_paths:
        parser.error("--out must list at least one path")
    listeners = [OutputListener(p) for p in out_paths]
    log(f"fanning {args.input_path} out to {len(listeners)} output(s): "
        f"{', '.join(out_paths)}")

    while True:
        conn = connect_input(args.input_path)
        try:
            while True:
                try:
                    chunk = conn.recv(args.chunk_size)
                except OSError as exc:
                    log(f"input read error ({exc}); reconnecting")
                    break
                if not chunk:
                    log("input closed by producer; reconnecting")
                    break
                for listener in listeners:
                    listener.write(chunk)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
