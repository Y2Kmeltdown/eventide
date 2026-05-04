"""
Trigger counter
===============
Connects to the EVK4 triggers Unix socket, counts trigger events over a
configurable time window, and prints a summary at the end of each window.

Wire format (matches the Rust pipeline):
  - 4 bytes  : u32 little-endian payload length
  - N x 26 bytes : packed TriggerEvent structs

TriggerEvent — 26 bytes:
  system_time      : u64 LE
  system_timestamp : u64 LE
  t                : u64 LE  — sensor timestamp (µs)
  id               : u8
  rising           : u8

Usage:
    python trigger_counter.py
    python trigger_counter.py --socket /tmp/evk4_triggers.sock --window 5.0
"""

import argparse
import socket
import struct
import time

TRIGGER_FORMAT = "<QQQbb"
TRIGGER_SIZE   = struct.calcsize(TRIGGER_FORMAT)  # 26 bytes


def recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly n bytes, returning None if the connection closes."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def main():
    parser = argparse.ArgumentParser(description="Count EVK4 trigger events per time window")
    parser.add_argument("--socket",  default="/tmp/evk4_triggers.sock", help="Path to the triggers Unix socket")
    parser.add_argument("--window",  type=float, default=1.0,           help="Counting window in seconds (default: 1.0)")
    args = parser.parse_args()

    print(f"Connecting to {args.socket} ...")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(args.socket)
    print(f"Connected. Counting triggers every {args.window}s  (Ctrl+C to stop)\n")

    window_start  = time.monotonic()
    window_count  = 0
    total_count   = 0
    window_number = 1

    try:
        while True:
            # ── Read length prefix ────────────────────────────────────────────
            header = recv_exact(sock, 4)
            if header is None:
                print("Socket closed.")
                break
            (length,) = struct.unpack("<I", header)

            # ── Read payload ──────────────────────────────────────────────────
            payload = recv_exact(sock, length)
            if payload is None:
                print("Socket closed mid-payload.")
                break

            # ── Count triggers in this payload ────────────────────────────────
            n_events = length // TRIGGER_SIZE
            window_count += n_events

            # ── Print and reset when the window expires ────────────────────────
            now = time.monotonic()
            elapsed = now - window_start
            if elapsed >= args.window:
                total_count += window_count
                rate = window_count / elapsed
                print(
                    f"Window {window_number:>4} | "
                    f"duration={elapsed:.3f}s | "
                    f"triggers={window_count:>6} | "
                    f"rate={rate:.1f}/s | "
                    f"total={total_count}"
                )
                window_count  = 0
                window_start  = now
                window_number += 1

    except KeyboardInterrupt:
        # Print any partial window on exit
        if window_count > 0:
            elapsed = time.monotonic() - window_start
            total_count += window_count
            print(
                f"\nPartial window | "
                f"duration={elapsed:.3f}s | "
                f"triggers={window_count} | "
                f"total={total_count}"
            )
        print("\nStopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()