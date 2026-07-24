#!/usr/bin/env python3
"""
hello_module.py — minimal Eventide module example.

Prints a timestamped heartbeat line to stdout every `--interval` seconds.
Supervisor captures stdout into /var/log/supervisor/hello_module.log, so the
heartbeat is visible from the dashboard's SUPERVISOR tab.

This is the simplest possible valid module program: a long-running process
that stays in the foreground and never daemonises.
"""

import argparse
import time
from datetime import datetime, timezone


def main() -> None:
    parser = argparse.ArgumentParser(description="Eventide hello-module example")
    parser.add_argument("--message", default="hello from eventide",
                        help="Text printed on each heartbeat.")
    parser.add_argument("--interval", type=int, default=5,
                        help="Seconds between heartbeats.")
    args = parser.parse_args()

    while True:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{stamp} UTC] {args.message}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
