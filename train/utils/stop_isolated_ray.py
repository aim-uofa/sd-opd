#!/usr/bin/env python3
"""Stop processes belonging to one Ray run, preserving other local instances."""

import argparse
import os
from pathlib import Path

import psutil


def find_processes(marker: str, protected: set[int]) -> dict[int, psutil.Process]:
    processes = {}
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.pid in protected:
            continue
        try:
            paths = [arg.split("=", 1)[-1] for arg in process.info["cmdline"] or []]
            matches = any(path == marker or path.startswith(marker + "/") for path in paths)
            # Ray workers may replace their command line with an actor name.
            if matches or process.environ().get("SD_OPD_RAY_RUN_ID") == marker:
                processes[process.pid] = process
                processes.update(
                    (child.pid, child)
                    for child in process.children(recursive=True)
                    if child.pid not in protected
                )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return processes


def stop_isolated_ray(ray_temp_dir: str) -> None:
    protected = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    processes = find_processes(ray_temp_dir, protected)
    print(f"Stopping {len(processes)} processes from Ray run {ray_temp_dir}", flush=True)
    for process in processes.values():
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(processes.values()), timeout=10)

    # Include children that were spawned or orphaned during shutdown.
    remaining = {process.pid: process for process in alive}
    remaining.update(find_processes(ray_temp_dir, protected))
    for process in remaining.values():
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(remaining.values()), timeout=3)
    live_pids = []
    for process in alive:
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                live_pids.append(process.pid)
        except psutil.NoSuchProcess:
            pass
    if live_pids:
        raise SystemExit(f"Ray processes still alive after cleanup: {live_pids}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ray_temp_dir", help="Unique absolute temporary directory for the Ray run.")
    args = parser.parse_args()
    path = Path(args.ray_temp_dir)
    if not path.is_absolute() or path == Path("/"):
        parser.error("ray_temp_dir must be an absolute path other than the filesystem root")
    stop_isolated_ray(str(path))


if __name__ == "__main__":
    main()
