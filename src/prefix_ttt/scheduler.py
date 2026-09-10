"""Fixed-manifest SQLite GPU queue; Linux single consumer, no automatic retries.

Design adapted from FastWAM schedule_robotwin_seed_search.py, commit
c13e1534ece96093c95e0bff48d4ee6506ec40d9. MIT, copyright 2026 The FastWAM
Authors; full notice in configs/systemd/FASTWAM_LICENSE.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import time


def identity(pid):
    """Boot and process start ticks prevent PID reuse from being mistaken for a job."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() + ":" + fields[19]
    except FileNotFoundError:
        return None


def group_alive(pgid):
    # Include descendants after a launcher exits; ignore unreaped zombies.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                return True
        except (FileNotFoundError, ProcessLookupError):
            continue
    return False


def stop_group(process, grace=2.0):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + grace
        while group_alive(process.pid) and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.02)
        if not group_alive(process.pid):
            break
    process.wait()
    if group_alive(process.pid):
        raise RuntimeError("Process group survived cleanup; resources remain reserved")


class Queue:
    def __init__(self, database, manifest, gpu_ids, capacity=1):
        self.path = Path(database).resolve()
        self.gpus = [str(g) for g in gpu_ids]
        if (not self.gpus or len(set(self.gpus)) != len(self.gpus)
                or any(not re.fullmatch(r"0|[1-9][0-9]*", g) for g in self.gpus)
                or type(capacity) is not int or capacity < 1):
            raise ValueError("Unique physical GPU indices and positive capacity required")
        self.capacity = capacity
        self.manifest = manifest
        self._validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(self.path.suffix + ".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("Another queue consumer holds the lock") from None
        try:
            self.db = sqlite3.connect(self.path)
            self.db.row_factory = sqlite3.Row
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS config (digest TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, spec TEXT NOT NULL, status TEXT NOT NULL,
                    gpus TEXT, pid INTEGER, identity TEXT, started REAL,
                    finished REAL, exit_code INTEGER);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, job TEXT, time REAL, kind TEXT, detail TEXT);
            """)
            payload = dict(manifest=manifest, gpu_ids=self.gpus, capacity=capacity)
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            old = self.db.execute("SELECT digest FROM config").fetchone()
            if old and old[0] != digest:
                raise ValueError("Immutable manifest/resource configuration changed")
            if not old:
                with self.db:
                    self.db.execute("INSERT INTO config VALUES (?,?)", (digest, json.dumps(payload, sort_keys=True)))
                    for job in manifest["jobs"]:
                        self.db.execute("INSERT INTO jobs(id,spec,status) VALUES (?,?,'queued')",
                                        (job["id"], json.dumps(job)))
            self.recover()
        except BaseException:
            self.close()
            raise

    def _validate(self):
        m = self.manifest
        required = {"PATH", "CUDA_HOME", "LD_LIBRARY_PATH", "HF_ENDPOINT"}
        if not required <= m.get("env", {}).keys():
            raise ValueError("Explicit PATH, CUDA_HOME, LD_LIBRARY_PATH, HF_ENDPOINT required")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in m["env"].items()):
            raise ValueError("Environment keys and values must be strings")
        if not Path(m["cwd"]).is_absolute() or not Path(m["cwd"]).is_dir():
            raise ValueError("cwd must be an existing absolute directory")
        names = set()
        for job in m["jobs"]:
            name, argv, count = job["id"], job["argv"], job["gpu_count"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in names:
                raise ValueError("Job IDs must be unique safe names")
            names.add(name)
            if (not isinstance(argv, list) or len(argv) < 3
                    or any(not isinstance(x, str) or not x for x in argv)
                    or not Path(argv[0]).is_absolute() or Path(argv[0]).name != "uv"
                    or argv[1] != "run"):
                raise ValueError("argv must start with absolute uv path and run")
            if type(count) is not int or not 1 <= count <= len(self.gpus):
                raise ValueError("gpu_count exceeds allowed physical GPUs")
            if not 0 < job["timeout_seconds"] < float("inf"):
                raise ValueError("Finite positive timeout required")

    def event(self, job, kind, detail=""):
        self.db.execute("INSERT INTO events(job,time,kind,detail) VALUES (?,?,?,?)",
                        (job, time.time(), kind, detail))

    def recover(self):
        for row in self.db.execute("SELECT * FROM jobs WHERE status IN ('claimed','running')").fetchall():
            if row["status"] == "claimed":
                raise RuntimeError("Uncertain launch window: inspect job before recovering queue")
            if ((row["identity"] is not None and identity(row["pid"]) == row["identity"])
                    or group_alive(row["pid"])):
                raise RuntimeError("Previous job or process group is still live; refusing duplicate dispatch")
            with self.db:
                self.db.execute("UPDATE jobs SET status='interrupted',finished=? WHERE id=?",
                                (time.time(), row["id"]))
                self.event(row["id"], "interrupted", "No live previous process; not automatically retried")

    def claim(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            used = {g: 0 for g in self.gpus}
            for row in self.db.execute("SELECT gpus FROM jobs WHERE status IN ('claimed','running')"):
                for gpu in json.loads(row[0]):
                    used[gpu] += 1
            row = self.db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY rowid LIMIT 1").fetchone()
            available = [g for g in self.gpus if used[g] < self.capacity]
            if row and len(available) >= json.loads(row["spec"])["gpu_count"]:
                assigned = available[:json.loads(row["spec"])["gpu_count"]]
                self.db.execute("UPDATE jobs SET status='claimed',gpus=? WHERE id=?",
                                (json.dumps(assigned), row["id"]))
                self.event(row["id"], "claimed", json.dumps(assigned))
                self.db.commit()
                return json.loads(row["spec"]), assigned
            self.db.commit()
            return None
        except BaseException:
            self.db.rollback()
            raise

    def finish(self, name, status, code):
        with self.db:
            self.db.execute("UPDATE jobs SET status=?,exit_code=?,finished=? WHERE id=?",
                            (status, code, time.time(), name))
            self.event(name, status, str(code))

    def run(self, poll_seconds=1.0):
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        running = {}
        def interrupted(signum, frame):
            # Supervisor signals the group; uv can forward the same signal.
            # A second KeyboardInterrupt must not abort job-group cleanup.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            raise KeyboardInterrupt("Scheduler received termination signal")
        previous = signal.signal(signal.SIGTERM, interrupted)
        previous_int = signal.signal(signal.SIGINT, interrupted)
        try:
            while True:
                for name, (process, started, timeout) in list(running.items()):
                    code = process.poll()
                    if code is None and time.monotonic() - started < timeout:
                        continue
                    status = "timed_out" if code is None else ("succeeded" if code == 0 else "failed")
                    stop_group(process)
                    self.finish(name, status, process.returncode)
                    del running[name]
                while item := self.claim():
                    spec, gpus = item
                    name = spec["id"]
                    env = dict(self.manifest["env"], CUDA_VISIBLE_DEVICES=",".join(gpus),
                               PREFIX_TTT_QUEUE_ROOT=str(self.path.parent))
                    try:
                        with (self.path.parent / (name + ".log")).open("ab") as log:
                            process = subprocess.Popen(spec["argv"], cwd=self.manifest["cwd"], env=env,
                                stdout=log, stderr=subprocess.STDOUT, shell=False, start_new_session=True)
                        running[name] = process, time.monotonic(), spec["timeout_seconds"]
                        with self.db:
                            self.db.execute("UPDATE jobs SET status='running',pid=?,identity=?,started=? WHERE id=?",
                                            (process.pid, identity(process.pid), time.time(), name))
                            self.event(name, "started", str(process.pid))
                    except OSError as error:
                        with self.db:
                            self.event(name, "launch_error", str(error))
                        self.finish(name, "failed", None)
                if not running:
                    break
                time.sleep(poll_seconds)
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                for name, (process, _, _) in running.items():
                    stop_group(process)
                    self.finish(name, "interrupted", process.returncode)
            finally:
                signal.signal(signal.SIGTERM, previous)
                signal.signal(signal.SIGINT, previous_int)

    def close(self):
        if hasattr(self, "db"):
            self.db.close()
        self.lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--max-tasks-per-gpu", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    queue = Queue(args.database, json.loads(Path(args.manifest).read_text()),
                  args.gpu_ids.split(","), args.max_tasks_per_gpu)
    try:
        queue.run(args.poll_seconds)
    finally:
        queue.close()


if __name__ == "__main__":
    main()
