"""CPU subprocess fixtures; GPU IDs are allocation labels, no CUDA is invoked."""
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import time
import unittest

from prefix_ttt.scheduler import Queue, group_alive, identity, stop_group


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.uv = shutil.which("uv")
        self.assertIsNotNone(self.uv)
        self.manifest = {
            "cwd": str(self.root),
            "env": {"PATH": os.environ["PATH"], "CUDA_HOME": "/usr/local/cuda-12.8",
                    "LD_LIBRARY_PATH": "/usr/local/cuda-12.8/lib64", "HF_ENDPOINT": "https://hf-mirror.com",
                    "HOME": os.environ["HOME"], "FIXTURE": "explicit"},
            "jobs": [],
        }

    def job(self, name, code="pass", count=1, timeout=10):
        return {"id": name, "argv": [self.uv, "run", "--no-project", "--no-managed-python",
                "--offline", "python", "-c", code], "gpu_count": count, "timeout_seconds": timeout}

    def queue(self, jobs, gpus=("2", "5"), capacity=1):
        self.manifest["jobs"] = jobs
        queue = Queue(self.root / "queue.sqlite3", self.manifest, gpus, capacity)
        self.addCleanup(queue.close)
        return queue

    def test_atomic_allocation_capacity_and_fifo(self):
        q = self.queue([self.job("one"), self.job("two", count=2)])
        self.assertEqual(q.claim()[1], ["2"])
        self.assertIsNone(q.claim())
        q.finish("one", "succeeded", 0)
        self.assertEqual(q.claim()[1], ["2", "5"])
        self.assertIsNone(q.claim())

    def test_configured_capacity(self):
        q = self.queue([self.job(str(i)) for i in range(3)], gpus=("2",), capacity=2)
        self.assertIsNotNone(q.claim())
        self.assertIsNotNone(q.claim())
        self.assertIsNone(q.claim())

    def test_whitelist_and_argv_validation(self):
        for ids in (("0", "0"), ("-1",), ("00",)):
            with self.assertRaises(ValueError):
                Queue(self.root / "bad", self.manifest, ids)
        self.manifest["jobs"] = [self.job("too_many", count=3)]
        with self.assertRaises(ValueError):
            Queue(self.root / "bad", self.manifest, ["2", "5"])
        self.manifest["jobs"][0]["gpu_count"] = 1
        self.manifest["jobs"][0]["argv"] = ["bash", "-c", "true"]
        with self.assertRaises(ValueError):
            Queue(self.root / "bad", self.manifest, ["2", "5"])

    def test_single_consumer(self):
        q = self.queue([self.job("one")])
        with self.assertRaisesRegex(RuntimeError, "consumer"):
            Queue(q.path, self.manifest, ["2", "5"])

    def test_execution_env_success_failure_timeout_and_resume(self):
        code = "import os,json; print(json.dumps({k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','FIXTURE','PREFIX_TTT_QUEUE_ROOT','SHOULD_NOT_LEAK']}))"
        os.environ["SHOULD_NOT_LEAK"] = "secret-fixture"
        self.addCleanup(os.environ.pop, "SHOULD_NOT_LEAK", None)
        q = self.queue([self.job("ok", code, 2), self.job("bad", "raise SystemExit(7)"),
                        self.job("timeout", "import time; time.sleep(30)", timeout=0.2)])
        q.run(0.02)
        result = {r["id"]: r["status"] for r in q.db.execute("SELECT * FROM jobs")}
        self.assertEqual(result, {"ok": "succeeded", "bad": "failed", "timeout": "timed_out"})
        env = json.loads((self.root / "ok.log").read_text().strip().splitlines()[-1])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,5")
        self.assertEqual(env["FIXTURE"], "explicit")
        self.assertIsNone(env["SHOULD_NOT_LEAK"])
        self.assertEqual(env["PREFIX_TTT_QUEUE_ROOT"], str(self.root))
        before = q.db.execute("SELECT COUNT(*) FROM events WHERE kind='started'").fetchone()[0]
        q.run(0.02)
        self.assertEqual(before, q.db.execute("SELECT COUNT(*) FROM events WHERE kind='started'").fetchone()[0])

    def test_launch_failure(self):
        spec = self.job("missing")
        spec["argv"][0] = "/does-not-exist/uv"
        q = self.queue([spec])
        q.run(0.02)
        self.assertEqual(q.db.execute("SELECT status FROM jobs").fetchone()[0], "failed")

    def test_recovery_dead_live_and_uncertain_launch(self):
        q = self.queue([self.job("recover")])
        q.claim()
        with self.assertRaisesRegex(RuntimeError, "Uncertain"):
            q.recover()
        with q.db:
            q.db.execute("UPDATE jobs SET status='running',pid=?,identity=?", (os.getpid(), identity(os.getpid())))
        with self.assertRaisesRegex(RuntimeError, "still live"):
            q.recover()
        with q.db:
            q.db.execute("UPDATE jobs SET pid=2147483647,identity=NULL")
        q.recover()
        self.assertEqual(q.db.execute("SELECT status FROM jobs").fetchone()[0], "interrupted")
        self.assertIsNone(q.claim())

    def test_manifest_is_immutable(self):
        q = self.queue([self.job("one")])
        q.close()
        self.manifest["jobs"][0]["timeout_seconds"] = 99
        with self.assertRaisesRegex(ValueError, "Immutable"):
            Queue(q.path, self.manifest, ["2", "5"])

    def test_group_cleanup_includes_descendants(self):
        command = [self.uv, "run", "--no-project", "--offline", "python", "-c",
                   "import subprocess,time; subprocess.Popen(['sleep','30']); time.sleep(30)"]
        p = subprocess.Popen(command, start_new_session=True, cwd=self.root)
        try:
            time.sleep(0.3)
            self.assertTrue(group_alive(p.pid))
            stop_group(p)
            self.assertFalse(group_alive(p.pid))
        finally:
            if p.poll() is None:
                stop_group(p)

    def test_consumer_sigterm_and_new_consumer_resume(self):
        q = self.queue([self.job("long", "import time; time.sleep(30)", count=2),
                        self.job("next")])
        q.close()
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest))
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        cmd = [self.uv, "run", "--no-project", "--offline", "python", "-m", "prefix_ttt.scheduler",
               "--manifest", str(manifest_path), "--database", str(q.path), "--gpu-ids", "2,5",
               "--poll-seconds", "0.02"]
        with (self.root / "consumer.log").open("w") as log:
            p = subprocess.Popen(cmd, env=env, cwd=self.root, stdout=log, stderr=log, start_new_session=True)
            try:
                with sqlite3.connect(q.path) as db:
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        row = db.execute("SELECT status,pid FROM jobs WHERE id='long'").fetchone()
                        if row[0] == "running":
                            break
                        time.sleep(0.02)
                    self.assertEqual(row[0], "running")
                    p.send_signal(signal.SIGTERM)
                    p.wait(timeout=5)
                    self.assertEqual(db.execute("SELECT status FROM jobs WHERE id='long'").fetchone()[0], "interrupted")
                    self.assertFalse(group_alive(row[1]))
                resumed = Queue(q.path, self.manifest, ["2", "5"])
                try:
                    resumed.run(0.02)
                    self.assertEqual(dict(resumed.db.execute("SELECT id,status FROM jobs")),
                                     {"long": "interrupted", "next": "succeeded"})
                finally:
                    resumed.close()
            finally:
                if p.poll() is None:
                    stop_group(p)


if __name__ == "__main__":
    unittest.main()
