"""Isolated real Supervisor lifecycle fixtures; no CUDA or production queue use."""

import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import time

import pytest

from prefix_ttt.scheduler import group_alive


REPO = Path(__file__).resolve().parents[1]


def _tool(name, override):
    """Locate a host tool, allowing an explicit override for non-PATH installs."""
    if override in os.environ:
        return Path(os.environ[override])
    found = shutil.which(name)
    return Path(found) if found else REPO / name


UV = _tool('uv', 'PREFIX_TTT_UV')
DAEMON = _tool('supervisord', 'PREFIX_TTT_SUPERVISORD')
CTL = _tool('supervisorctl', 'PREFIX_TTT_SUPERVISORCTL')
pytestmark = pytest.mark.skipif(not DAEMON.exists(), reason='Supervisor not installed')


def eventually(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.2)
    raise AssertionError('Lifecycle condition did not become true')


@pytest.fixture
def service():
    root = Path(tempfile.mkdtemp(prefix='prefix-ttt-supervisor-test-', dir='/var/tmp'))
    print(f'Supervisor evidence: {root}', flush=True)
    config = root / 'supervisord.conf'
    config.write_text(f'''[unix_http_server]
file={root}/control.sock
chmod=0600
[supervisord]
logfile={root}/supervisord.log
pidfile={root}/supervisord.pid
childlogdir={root}
nodaemon=false
[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface
[supervisorctl]
serverurl=unix://{root}/control.sock
[program:queue]
command={UV} run --locked --no-sync python -m prefix_ttt.scheduler --manifest {root}/manifest.json --database {root}/queue.sqlite3 --gpu-ids 0,1,2,3 --poll-seconds 0.1
directory={REPO}
autostart=false
autorestart=false
startsecs=0
startretries=0
stopsignal=TERM
stopwaitsecs=30
stopasgroup=true
killasgroup=true
stdout_logfile={root}/consumer.log
redirect_stderr=true
''')

    def control(*args):
        result = subprocess.run([str(CTL), '-c', str(config), *args],
                                capture_output=True, text=True, timeout=40)
        with (root / 'control.log').open('a') as log:
            log.write(f'{args}: {result.returncode}\n{result.stdout}{result.stderr}')
        return result

    def rows():
        if not (root / 'queue.sqlite3').exists():
            return []
        with sqlite3.connect(root / 'queue.sqlite3') as db:
            db.row_factory = sqlite3.Row
            try:
                return [dict(row) for row in db.execute('SELECT * FROM jobs')]
            except sqlite3.OperationalError:
                return []

    def launch(jobs):
        env = {'HOME': str(Path.home()), 'PATH': f'{REPO}/.venv/bin:/usr/bin:/bin',
               'CUDA_HOME': '/usr/local/cuda-12.8',
               'LD_LIBRARY_PATH': '/usr/local/cuda-12.8/lib64',
               'HF_ENDPOINT': 'https://hf-mirror.com', 'FIXTURE': 'explicit'}
        (root / 'manifest.json').write_text(json.dumps({'cwd': str(REPO), 'env': env, 'jobs': jobs}))
        # Launcher is a separate session and exits; the daemon must keep serving.
        result = subprocess.run([str(DAEMON), '-c', str(config)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True, timeout=15)
        assert result.returncode == 0
        assert control('status').returncode == 3  # queue STOPPED, manager alive
        assert control('start', 'queue').returncode == 0

    try:
        yield root, launch, rows, control
    finally:
        if (root / 'control.sock').exists():
            control('stop', 'queue')
            control('shutdown')
        # Cleanup only fixture-owned recorded job groups, including crash leftovers.
        for row in rows():
            if row['pid'] and group_alive(row['pid']):
                os.killpg(row['pid'], signal.SIGKILL)
                eventually(lambda: not group_alive(row['pid']))
        if (root / 'supervisord.pid').exists():
            pid = int((root / 'supervisord.pid').read_text())
            eventually(lambda: not group_alive(pid))


def job(name, code, count=1, timeout=15):
    return {'id': name, 'argv': [str(UV), 'run', '--no-project', '--offline',
                               '--no-managed-python', 'python', '-c', code],
            'gpu_count': count, 'timeout_seconds': timeout}


def test_detached_execution_and_no_retry(service):
    root, launch, rows, control = service
    code = "import os,json; print(json.dumps({k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','FIXTURE','PREFIX_TTT_QUEUE_ROOT']}))"
    launch([job('ok', code, 4), job('failed', 'raise SystemExit(7)'),
            job('timeout', 'import time; time.sleep(60)', timeout=0.5)])
    eventually(lambda: len(rows()) == 3 and all(r['status'] in {'succeeded', 'failed', 'timed_out'} for r in rows()))
    eventually(lambda: 'EXITED' in control('status').stdout)
    assert {r['id']: r['status'] for r in rows()} == {'ok': 'succeeded', 'failed': 'failed', 'timeout': 'timed_out'}
    env = json.loads((root / 'ok.log').read_text().strip().splitlines()[-1])
    assert env == {'CUDA_VISIBLE_DEVICES': '0,1,2,3', 'FIXTURE': 'explicit', 'PREFIX_TTT_QUEUE_ROOT': str(root)}
    with sqlite3.connect(root / 'queue.sqlite3') as db:
        starts = db.execute("SELECT COUNT(*) FROM events WHERE kind='started'").fetchone()[0]
    assert control('start', 'queue').returncode == 0
    eventually(lambda: 'EXITED' in control('status').stdout)
    with sqlite3.connect(root / 'queue.sqlite3') as db:
        assert db.execute("SELECT COUNT(*) FROM events WHERE kind='started'").fetchone()[0] == starts == 3


def test_graceful_stop_cleans_separate_job_session(service):
    root, launch, rows, control = service
    ready = root / 'ready'
    code = f"import subprocess,sys,time,pathlib; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); pathlib.Path({str(ready)!r}).touch(); time.sleep(60)"
    launch([job('long', code, 4, 90)])
    eventually(lambda: ready.exists() and rows()[0]['status'] == 'running')
    pid = rows()[0]['pid']
    assert group_alive(pid)
    assert control('stop', 'queue').returncode == 0
    assert rows()[0]['status'] == 'interrupted'
    eventually(lambda: not group_alive(pid))
    assert 'STOPPED' in control('status').stdout


def test_forced_consumer_death_refuses_duplicate_dispatch(service):
    root, launch, rows, control = service
    ready = root / 'ready'
    code = f"import pathlib,time; pathlib.Path({str(ready)!r}).touch(); time.sleep(60)"
    launch([job('orphan', code, 4, 90)])
    eventually(lambda: ready.exists() and rows()[0]['status'] == 'running')
    job_pid = rows()[0]['pid']
    consumer_pid = int(control('pid', 'queue').stdout.strip())
    assert os.getpgid(consumer_pid) == consumer_pid
    os.killpg(consumer_pid, signal.SIGKILL)
    eventually(lambda: 'EXITED' in control('status').stdout)
    assert group_alive(job_pid)  # Separate session is outside Supervisor's group.
    assert control('start', 'queue').returncode == 0
    eventually(lambda: 'EXITED' in control('status').stdout)
    assert 'refusing duplicate dispatch' in (root / 'consumer.log').read_text()
    assert rows()[0]['status'] == 'running'
    with sqlite3.connect(root / 'queue.sqlite3') as db:
        assert db.execute("SELECT COUNT(*) FROM events WHERE kind='started'").fetchone()[0] == 1
