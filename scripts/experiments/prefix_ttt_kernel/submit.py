"""Local experiment driver: immutable per-phase queues under one Supervisor."""
import json, pathlib, subprocess, sys, hashlib
root=pathlib.Path(__file__).resolve().parents[3]; work=root/'artifacts/mamba-kernel'
name=sys.argv[1]; jobs=json.loads(pathlib.Path(sys.argv[2]).read_text())
gpu_ids=','.join(str(int(value)) for value in (sys.argv[3] if len(sys.argv)>3 else '7').split(','))
phase=work/name; phase.mkdir(exist_ok=False)
evidence=root/'artifacts/experiments/prefix_ttt_kernel/evidence'/('mamba-'+name)
evidence.mkdir(parents=True,exist_ok=True)
env=json.loads((work/'environment.json').read_text())
env['CUDA_VISIBLE_DEVICES']=gpu_ids
manifest={'cwd':str(root),'env':env,'jobs':jobs}
manifest_path=evidence/'manifest.json'; manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('src/prefix_ttt','tests','scripts/experiments/prefix_ttt_kernel') for p in (root/folder).rglob('*.py')}
(evidence/'source-sha256.json').write_text(json.dumps(sources,indent=2)+'\n')
conf=f'''[program:{name}]
command=/data/mjyang/.local/bin/uv run --locked --no-sync python -m prefix_ttt.scheduler --manifest {manifest_path} --database {phase}/queue.sqlite3 --gpu-ids {gpu_ids} --max-tasks-per-gpu 1 --poll-seconds 2
directory={root}
autostart=false
autorestart=false
startsecs=0
startretries=0
stopsignal=TERM
stopwaitsecs=30
stopasgroup=true
killasgroup=true
stdout_logfile={phase}/consumer.log
redirect_stderr=true
environment=PATH="{env['PATH']}",CUDA_HOME="{env['CUDA_HOME']}",LD_LIBRARY_PATH="{env['LD_LIBRARY_PATH']}",HF_ENDPOINT="{env['HF_ENDPOINT']}",CUDA_VISIBLE_DEVICES="{gpu_ids}",PREFIX_TTT_QUEUE_ROOT="{phase}",UV_CACHE_DIR="{env['UV_CACHE_DIR']}"
'''
(work/'programs'/f'{name}.ini').write_text(conf); (evidence/'program.ini').write_text(conf)
ctl=['/data/mjyang/.pixi/bin/supervisorctl','-c',str(work/'supervisord.conf')]
for command in (['reread'],['update'],['start',name]): subprocess.run(ctl+command,check=True)
with (root/'docs/experiments/2026-09-11-prefix-ttt-kernel/ledger.md').open('a') as f:
 f.write(f'\n### 队列提交：{name}\n\n- 作业：{", ".join(j["id"] for j in jobs)}；GPU {gpu_ids}，单卡并发 1，无自动重试。\n- 配置/源码哈希：`artifacts/experiments/prefix_ttt_kernel/evidence/mamba-{name}/`；SQLite/原始日志：`artifacts/mamba-kernel/{name}/`。\n- 状态：已由 Supervisor 启动；完成与失败另追加实测记录。\n')
