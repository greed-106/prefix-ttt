"""Record the actual CPU-stage environment without requiring CUDA or a bus."""
import argparse
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {'argv': argv, 'returncode': result.returncode,
                'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'argv': argv, 'error': str(exc)}


def collect():
    import torch
    packages = {}
    for distribution, module in (
        ('torch', 'torch'), ('torchvision', 'torchvision'), ('triton', 'triton'),
        ('transformers', 'transformers'), ('peft', 'peft'), ('accelerate', 'accelerate'),
        ('flash-linear-attention', 'fla'), ('prefix-ttt', 'prefix_ttt'),
        ('tokenizers', 'tokenizers'), ('safetensors', 'safetensors'),
    ):
        item = {'version': metadata.version(distribution)}
        try:
            item['file'] = importlib.import_module(module).__file__
        except Exception as exc:
            item['import_error'] = f'{type(exc).__name__}: {exc}'
        packages[distribution] = item
    import llava
    packages['copied_llava'] = {'file': llava.__file__}
    repos = {}
    for name in ('LLaVA', 'LaCT', 'Spatial-TTT'):
        directory = Path('preference') / name
        repos[name] = {'path': str(directory.resolve())}
        for field, argv in (('commit', ['rev-parse', 'HEAD']), ('branch', ['branch', '--show-current']),
                            ('remote', ['remote', 'get-url', 'origin']), ('dirty', ['status', '--porcelain'])):
            repos[name][field] = command(['git', '-C', str(directory), *argv])
    quota = Path('/sys/fs/cgroup/cpu.max')
    return {'schema_version': 1, 'recorded_at': datetime.now(timezone.utc).isoformat(),
            'cpu_cgroup_quota': quota.read_text().strip() if quota.exists() else None,
            'platform': platform.platform(), 'python': sys.version, 'executable': sys.executable,
            'uv': command([shutil.which('uv') or 'uv', '--version']), 'packages': packages,
            'cuda': {'torch_build': torch.version.cuda, 'available': torch.cuda.is_available(),
                     'device_count': torch.cuda.device_count(),
                     'toolkit': command(['/usr/local/cuda-12.8/bin/nvcc', '--version']),
                     'devices': [{'name': torch.cuda.get_device_name(i),
                                  'total_memory': torch.cuda.get_device_properties(i).total_memory,
                                  'capability': list(torch.cuda.get_device_capability(i))}
                                 for i in range(torch.cuda.device_count())],
                     'gpu_tests': 'not_executed_by_environment_audit'},
            'systemd_user': command(['systemctl', '--user', 'is-system-running']),
            'systemd_bus': command(['systemctl', '--user', 'list-units', '--type=service', '--no-pager']),
            'reference_repositories': repos,
            'uv_lock_sha256': hashlib.sha256(Path('uv.lock').read_bytes()).hexdigest(),
            'fla_revision': 'c51953382397da5c3b7b8a41e568915b703e2934',
            'flash_attention': 'not_installed; PyTorch SDPA selected; actual GPU kernel not verified'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='docs/experiments/2026-09-09-prefix-ttt/environment.lock.json')
    args = parser.parse_args()
    result = collect()
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': args.output, 'cuda': result['cuda'],
                     'package_import_errors': {k: v['import_error'] for k, v in result['packages'].items()
                                              if 'import_error' in v}}, indent=2))


if __name__ == '__main__':
    main()
