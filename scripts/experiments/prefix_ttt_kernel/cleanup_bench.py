"""Paired eager complete-layer regression against the pre-cleanup source snapshot.

Use the project GPU queue. Both arms retain all existing optimizations and share
QKV/O weights; only the source implementation differs. No CUDA Graph is used.
"""

import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding

from mamba_bench import DIM, HEADS, HEAD_DIM, difference, file_sha256, measure
from prefix_ttt.model.generation import TransformersHybridCache
from prefix_ttt.model.hybrid import PrefixTTTAttention


ARMS = ('before', 'after')
LENGTHS, STEPS, SEED = (640, 1536), 128, 42


def frozen_modules(snapshot):
    """Bind changed imports to snapshot modules; verify every shared dependency."""
    changed = {'ops/fla.py', 'ops/local.py', 'ops/features.py',
               'ops/prefill.py', 'model/hybrid.py'}
    for path in snapshot.rglob('*.py'):
        relative = path.relative_to(snapshot)
        if relative.as_posix() not in changed:
            if path.read_bytes() != (Path('src/prefix_ttt') / relative).read_bytes():
                raise RuntimeError(f'Shared snapshot dependency changed: {relative}')
    modules = {}
    for relative in ('ops/fla.py', 'ops/local.py', 'ops/features.py',
                     'ops/prefill.py', 'model/hybrid.py'):
        path = snapshot / relative
        name = 'prefix_ttt.' + relative[:-3].replace('/', '.')
        module = ModuleType('cleanup_before_' + name.replace('.', '_'))
        sys.modules[module.__name__] = module
        tree = ast.parse(path.read_text(), filename=str(path))
        body = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module in modules:
                for alias in node.names:
                    module.__dict__[alias.asname or alias.name] = getattr(modules[node.module], alias.name)
            else:
                body.append(node)
        tree.body = body
        exec(compile(tree, str(path), 'exec'), module.__dict__)
        modules[name] = module
    return modules


def cache_fields(cache):
    return dict(vars(cache.storage.layers[0]), seen_tokens=cache.storage.seen_tokens,
                finished=cache.storage.finished, valid_history=cache.valid_history)


class PairedLayer:
    def __init__(self, modules, weights):
        config = LlamaConfig(hidden_size=DIM, num_attention_heads=HEADS,
                             num_key_value_heads=HEADS, max_position_embeddings=4096,
                             attention_dropout=0.0, attention_bias=False)
        self.layer = PrefixTTTAttention(LlamaAttention(config, 0).cuda().bfloat16(), backend='fla')
        self.layer.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        self.layer.eval().requires_grad_(False)
        current = self.layer.prefix_ttt
        previous = modules['prefix_ttt.ops.features'].FeatureReadout(DIM, HEADS, HEAD_DIM).cuda()
        previous.load_state_dict(current.state_dict(), strict=True)
        previous.eval().requires_grad_(False)
        for module in (previous, current):
            module.prepare_inference()
        self.features = dict(before=previous, after=current)
        self.forwards = dict(before=modules['prefix_ttt.model.hybrid'].PrefixTTTAttention.forward,
                             after=PrefixTTTAttention.forward)
        self.rope = LlamaRotaryEmbedding(config, device='cuda')

    @contextmanager
    def use(self, arm):
        saved = self.layer.prefix_ttt
        self.layer.prefix_ttt = self.features[arm]
        try:
            yield
        finally:
            self.layer.prefix_ttt = saved

    def forward(self, arm, x, cache=None):
        cache = TransformersHybridCache(1, [], x.device) if cache is None else cache
        valid = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        _, positions = cache.begin(valid)
        output, _ = self.forwards[arm](self.layer, x, self.rope(x, positions),
                                      past_key_value=cache, prefix_valid_mask=valid, use_cache=True)
        cache.finish()
        return output, cache

    def decode(self, arm, continuation, cache):
        for step in range(STEPS):
            output, cache = self.forward(arm, continuation[:, step:step + 1], cache)
        return output, cache


def correctness(layer, prefix, continuation):
    caches = dict.fromkeys(ARMS)
    rows = []
    for step in range(STEPS + 1):
        x = prefix if step == 0 else continuation[:, step - 1:step]
        outputs, unchanged = {}, {}
        for arm in ARMS:
            old = {} if caches[arm] is None else vars(caches[arm].storage.layers[0])
            copies = {key: value.clone() for key, value in old.items()}
            with layer.use(arm):
                outputs[arm], caches[arm] = layer.forward(arm, x, caches[arm])
            unchanged[arm] = all(torch.equal(value, copies[key]) for key, value in old.items())
        reference, actual = (cache_fields(caches[arm]) for arm in ARMS)
        fields = {key: torch.equal(value, actual[key]) for key, value in reference.items()}
        fields['state_fp32'] = all(cache.storage.layers[0].state.dtype == torch.float32
                                   for cache in caches.values())
        fields['request_finished'] = all(cache.current_valid is None and not cache._dense_prefill
                                         for cache in caches.values())
        fields['physical_length'] = all(cache.get_seq_length() == prefix.shape[1] + step
                                        for cache in caches.values())
        fields['seen_tokens_count'] = all(bool((cache.storage.seen_tokens == prefix.shape[1] + step).all())
                                          for cache in caches.values())
        output = difference(outputs['before'], outputs['after'])
        passed = output['equal'] and output['finite'] and all(fields.values()) and all(unchanged.values())
        rows.append(dict(step=step, output=output, cache=fields, old_tensors_unchanged=unchanged, passed=passed))
        if not passed:
            break
    return dict(steps=rows, passed=len(rows) == STEPS + 1 and all(row['passed'] for row in rows))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True, help='Before snapshot src/prefix_ttt directory')
    parser.add_argument('--weights', type=Path, default=Path('artifacts/mamba-kernel/weights/layer1.pt'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x'):
        pass
    sources = [Path(__file__), Path(__file__).with_name('mamba_bench.py')]
    result = dict(status='running', config={key: str(value) for key, value in vars(args).items()},
                  protocol=dict(cuda_graph=False, batch=1, lengths=LENGTHS, decode_steps=STEPS,
                                warmup=2, rounds=5, repetitions=5, seed=SEED, dtype='bfloat16',
                                hidden_size=DIM, heads=HEADS, head_dim=HEAD_DIM, depth=1,
                                allow_tf32=False, persistent_state_dtype='float32',
                                before='source snapshot immediately before cleanup; all optimizations retained',
                                weights='same checkpoint; identical feature parameters; shared QKV/O module objects',
                                prefill='fresh cache, complete layer including projections, RoPE, cache begin/finish',
                                decode='128 consecutive complete-layer steps; prefix outside timer; one event pair',
                                samples='alternating arms; ms/step is a 128-step request average',
                                scope='fixed all-valid B1 inputs; functional edge cases are separate unit tests'),
                  source_sha256={str(path): file_sha256(path) for path in sources},
                  before_sha256={str(path.relative_to(args.snapshot)): file_sha256(path)
                                 for path in sorted(args.snapshot.rglob('*.py'))},
                  after_sha256={str(path.relative_to('src/prefix_ttt')): file_sha256(path)
                                for path in sorted(Path('src/prefix_ttt').rglob('*.py'))}, measurements={})

    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(args.output)

    save()
    try:
        modules = frozen_modules(args.snapshot)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(SEED)
        result['environment'] = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                                     cuda=torch.version.cuda, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        result['weights'] = dict(path=str(args.weights), bytes=args.weights.stat().st_size,
                                 sha256=file_sha256(args.weights))
        layer = PairedLayer(modules, args.weights)
        timing = SimpleNamespace(warmup=2, rounds=5, repetitions=5)
        for length in LENGTHS:
            torch.manual_seed(SEED + length)
            prefix = torch.randn(1, length, DIM, device='cuda', dtype=torch.bfloat16)
            continuation = torch.randn(1, STEPS, DIM, device='cuda', dtype=torch.bfloat16)
            entry = result['measurements'][f'b1-t{length}'] = dict(status='correctness')
            entry['input_sha256'] = {name: hashlib.sha256(value.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                                     for name, value in (('prefix', prefix), ('continuation', continuation))}
            entry['correctness'] = correctness(layer, prefix, continuation)
            save()
            if not entry['correctness']['passed']:
                raise AssertionError(f'Output/cache equality failed at length {length}')
            calls = {arm: lambda _, arm=arm: layer.forward(arm, prefix) for arm in ARMS}
            entry['prefill'] = measure(calls, layer.use, lambda _: None, timing, length)
            save()
            calls = {arm: lambda cache, arm=arm: layer.decode(arm, continuation, cache) for arm in ARMS}
            entry['decode'] = measure(calls, layer.use, lambda arm: layer.forward(arm, prefix)[1],
                                      timing, STEPS, STEPS)
            entry['status'] = 'succeeded'
            save()
            print(json.dumps({'length': length, 'prefill': entry['prefill']['summary'],
                              'decode': entry['decode']['summary']}), flush=True)
        result['status'] = 'succeeded'
        save()
    except Exception as exc:
        result.update(status='failed', failure=dict(type=type(exc).__name__, message=str(exc)))
        save()
        raise


if __name__ == '__main__':
    main()
