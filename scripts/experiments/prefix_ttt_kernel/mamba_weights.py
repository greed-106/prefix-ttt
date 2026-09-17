"""Export E2 attention layer 1 for the fixed-input benchmark, entirely on CPU.

Read only selected safetensors tensors and mmap the training checkpoint. The
seven output keys are a PrefixTTTAttention state_dict; provenance is a JSON
sidecar. Existing outputs are never overwritten.
"""

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
import torch

from prefix_ttt.model.bridge import bridge_config
from prefix_ttt.model.hybrid import FULL_ATTENTION_LAYERS
from prefix_ttt.model.trainability import LORA_ALPHA, LORA_RANK


LAYER = 1
DIM, HEADS, HEAD_DIM, MLP_DIM = 4096, 32, 128, 11008
PROJECTIONS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
FEATURE_SHAPES = {'a_phi': (HEADS, HEAD_DIM, HEAD_DIM),
                  'b_phi': (HEADS, HEAD_DIM, HEAD_DIM),
                  'gate_weight': (HEADS, DIM)}


def validate_checkpoint(state):
    if (state.get('stage') != 'B' or state.get('layout') != 'E2'
            or state.get('complete') is not True or state.get('diagnostic_only', False)
            or state.get('trainable_mode') not in (None, 'lora')
            or state.get('global_step', 0) <= 0
            or state.get('global_step') != state.get('total_steps')):
        raise ValueError('Expected a completed, non-debug E2 phase-B LoRA checkpoint')
    # Audit all trainable names/shapes without touching their mmap storage.
    expected = {}
    dimensions = {f'self_attn.{name}': (DIM, DIM) for name in PROJECTIONS}
    dimensions.update({'mlp.gate_proj': (MLP_DIM, DIM),
                       'mlp.up_proj': (MLP_DIM, DIM), 'mlp.down_proj': (DIM, MLP_DIM)})
    for layer in range(32):
        root = f'base_model.model.model.layers.{layer}.'
        for name, (output_dim, input_dim) in dimensions.items():
            expected[root + name + '.lora_A.default.weight'] = (LORA_RANK, input_dim)
            expected[root + name + '.lora_B.default.weight'] = (output_dim, LORA_RANK)
        if layer not in FULL_ATTENTION_LAYERS:
            for name, shape in FEATURE_SHAPES.items():
                expected[root + 'self_attn.prefix_ttt.' + name] = shape
    tensors = state['trainable']
    if tensors.keys() != expected.keys():
        raise ValueError('Checkpoint does not contain the complete E2 LoRA/TTT parameter layout')
    for name, shape in expected.items():
        tensor = tensors[name]
        if tensor.shape != shape or tensor.dtype != torch.float32:
            raise ValueError(f'Checkpoint shape/dtype mismatch: {name}')


def tensor_record(value):
    raw = value.detach().contiguous().view(torch.uint8).numpy()
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(raw).hexdigest()}


def file_record(path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def export(pretrained, checkpoint, output):
    sidecar = output.with_suffix(output.suffix + '.json')
    if output.exists() or sidecar.exists():
        raise FileExistsError('Output or provenance already exists; choose a new --output')
    config = bridge_config(json.loads((pretrained / 'config.json').read_text()))
    if config.intermediate_size != MLP_DIM or config.num_key_value_heads != HEADS:
        raise ValueError('Expected the pinned 7B model without GQA')
    index_path = pretrained / 'model.safetensors.index.json'
    weight_map = json.loads(index_path.read_text())['weight_map']
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    validate_checkpoint(state)
    trained = state['trainable']
    root = f'base_model.model.model.layers.{LAYER}.self_attn.'
    tensors, sources = {}, {}
    shards = {}
    for projection in PROJECTIONS:
        key = f'language_model.model.layers.{LAYER}.self_attn.{projection}.weight'
        shard = pretrained / weight_map[key]
        with safe_open(shard, framework='pt', device='cpu') as source:
            original = source.get_tensor(key)
            if original.shape != (DIM, DIM) or original.dtype != torch.float16:
                raise ValueError(f'Unexpected pretrained projection shape/dtype: {key}')
            sources[key] = tensor_record(original)
            base = original.to(torch.bfloat16)
        shards[str(shard.resolve())] = file_record(shard)
        a_key = root + projection + '.lora_A.default.weight'
        b_key = root + projection + '.lora_B.default.weight'
        a, b = trained[a_key], trained[b_key]
        sources[a_key], sources[b_key] = tensor_record(a), tensor_record(b)
        # Identical operation order to model.trainability.merge_lora_weights:
        # FP32 B@A; FP32 alpha/r; cast delta to BF16; add into BF16 base.
        delta = b.float() @ a.float()
        base += (delta * (LORA_ALPHA / LORA_RANK)).to(base.dtype)
        tensors[projection + '.weight'] = base
        del delta
    for name in FEATURE_SHAPES:
        key = root + 'prefix_ttt.' + name
        value = trained[key]
        sources[key] = tensor_record(value)
        tensors['prefix_ttt.' + name] = value.clone()
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError('Export contains non-finite values')
    if not torch.count_nonzero(tensors['prefix_ttt.gate_weight']):
        raise ValueError('Trained gate is entirely zero; refusing a degenerate benchmark')
    metadata_keys = ('stage', 'layout', 'trainable_mode', 'complete', 'diagnostic_only',
                     'global_step', 'total_steps', 'samples_seen', 'manifest_sha256',
                     'config_sha256', 'stage_a_sha256')
    provenance = {
        'layer': LAYER, 'pretrained': str(pretrained.resolve()),
        'checkpoint': file_record(checkpoint),
        'checkpoint_metadata': {key: state[key] for key in metadata_keys if key in state},
        'checkpoint_trainable_tensor_count': len(trained),
        'pretrained_shards': list(shards.values()),
        'pretrained_index_sha256': file_sha256(index_path),
        'lora': {'rank': LORA_RANK, 'alpha': LORA_ALPHA, 'scale': LORA_ALPHA / LORA_RANK,
                 'merge': 'BF16(base) += BF16((FP32(B) @ FP32(A)) * alpha/r)'},
        'source_tensors': sources,
        'tensor_hash_format': 'SHA256 of contiguous native-endian tensor storage bytes',
        'output_tensors': {name: tensor_record(value) for name, value in tensors.items()},
        'torch': torch.__version__, 'export_script_sha256': file_sha256(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        torch.save(tensors, stream)
    provenance['output'] = {**file_record(output), 'sha256': file_sha256(output)}
    with sidecar.open('x') as stream:
        json.dump(provenance, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(output.resolve()), 'provenance': str(sidecar.resolve()),
                      'keys': list(tensors), 'layer': LAYER}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    export(args.pretrained, args.checkpoint, args.output)


if __name__ == '__main__':
    main()
