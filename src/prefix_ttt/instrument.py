"""Per-sample inference cost instrumentation for benchmark runs.

Times every model forward with CUDA events, so the first call of a request is its
prefill and the remaining calls are decode steps. Peak memory is reset per request
and cache bytes are read from the hybrid cache afterwards, split into the full
attention KV, the TTT recurrent state and the Local-32 windows.
"""
import json
from pathlib import Path

import torch


def cache_buckets(cache):
    """Split cache bytes into full-attention KV, TTT state and Local-32 windows.

    E0 (base LLaVA) keeps a plain DynamicCache: every layer stores KV and there is
    no TTT state. The hybrid layout stores KV only in the anchor layers plus a
    state and a 32-token window everywhere else.
    """
    if cache is None:
        return {'full_kv_bytes': 0, 'ttt_state_bytes': 0, 'local_window_bytes': 0,
                'total_bytes': 0}
    if not hasattr(cache, 'storage'):
        tensors = list(getattr(cache, 'key_cache', [])) + list(getattr(cache, 'value_cache', []))
        total = sum(t.numel() * t.element_size() for t in tensors)
        return {'full_kv_bytes': total, 'ttt_state_bytes': 0, 'local_window_bytes': 0,
                'total_bytes': total}
    full = set(cache.storage.full_attention_layers)
    buckets = {'full_kv_bytes': 0, 'ttt_state_bytes': 0, 'local_window_bytes': 0}
    for index, layer in cache.storage.layers.items():
        for name, value in vars(layer).items():
            if value is None:
                continue
            size = value.numel() * value.element_size()
            if name == 'state':
                buckets['ttt_state_bytes'] += size
            elif index in full:
                buckets['full_kv_bytes'] += size
            else:
                buckets['local_window_bytes'] += size
    buckets['total_bytes'] = sum(buckets.values())
    return buckets


class ForwardMeter:
    """Record one timing per model forward between begin() and finish()."""

    def __init__(self, model):
        self.times = []
        self.spans = []
        self.cache = None
        self._start = None
        model.register_forward_pre_hook(self._before, with_kwargs=True)
        model.register_forward_hook(self._after, with_kwargs=True)

    def _before(self, module, args, kwargs):
        self._start = torch.cuda.Event(enable_timing=True)
        self._start.record()

    def _after(self, module, args, kwargs, output):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()
        self.times.append(self._start.elapsed_time(end))
        # Positions computed by this forward: equals the expanded prompt length for
        # the prefill, independent of whether the model received ids or embeddings.
        logits = getattr(output, 'logits', None)
        self.spans.append(int(logits.shape[1]) if logits is not None else None)
        self.cache = getattr(output, 'past_key_values', None)

    def begin(self):
        self.times = []
        self.spans = []
        self.cache = None

    def finish(self, extra=None):
        if not self.times:
            return None
        decode = self.times[1:]
        record = {
            'prefill_positions': self.spans[0] if self.spans else None,
            'generated_tokens': len(self.times),
            'prefill_ms': self.times[0],
            'tpot_ms': sum(decode) / len(decode) if decode else None,
            'decode_ms': decode,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'cache': cache_buckets(self.cache),
        }
        if extra:
            record.update(extra)
        return record


def append_record(path, record):
    """One JSON object per line; appended so every worker keeps its own file."""
    if record is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as handle:
        handle.write(json.dumps(record) + '\n')
