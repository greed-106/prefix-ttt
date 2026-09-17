"""Transformers cache adapter with separate physical and effective positions."""
import torch
from transformers.cache_utils import Cache

from prefix_ttt.cache import HybridCache, LayerState


class TransformersHybridCache(Cache):
    def __init__(self, batch_size, full_attention_layers, device='cpu'):
        super().__init__()
        self.storage = HybridCache(batch_size, full_attention_layers, device)
        self.valid_history = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        self.current_valid = None
        self._dense_prefill = False

    def get_seq_length(self, layer_idx=0):
        return self.valid_history.shape[1]

    def get_max_cache_shape(self):
        return None

    def begin(self, valid):
        if self.current_valid is not None:
            raise RuntimeError('Cache forward already in progress; discard cache after a failed forward')
        self.current_valid = self.storage.active_mask(valid)
        # One request-level check replaces valid-token packing in every TTT
        # layer. Never assume that an explicitly supplied mask has no holes.
        self._dense_prefill = (valid.is_cuda and not torch.is_grad_enabled()
            and self.valid_history.shape[1] == 0 and valid.shape[1] > 1
            and bool(self.current_valid.all().item()))
        positions = self.storage.seen_tokens[:, None] + self.current_valid.long().cumsum(1) - 1
        positions = positions.clamp_min(0).masked_fill(~self.current_valid, 0)
        return torch.cat((self.valid_history, self.current_valid), 1), positions

    def finish(self):
        if self.current_valid is None:
            raise RuntimeError('Cache forward not begun')
        self.storage.advance(self.current_valid)
        self.valid_history = torch.cat((self.valid_history, self.current_valid), 1)
        self.current_valid = None
        self._dense_prefill = False

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx not in self.storage.full_attention_layers or self.current_valid is None:
            raise ValueError('Invalid MSA cache update')
        key, value = key_states.transpose(1, 2), value_states.transpose(1, 2)
        mask = self.current_valid[..., None, None]
        key, value = key.masked_fill(~mask, 0), value.masked_fill(~mask, 0)
        old = self.storage.layers.get(layer_idx)
        if old is not None:
            key, value = torch.cat((old.key, key), 1), torch.cat((old.value, value), 1)
        self.storage.set_layer(layer_idx, LayerState(key=key, value=value))
        layer = self.storage.layers[layer_idx]
        return layer.key.transpose(1, 2), layer.value.transpose(1, 2)

    def reorder_cache(self, beam_idx):
        beam_idx = beam_idx.to(device=self.valid_history.device, dtype=torch.long)
        self.storage = self.storage.select(beam_idx)
        self.valid_history = self.valid_history.index_select(0, beam_idx)
        if self.current_valid is not None:
            self.current_valid = self.current_valid.index_select(0, beam_idx)

    def batch_select_indices(self, indices):
        self.reorder_cache(indices)


# The hybrid path supports cached greedy single-beam decoding only.
MAX_NEW_TOKENS, NUM_BEAMS, DO_SAMPLE = 128, 1, False


def new_cache(model, batch_size, device):
    ttt = set(model.config.prefix_ttt_layers)
    return TransformersHybridCache(batch_size,
        [i for i in range(model.config.num_hidden_layers) if i not in ttt], device)


@torch.no_grad()
def greedy_generate(model, inputs, *, images=None, image_sizes=None,
                    attention_mask=None, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS,
                    do_sample=DO_SAMPLE, use_cache=True, eos_token_id=None,
                    pad_token_id=None, temperature=None, top_p=None, **kwargs):
    """Greedy continuation IDs; prompt is expanded once, subsequent qlen is one."""
    if model.training:
        raise ValueError('Call model.eval() before hybrid generation')
    if num_beams != 1 or do_sample or not use_cache or kwargs:
        raise NotImplementedError('Hybrid generation supports cached greedy num_beams=1 only; unsupported options: ' + ','.join(kwargs))
    if max_new_tokens < 0:
        raise ValueError('max_new_tokens must be nonnegative')
    if inputs is None or inputs.ndim != 2:
        raise ValueError('inputs must be [B,T] token IDs')
    if max_new_tokens == 0:
        return inputs.new_empty(inputs.shape[0], 0)
    if eos_token_id is None:
        eos_token_id = model.generation_config.eos_token_id
    eos = [] if eos_token_id is None else ([eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id))
    pad = pad_token_id if pad_token_id is not None else model.generation_config.pad_token_id
    if pad is None:
        pad = eos[0] if eos else 0
    output = model(input_ids=inputs, images=images, image_sizes=image_sizes,
                   attention_mask=attention_mask, use_cache=True)
    cache = output.past_key_values
    valid = cache.valid_history
    if not valid.any(1).all():
        raise ValueError('Every request requires at least one valid prompt token')
    if (cache.storage.seen_tokens + max_new_tokens > model.config.max_position_embeddings).any():
        raise ValueError('Requested generation exceeds checkpoint context capacity')
    positions = torch.arange(valid.shape[1], device=valid.device)[None].expand_as(valid)
    last = positions.masked_fill(~valid, -1).max(1).values
    logits = output.logits[torch.arange(inputs.shape[0], device=inputs.device), last]
    continuation = []
    for step in range(max_new_tokens):
        token = logits.argmax(-1).masked_fill(cache.storage.finished, pad)
        continuation.append(token)
        finished = torch.zeros_like(cache.storage.finished)
        for end_token in eos:
            finished |= token == end_token
        cache.storage.mark_finished(finished)
        if step + 1 == max_new_tokens or cache.storage.finished.all():
            break
        output = model(input_ids=token[:, None], past_key_values=cache, use_cache=True)
        logits = output.logits[:, -1]
    return torch.stack(continuation, 1)
