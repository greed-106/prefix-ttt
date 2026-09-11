"""LLaVA attention replacement with stateless training and private inference cache."""
import torch
from torch import nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from prefix_ttt.ops.features import FeatureReadout
from prefix_ttt.ops import TILE_SIZE
from prefix_ttt.ops.local import (local_attention, local_attention_cached,
                                 local_attention_decode, LocalCache)
from prefix_ttt.ops.reference import chunk_prefix
from prefix_ttt.ops.fla import fla_prefix, recurrent_step
from prefix_ttt.cache import LayerState
from prefix_ttt.runtime import SEED
from prefix_ttt.model.generation import TransformersHybridCache


FULL_ATTENTION_LAYERS = (0, 3, 7, 11, 15, 19, 23, 27, 31)


class PrefixTTTAttention(nn.Module):
    def __init__(self, original, *, backend, seed=SEED, tile_size=TILE_SIZE):
        super().__init__()
        if backend not in ('reference', 'fla'):
            raise ValueError('Select reference or fla explicitly')
        config = original.config
        if config.num_attention_heads != config.num_key_value_heads:
            raise ValueError('Prefix-TTT v1 requires shared-size Q/K/V heads, not GQA')
        self.config, self.layer_idx = config, original.layer_idx
        self.head_dim = original.head_dim
        self.heads = config.num_attention_heads
        self.backend, self.tile_size = backend, tile_size
        # Preserve exact module objects/names: PEFT whitelist remains valid.
        self.q_proj, self.k_proj = original.q_proj, original.k_proj
        self.v_proj, self.o_proj = original.v_proj, original.o_proj
        self.prefix_ttt = FeatureReadout(config.hidden_size, self.heads, self.head_dim,
                                        seed=seed + self.layer_idx)
        self.prefix_ttt.to(device=self.q_proj.weight.device, dtype=torch.float32)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_value=None, cache_position=None, *, prefix_valid_mask=None,
                use_cache=False, output_attentions=False, **kwargs):
        caching = use_cache or past_key_value is not None
        if caching and not isinstance(past_key_value, TransformersHybridCache):
            raise ValueError('Hybrid forward requires TransformersHybridCache')
        if caching and torch.is_grad_enabled():
            raise RuntimeError('Mutable inference cache requires no_grad; training uses use_cache=False')
        if output_attentions:
            raise NotImplementedError('Hybrid branch attention maps are not a Full Attention matrix')
        b, t, _ = hidden_states.shape
        valid = prefix_valid_mask
        if valid is None:
            if attention_mask is not None:
                raise ValueError('Pass prefix_valid_mask from multimodal metadata; do not infer padding from an expanded causal mask')
            valid = torch.ones((b, t), device=hidden_states.device, dtype=torch.bool)
        if valid.shape != (b, t) or valid.dtype != torch.bool:
            raise ValueError('prefix_valid_mask must be boolean [B,T]')
        shape = (b, t, self.heads, self.head_dim)
        q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        old = past_key_value.storage.layers.get(self.layer_idx) if caching else None
        initial_state = None if old is None else old.state
        if caching:
            valid = past_key_value.current_valid
            local_cache = None if old is None else LocalCache(old.key, old.value,
                old.local_position, past_key_value.storage.seen_tokens)
            local, final_local = (local_attention_decode(q, k, v, valid, local_cache)
                                  if t == 1 and local_cache is not None
                                  else local_attention_cached(q, k, v, valid, local_cache))
        else:
            local = local_attention(q, k, v, valid)
        # New parameters remain FP32 masters. Inference cannot rely on the
        # caller installing autocast around a BF16 base model's projections.
        amp_dtype = q.dtype if q.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        amp_enabled = q.dtype in (torch.bfloat16, torch.float16)
        with torch.autocast(device_type=q.device.type, dtype=amp_dtype, enabled=amp_enabled):
            qf, kf = self.prefix_ttt.features(q), self.prefix_ttt.features(k)
        if caching and t == 1:
            memory, state = recurrent_step(qf, kf, v, initial_state=initial_state, valid=valid)
        elif self.backend == 'reference':
            memory, state = chunk_prefix(qf, kf, v, initial_state=initial_state,
                                         valid=valid, tile_size=self.tile_size)
        else:
            memory, state = fla_prefix(qf, kf, v, initial_state=initial_state,
                valid=valid, tile_size=self.tile_size, need_final_state=caching)
        if caching:
            past_key_value.storage.set_layer(self.layer_idx, LayerState(state=state,
                key=final_local.key, value=final_local.value, local_position=final_local.lengths))
        with torch.autocast(device_type=q.device.type, dtype=amp_dtype, enabled=amp_enabled):
            residual = self.prefix_ttt.readout(hidden_states, memory)
        combined = (local + residual).masked_fill(~valid[..., None, None], 0)
        output = self.o_proj(combined.reshape(b, t, -1))
        return output.masked_fill(~valid[..., None], 0), None


def install_prefix_ttt(model, *, backend, full_attention_layers=FULL_ATTENTION_LAYERS,
                       seed=SEED, tile_size=TILE_SIZE):
    layers = model.get_model().layers
    anchors = set(full_attention_layers)
    if any(i < 0 or i >= len(layers) for i in anchors):
        raise ValueError('Anchor index outside model')
    if any(isinstance(layer.self_attn, PrefixTTTAttention) for layer in layers):
        raise ValueError('Prefix-TTT is already installed')
    model.requires_grad_(False)
    new_parameters = []
    for index, layer in enumerate(layers):
        if index not in anchors:
            layer.self_attn = PrefixTTTAttention(layer.self_attn, backend=backend,
                                                seed=seed, tile_size=tile_size)
            new_parameters.extend(layer.self_attn.prefix_ttt.parameters())
    model.config.use_cache = False
    model.config.prefix_ttt_layers = [i for i in range(len(layers)) if i not in anchors]
    model.config.prefix_ttt_backend = backend
    return new_parameters
