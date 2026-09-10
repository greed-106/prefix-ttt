"""Request-private storage foundations, not yet a Transformers Cache adapter."""

from dataclasses import dataclass
import torch


@dataclass
class LayerState:
    state: torch.Tensor | None = None
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    local_position: torch.Tensor | None = None

    def select(self, indices):
        return LayerState(**{name: None if value is None else value.index_select(0, indices)
                            for name, value in vars(self).items()})


class HybridCache:
    """Counts advance once per model call, never once per layer.

    Full/Local KV tensors use [B,T,H,D]. Layer writes are owned by the future
    attention adapter; this class enforces FP32 state and local capacity.
    """
    def __init__(self, batch_size, full_attention_layers, device="cpu"):
        self.full_attention_layers = frozenset(full_attention_layers)
        self.seen_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.layers = {}

    @property
    def next_position(self):
        return self.seen_tokens.clone()

    def active_mask(self, valid):
        if valid.ndim != 2 or valid.shape[0] != len(self.seen_tokens) or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean [B,T]")
        return valid & ~self.finished[:, None]

    def advance(self, valid):
        self.seen_tokens += self.active_mask(valid).sum(1)

    def mark_finished(self, finished):
        if finished.shape != self.finished.shape or finished.dtype != torch.bool:
            raise ValueError("finished must be boolean [B]")
        self.finished |= finished

    def set_layer(self, index, layer):
        if layer.state is not None:
            if layer.state.dtype != torch.float32 or layer.state.shape[0] != len(self.seen_tokens):
                raise ValueError("state must be request-batched FP32")
            if index in self.full_attention_layers:
                raise ValueError("MSA layer cannot own a TTT state")
        if (layer.key is None) != (layer.value is None):
            raise ValueError("key/value must both be present")
        if layer.key is not None:
            if layer.key.shape != layer.value.shape or layer.key.shape[0] != len(self.seen_tokens):
                raise ValueError("KV shape mismatch")
            if index not in self.full_attention_layers and layer.key.shape[1] > 32:
                raise ValueError("Local KV capacity exceeds 32")
        previous = self.layers.get(index)
        protected = {}
        for name, value in vars(layer).items():
            old = None if previous is None else getattr(previous, name)
            if value is None:
                if old is not None:
                    raise ValueError("cannot discard an existing cache field")
                protected[name] = None
                continue
            if value.shape[0] != len(self.seen_tokens):
                raise ValueError("every cache field must be request-batched")
            if old is None:
                old = torch.zeros_like(value)
            elif old.shape != value.shape:
                # Full KV may grow its physical buffer while finished rows
                # preserve their old valid prefix and get only zero padding.
                if name not in ("key", "value") or old.shape[2:] != value.shape[2:] or old.shape[1] > value.shape[1]:
                    raise ValueError("cache field cannot shrink or change non-time dimensions")
                old = torch.cat((old, old.new_zeros(old.shape[0], value.shape[1] - old.shape[1], *old.shape[2:])), dim=1)
            mask = self.finished.reshape((-1,) + (1,) * (value.ndim - 1))
            protected[name] = torch.where(mask, old, value)
        layer = LayerState(**protected)
        self.layers[index] = layer

    def select(self, indices):
        result = HybridCache(len(indices), self.full_attention_layers, self.seen_tokens.device)
        result.seen_tokens = self.seen_tokens.index_select(0, indices)
        result.finished = self.finished.index_select(0, indices)
        result.layers = {i: layer.select(indices) for i, layer in self.layers.items()}
        return result

    @property
    def tensor_bytes(self):
        tensors = [self.seen_tokens, self.finished]
        tensors += [x for layer in self.layers.values() for x in vars(layer).values() if x is not None]
        return sum(x.numel() * x.element_size() for x in tensors)
