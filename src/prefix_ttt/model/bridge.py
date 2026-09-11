"""Strict, offline HF LLaVA checkpoint bridge into the original LLaVA package."""
import json
from pathlib import Path

import torch
from torch import nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel


# Expanded sequence cap of the pinned checkpoint: 576 image tokens plus text.
# The data pipeline enforces the same bound; the audit buckets are a separate
# decision (see manifests.LENGTH_BUCKET_BOUNDS).
MAX_EXPANDED_LENGTH = 2048

# Shape of the pinned production checkpoint. This is an identity assertion about one
# artifact, not a tunable: it must fail if a different model is supplied.
PINNED_LAYERS, PINNED_HIDDEN, PINNED_HEADS, PINNED_HEAD_DIM = 32, 4096, 32, 128
PINNED_SHAPE = (PINNED_LAYERS, PINNED_HIDDEN, PINNED_HEADS, PINNED_HEAD_DIM)
IMAGE_ASPECT_RATIO = 'square'


class EmbeddedCLIPVisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_tower = CLIPVisionModel(CLIPVisionConfig(**config.embedded_vision_config))
        self.select_layer = config.mm_vision_select_layer
        self.is_loaded = True
        self.image_processor = None
        self.requires_grad_(False)

    def forward(self, images):
        outputs = self.vision_tower(images.to(dtype=self.dtype), output_hidden_states=True)
        return outputs.hidden_states[self.select_layer][:, 1:]

    def load_model(self, device_map=None):
        return None

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2

    @property
    def hidden_size(self):
        return self.config.hidden_size


def bridge_config(hf_config, *, validate_production=True):
    from llava.model.language_model.llava_llama import LlavaConfig
    config = LlavaConfig(**hf_config['text_config'])
    if validate_production:
        actual = (config.num_hidden_layers, config.hidden_size,
                  config.num_attention_heads, config.hidden_size // config.num_attention_heads)
        if actual != PINNED_SHAPE:
            raise ValueError(f'Checkpoint architecture mismatch: {actual}')
        if config.vocab_size != 32064:
            raise ValueError('Expected all 32064 embedding/lm_head rows')
    config.embedded_vision_config = hf_config['vision_config']
    config.mm_vision_tower = 'embedded-checkpoint'
    config.mm_hidden_size = hf_config['vision_config']['hidden_size']
    config.mm_projector_type = 'mlp2x_gelu'
    config.mm_vision_select_layer = hf_config.get('vision_feature_layer', -2)
    if hf_config.get('vision_feature_select_strategy', 'default') != 'default':
        raise ValueError('Only default CLIP patch selection is supported')
    if hf_config.get('projector_hidden_act', 'gelu') != 'gelu':
        raise ValueError('Projector activation mismatch')
    config.mm_patch_merge_type = 'flat'
    config.image_aspect_ratio = IMAGE_ASPECT_RATIO
    config.tokenizer_model_max_length = MAX_EXPANDED_LENGTH
    config.tokenizer_padding_side = 'right'
    config.tie_word_embeddings = False
    config.pad_token_id = hf_config.get('pad_token_id')
    return config


def map_checkpoint_key(key):
    for source, target in (
        ('language_model.model.', 'model.'),
        ('language_model.lm_head.', 'lm_head.'),
        ('vision_tower.', 'model.vision_tower.vision_tower.'),
        ('multi_modal_projector.linear_1.', 'model.mm_projector.0.'),
        ('multi_modal_projector.linear_2.', 'model.mm_projector.2.'),
    ):
        if key.startswith(source):
            return target + key[len(source):]
    raise ValueError(f'Unmapped checkpoint tensor: {key}')


def strict_assign_shards(model, shard_paths):
    """Assign one safetensors shard at a time, checking global exact coverage."""
    from safetensors.torch import load_file
    expected = model.state_dict()
    seen = set()
    for path in shard_paths:
        mapped = {}
        for key, tensor in load_file(str(path), device='cpu').items():
            target = map_checkpoint_key(key)
            if target in seen or target not in expected:
                raise ValueError(f'Duplicate/unexpected tensor: {target}')
            if tensor.shape != expected[target].shape:
                raise ValueError(f'Shape mismatch: {target}: {tensor.shape} != {expected[target].shape}')
            mapped[target] = tensor
            seen.add(target)
        model.load_state_dict(mapped, strict=False, assign=True)
    missing = set(expected) - seen
    if missing:
        raise ValueError(f'Missing checkpoint tensors: {sorted(missing)}')
    return {'tensor_count': len(seen), 'missing': [], 'unexpected': []}


def load_checkpoint(path, *, dtype=torch.bfloat16, validate_production=True):
    from accelerate import init_empty_weights
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    path = Path(path)
    config = bridge_config(json.loads((path / 'config.json').read_text()),
                           validate_production=validate_production)
    with init_empty_weights():
        model = LlavaLlamaForCausalLM(config)
    index = path / 'model.safetensors.index.json'
    if index.exists():
        names = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    else:
        names = ['model.safetensors']
    report = strict_assign_shards(model, [path / name for name in names])
    # Match HF from_pretrained(torch_dtype=...): checkpoint parameters use the
    # requested dtype, but freshly initialized nonpersistent RoPE buffers keep
    # their original FP32 values. Casting the whole module first would lose
    # frequencies irreversibly, even if inv_freq were subsequently .float().
    for parameter in model.parameters():
        parameter.data = parameter.data.to(dtype=dtype)
    model.requires_grad_(False)
    model.eval()
    model.get_vision_tower().image_processor = CLIPImageProcessor.from_pretrained(path, local_files_only=True)
    return model, report


def load_tokenizer(path):
    """Keep local tokenizer semantics and all model rows; never resize embeddings."""
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(path, local_files_only=True,
                                               model_max_length=MAX_EXPANDED_LENGTH, padding_side='right')
    if tokenizer.pad_token_id is None:
        raise ValueError('Checkpoint tokenizer must define its padding token')
    return tokenizer

