"""Checkpoint-loading adapter; all benchmark protocol remains in lmms-eval."""
from pathlib import Path

import torch
from accelerate import Accelerator

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models import MODEL_REGISTRY_V2
from lmms_eval.models.registry_v2 import ModelManifest
from lmms_eval.models.simple.llava import Llava

from prefix_ttt.model.bridge import load_checkpoint, load_tokenizer
from prefix_ttt.model.hybrid import install_prefix_ttt
from prefix_ttt.model.trainability import install_lora
from prefix_ttt.sft import load_trainable


@register_model('prefix_ttt_llava')
class PrefixTTTLlava(Llava):
    """Use official Llava.generate_until with our original-LLaVA model bridge.

    E0 needs only ``pretrained``. E1/E2 additionally pass the project's phase-B
    ``checkpoint``; the complete trainable tensors include both LoRA and TTT.
    No benchmark task, prompt, answer parser or score is implemented here.
    """

    def __init__(self, pretrained, checkpoint=None, batch_size=1,
                 device='cuda:0', conv_template='vicuna_v1', **kwargs):
        lmms.__init__(self)
        if kwargs:
            raise ValueError(f'Unsupported model options: {sorted(kwargs)}')
        if int(batch_size) != 1:
            raise ValueError('The fixed benchmark protocol uses batch_size=1')
        if conv_template != 'vicuna_v1':
            raise ValueError('The fixed benchmark protocol uses vicuna_v1')
        if not Path(pretrained).is_dir():
            raise ValueError('pretrained must be the local complete HF checkpoint')
        self.accelerator = Accelerator()
        self._rank = self.accelerator.process_index
        self._world_size = self.accelerator.num_processes
        self._device = (self.accelerator.device if self._world_size > 1
                        else torch.device(device))
        self.device_map = str(self._device)
        model, self.bridge_report = load_checkpoint(pretrained, dtype=torch.bfloat16)
        self._tokenizer = load_tokenizer(pretrained)
        self._image_processor = model.get_vision_tower().image_processor
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if (state.get('stage') != 'B' or state.get('layout') not in ('E1', 'E2')
                    or state.get('diagnostic_only', False)):
                raise ValueError('Expected a non-debug E1/E2 phase-B checkpoint')
            new_parameters = (install_prefix_ttt(model, backend='fla')
                              if state['layout'] == 'E2' else [])
            model = install_lora(model, new_parameters=new_parameters)
            load_trainable(model, state['trainable'])
            # Keep installed LoRA modules unmerged for every trained layout,
            # without PEFT's generate wrapper injecting extra cache arguments.
            model = model.get_base_model()
            del state
        # Never cast the whole module: preserve FP32 RoPE and new parameters.
        self._model = model.to(self._device).eval().requires_grad_(False)
        self._config = self._model.config
        self._config.use_cache = True
        self._max_length = self._config.tokenizer_model_max_length
        self.batch_size_per_gpu = 1
        self.conv_template = conv_template
        self.use_cache = True
        self.truncation = True
        self.truncate_context = False


# The pinned lmms-eval 0.7.2 CLI resolves declarative model manifests.
MODEL_REGISTRY_V2.register_manifest(ModelManifest(
    model_id='prefix_ttt_llava',
    simple_class_path='prefix_ttt.lmms_model.PrefixTTTLlava',
))
