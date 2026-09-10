"""Audit the production-size architecture on meta, without GPU/weight loads."""
import json
from pathlib import Path

import torch

from accelerate import init_empty_weights
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

from prefix_ttt.model.bridge import bridge_config
from prefix_ttt.model.hybrid import install_prefix_ttt
from prefix_ttt.model.trainability import install_lora, audit_parameters


def main():
    directory = Path('data/llava-v1.5-assets-v1/models/llava-1.5-7b-hf-b234b804b114d9e37bb655e11cbbb5f5e971b7a9')
    config = bridge_config(json.loads((directory / 'config.json').read_text()))
    with init_empty_weights():
        model = LlavaLlamaForCausalLM(config)
        for parameter in model.parameters():
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
        new = install_prefix_ttt(model, backend='reference')
        model = install_lora(model, new_parameters=new)
    records = audit_parameters(model, new_parameters=new)
    totals = {kind: sum(r['numel'] for r in records if r['optimizer_group'] == kind)
              for kind in ('new_module', 'lora')}
    assert totals == {'new_module': 27131904, 'lora': 79953920}, totals
    result = {'execution': 'production config on meta; no weights loaded or model execution',
              'totals': totals, 'trainable_total': sum(totals.values()),
              'parameters': records}
    output = Path('docs/experiments/2026-09-09-prefix-ttt/trainable_params.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'totals': totals}))


if __name__ == '__main__':
    main()
