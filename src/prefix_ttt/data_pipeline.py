"""Fixed manifest selection around the original LLaVA dataset and expansion."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from prefix_ttt.manifests import digest_file, digest_json

MANIFEST_SHA256 = '246586be1ddd3d0cea09e17047e6d658c5e9a38637654e46906605578707a93b'


def load_manifest(path, expected_sha256=MANIFEST_SHA256):
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != expected_sha256:
        raise ValueError('Fixed manifest SHA256 mismatch')
    manifest = json.loads(raw)
    for split in ('train', 'dev', 'A'):
        indices = manifest[split]
        if len(indices) != len(set(indices)) or digest_json(indices) != manifest['order_sha256'][split]:
            raise ValueError(f'Invalid fixed {split} order')
    if set(manifest['dev']) & set(manifest['train']) or not set(manifest['A']) <= set(manifest['train']):
        raise ValueError('Invalid fixed split separation')
    annotation = str(Path(manifest['annotation']).resolve())
    if digest_file(annotation) != manifest['inputs'][annotation]:
        raise ValueError('Original annotation changed after audit')
    return manifest, sha


def build_dataset(config, model, tokenizer, manifest):
    from llava import conversation
    from llava.train.train import LazySupervisedDataset, DataCollatorForSupervisedDataset
    conversation.default_conversation = conversation.conv_templates['v1']
    tokenizer.padding_side = 'right'
    args = SimpleNamespace(is_multimodal=True, mm_use_im_start_end=False,
        image_aspect_ratio=model.config.image_aspect_ratio,
        image_folder=str(Path(config['data_root']) / 'datasets/llava-665k/images'),
        image_processor=model.get_vision_tower().image_processor)
    dataset = LazySupervisedDataset(manifest['annotation'], tokenizer, args)
    return dataset, DataCollatorForSupervisedDataset(tokenizer)


def prepare_sample(base, dataset, collate, index, device):
    batch = {key: value.to(device) for key, value in collate([dataset[index]]).items()}
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        prepared, metadata = base.prepare_inputs_labels_for_multimodal(
            batch['input_ids'], None, batch['attention_mask'], None,
            batch['labels'], batch['images'], return_metadata=True)
    ids, positions, mask, _, embeds, labels = prepared
    if labels[:, 1:].ne(-100).sum() == 0:
        raise ValueError(f'Preprocessing lost supervision for audited index {index}; do not discard')
    if metadata['valid_mask'].sum() > 2048:
        raise ValueError('Expanded sequence exceeds fixed limit')
    values = dict(input_ids=ids, position_ids=positions, attention_mask=mask,
                  inputs_embeds=embeds, labels=labels, prefix_valid_mask=metadata['valid_mask'])
    return ({key: value.cpu() for key, value in values.items() if value is not None},
            {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in metadata.items()})
