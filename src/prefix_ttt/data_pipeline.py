"""Fixed manifest selection around the original LLaVA dataset and expansion."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from prefix_ttt.digests import digest_file, digest_json
from prefix_ttt.model.bridge import MAX_EXPANDED_LENGTH
from prefix_ttt.model.labels import IGNORE_INDEX
from prefix_ttt.training import EFFECTIVE_BATCH_SIZE


CONV_TEMPLATE = 'v1'   # the template the pinned checkpoint was trained with

def local_path(recorded, data_root):
    """A manifest records source-machine paths; locate the same file under the local data root."""
    path = Path(recorded)
    if path.is_file():
        return path
    root = Path(data_root).parts
    for index in range(len(path.parts) - len(root) + 1):
        if path.parts[index:index + len(root)] == root:
            return Path(data_root).joinpath(*path.parts[index + len(root):])
    raise FileNotFoundError(f'Recorded source path not found under {data_root}: {recorded}')


def load_manifest(path, data_root):
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    for split in ('train', 'dev', 'A'):
        indices = manifest[split]
        if len(indices) != len(set(indices)) or digest_json(indices) != manifest['order_sha256'][split]:
            raise ValueError(f'Invalid fixed {split} order')
    if set(manifest['dev']) & set(manifest['train']) or not set(manifest['A']) <= set(manifest['train']):
        raise ValueError('Invalid fixed split separation')
    recorded = manifest['annotation']
    manifest['annotation'] = str(local_path(recorded, data_root))
    if digest_file(manifest['annotation']) != manifest['inputs'][recorded]:
        raise ValueError('Original annotation changed after audit')
    return manifest, sha


def build_dataset(config, model, tokenizer, manifest):
    from llava import conversation
    from llava.train.train import LazySupervisedDataset, DataCollatorForSupervisedDataset
    conversation.default_conversation = conversation.conv_templates[CONV_TEMPLATE]
    tokenizer.padding_side = 'right'
    args = SimpleNamespace(is_multimodal=True, mm_use_im_start_end=False,
        image_aspect_ratio=model.config.image_aspect_ratio,
        image_folder=str(Path(config['data_root']) / 'datasets/llava-665k/images'),
        image_processor=model.get_vision_tower().image_processor)
    dataset = LazySupervisedDataset(manifest['annotation'], tokenizer, args)
    return dataset, DataCollatorForSupervisedDataset(tokenizer)


def micro_batches(order, cursor, rank, world_size, micro_batch_size,
                  effective_batch_size=EFFECTIVE_BATCH_SIZE):
    """This rank's micro-batches for one fixed group, in manifest order."""
    group = order[cursor:cursor + effective_batch_size][rank::world_size]
    return [group[start:start + micro_batch_size] for start in range(0, len(group), micro_batch_size)]


def batch_loader(dataset, collate, index_batches, workers):
    """Collate micro-batches on CPU workers so loading overlaps GPU compute."""
    class IndexBatches(torch.utils.data.Dataset):
        def __len__(self):
            return len(index_batches)

        def __getitem__(self, position):
            return [dataset[index] for index in index_batches[position]]

    return torch.utils.data.DataLoader(IndexBatches(), batch_size=None, collate_fn=collate,
        num_workers=workers, prefetch_factor=2 if workers else None)


def prepare_sample(base, batch, device, trainable_embedding=False):
    """Expand one sample and move it to the device.

    ``trainable_embedding`` is set by full fine-tuning, whose whitelist contains the
    multimodal projector and the token embedding. Both are built in here, so the
    fixed ``no_grad`` would silently deny them every gradient. Dropping it does not
    retain vision activations: the vision tower's parameters are frozen and its
    input never requires grad, so autograd records nothing through it.
    """
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.set_grad_enabled(trainable_embedding), torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        prepared, metadata = base.prepare_inputs_labels_for_multimodal(
            batch['input_ids'], None, batch['attention_mask'], None,
            batch['labels'], batch['images'], return_metadata=True)
    ids, positions, mask, _, embeds, labels = prepared
    if trainable_embedding:
        # The residual stream must stay BF16. Llama's rotary embedding adopts the
        # hidden dtype, so an FP32 stream would hand FP32 q/k to the Prefix-TTT
        # features while v is BF16, which the FLA kernel rejects. With FP32 weights
        # autocast already casts each projection to BF16, so this changes storage,
        # not values -- and it halves the activation bytes the stream costs.
        embeds = embeds.to(torch.bfloat16)
    if labels[:, 1:].ne(IGNORE_INDEX).sum() == 0:
        raise ValueError('Preprocessing lost supervision for an audited sample; do not discard')
    if int(metadata['valid_mask'].sum(1).max()) > MAX_EXPANDED_LENGTH:
        raise ValueError('Expanded sequence exceeds fixed limit')
    values = dict(input_ids=ids, position_ids=positions, attention_mask=mask,
                  inputs_embeds=embeds, labels=labels, prefix_valid_mask=metadata['valid_mask'])
    return {key: value for key, value in values.items() if value is not None}, metadata
