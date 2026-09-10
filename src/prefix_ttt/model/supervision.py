"""Derive v1 assistant spans from original rendered prefixes, not round offsets."""
import torch
from llava.constants import IGNORE_INDEX
from llava.mm_utils import tokenizer_image_token
from .labels import LabelPreprocessingError


def v1_targets(conversation, tokenizer, input_ids, *, has_image):
    def encode(text):
        if has_image:
            return tokenizer_image_token(text, tokenizer)
        return tokenizer(text, truncation=False).input_ids

    prompt = conversation.get_prompt()
    complete = encode(prompt)
    actual = input_ids.tolist()
    n = min(len(actual), len(complete))
    if actual[:n] != complete[:n]:
        raise LabelPreprocessingError('Full prompt tokenization is not prefix consistent')
    target = torch.full_like(input_ids, IGNORE_INDEX)
    for index, (role, answer) in enumerate(conversation.messages):
        if role != conversation.roles[1] or not answer:
            continue
        prefix = conversation.copy()
        prefix.messages = conversation.messages[:index] + [[role, None]]
        header = encode(prefix.get_prompt() + ' ')
        # Tokenization of trailing whitespace can merge with the first answer
        # token. Longest common prefix excludes that boundary token safely.
        start = 0
        for left, right in zip(header, complete):
            if left != right:
                break
            start += 1
        prefix.messages = conversation.messages[:index + 1]
        end_ids = encode(prefix.get_prompt())
        if complete[:len(end_ids)] != end_ids:
            raise LabelPreprocessingError('Assistant end prefix changed tokenization; repair tokenizer/template')
        end = min(len(end_ids), len(target))
        target[start:end] = input_ids[start:end]
    return target
