"""Content identities used to prove that two files or orders are the same object.

Small enough to be imported by anything, including the training and serving paths
that must not pull in the manifest builder's parsing dependencies.
"""
import hashlib
import json
from pathlib import Path


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()
