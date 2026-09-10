from pathlib import Path

import pytest

from prefix_ttt.audit_data import audit_labels, init_labels, decode_image


MODEL = Path('data/llava-v1.5-assets-v1/models/llava-1.5-7b-hf-b234b804b114d9e37bb655e11cbbb5f5e971b7a9')


@pytest.fixture(scope='module', autouse=True)
def tokenizer():
    if not MODEL.exists():
        pytest.skip('local checkpoint tokenizer required')
    init_labels(str(MODEL))


def row(question='What?', answer='A useful answer.', image=None):
    result = {'id': 'fixture', 'conversations': [
        {'from': 'human', 'value': question}, {'from': 'gpt', 'value': answer}]}
    if image:
        result['image'] = image
    return result


def test_real_label_audit_distinguishes_data_and_preprocessing():
    result = audit_labels((0, row('<image>\nWhat?', image='coco/example.jpg')))
    assert result['status'] == 'valid'
    assert result['expanded_length'] == result['text_token_count'] + 575
    missing_sentinel = audit_labels((1, row(image='coco/example.jpg')))
    assert missing_sentinel['status'] == 'preprocessing_error'
    assert 'sentinel mismatch' in missing_sentinel['error']
    assert audit_labels((2, row(answer='')))['status'] == 'data_no_assistant_content'
    assert audit_labels((3, {'id': 'broken'}))['status'] == 'data_invalid_conversations'


def test_answer_truncated_is_not_preprocessing_failure():
    result = audit_labels((4, row(question='word ' * 2500)))
    assert result['status'] == 'protocol_answer_truncated'
    assert result['targets_before'] > 0 and result['targets_after'] == 0


def test_image_decode_failure_retains_evidence(tmp_path):
    result = decode_image(tmp_path / 'missing.jpg')
    assert result['error_type'] == 'FileNotFoundError'
    assert result['path'].endswith('missing.jpg')
