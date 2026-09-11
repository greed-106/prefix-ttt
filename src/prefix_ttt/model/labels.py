"""Label semantics: the ignored-target sentinel and supervision classification."""

# Must stay equal to llava.constants.IGNORE_INDEX; kept here so the loss, the data
# pipeline and the audits share one name without importing the LLaVA package.
IGNORE_INDEX = -100


class LabelPreprocessingError(ValueError):
    pass


def classify_supervision(*, has_assistant_content, targets_before_truncation,
                         targets_after_truncation, image_truncated=False):
    if has_assistant_content and targets_before_truncation == 0:
        raise LabelPreprocessingError('Assistant content exists but tokenizer/template produced no shifted targets; repair and re-audit')
    if image_truncated:
        return 'protocol_image_truncated'
    if targets_after_truncation == 0:
        return 'protocol_answer_truncated' if targets_before_truncation else 'data_no_assistant_content'
    return 'valid'
