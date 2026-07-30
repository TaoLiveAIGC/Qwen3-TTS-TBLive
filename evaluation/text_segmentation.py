# coding=utf-8
"""Text segmentation for long-text TTS inference.

Splits long text into utterance-sized segments based on punctuation,
respecting min/max length bounds. Used by zero-shot inference to handle
inputs that exceed a single forward pass.
"""

import re
from typing import List, Union


_ERASE_PUNCS = r'[“”"‘’\'（）()【】[\]{}<>《》〈〉〔〕〖〗〘〙〚〛〛〞〟]'
_HARD_SPLIT_PUNCS = r'[?!。？！~：]'
_SOFT_SPLIT_PUNCS = r'(?<!>)[；?!。？！~：:，,—…](?!<)'
_DEDUP_PUNCS = list('；?!。？！~：:，,、—…')


def _count_words_multilang(text: str) -> int:
    """Count CJK chars + English words; whitespace separates English words."""
    word_count = 0
    in_word = False
    for char in text:
        if char.isspace():
            in_word = False
        elif char.isascii() and not in_word:
            word_count += 1
            in_word = True
        elif not char.isascii():
            word_count += 1
    return word_count


def _count_words_no_punc(text: str) -> int:
    cleaned = re.sub(r'[^\w\s]', '', text)
    return _count_words_multilang(cleaned)


def _cut_sentence_multilang(text: str, max_length: int):
    """Hard-cut a sentence at the position where word count exceeds max_length."""
    word_count = 0
    in_word = False
    for index, char in enumerate(text):
        if char.isspace():
            in_word = False
        elif char.isascii() and not in_word:
            word_count += 1
            in_word = True
        elif not char.isascii():
            word_count += 1
        if word_count > max_length:
            return text[:index], text[index:]
    return text, ""


def _split_long_sentence(text: str, max_length: int) -> str:
    """Force-split lines that exceed max_length, joined back by '\\n'."""
    opts = []
    for sentence in text.split('\n'):
        prev_text, sentence = _cut_sentence_multilang(sentence, max_length)
        while sentence.strip() != "":
            opts.append(prev_text)
            prev_text, sentence = _cut_sentence_multilang(sentence, max_length)
        opts.append(prev_text)
    return "\n".join(opts)


def _process_commas(text: str, max_length: int, min_length: int, force_cut: bool = True) -> str:
    """Split by commas/soft puncs, merge runs to stay within [min_length, max_length]."""
    # Dedup adjacent puncts (e.g. "。。" → "。")
    for a in _DEDUP_PUNCS:
        for b in _DEDUP_PUNCS:
            if a + b in text:
                text = text.replace(a + b, a)

    items = re.split(f'({_SOFT_SPLIT_PUNCS})', text)
    if len(items) % 2 == 1:
        items.append("")
    sentences = ["".join(group).replace("<|>", "") for group in zip(items[::2], items[1::2])]

    if force_cut:
        final_sentences = []
        for sentence in sentences:
            if _count_words_multilang(sentence) > max_length:
                final_sentences += _split_long_sentence(sentence, max_length=max_length).split("\n")
            else:
                final_sentences.append(sentence)
    else:
        final_sentences = sentences[:]

    processed = [""]
    current_line = ""
    for sentence in final_sentences:
        if _count_words_no_punc(current_line) < min_length:
            current_line += sentence
        else:
            processed.append(current_line.strip())
            current_line = sentence + " "

    if _count_words_no_punc(current_line) < min_length:
        processed[-1] += current_line.strip()
    else:
        processed.append(current_line.strip())

    return "\n".join(processed)


def auto_cut_llm(
    text: str,
    max_length: int = 70,
    min_length: int = 35,
    return_type: str = "list",
    force_cut: bool = True,
) -> Union[str, List[str]]:
    """Split long text into utterance-sized chunks for TTS synthesis.

    Pipeline:
      1. Strip noisy quotes / brackets / em-dashes.
      2. Normalize CJK-English spacing and unify ~ … — to commas.
      3. Hard-split by [?!。？！~：], then for each piece run process_commas
         to merge into chunks of length in [min_length, max_length].

    Args:
        text: Input text (may contain newlines, control tags, mixed languages).
        max_length: Soft upper bound on per-chunk word count.
        min_length: Lower bound used to merge short consecutive pieces.
        return_type: "list" returns List[str]; "str" returns "\\n"-joined string.
        force_cut: When True, hard-split any sentence still longer than max_length.

    Returns:
        List[str] of segments, or a "\\n"-joined string if return_type="str".
    """
    text = re.sub(_ERASE_PUNCS, '', text)
    text = text.replace("\n", "")
    text = text.strip("\n")
    text = text.replace(". ", "。")
    text = text.replace("~", "，").replace("～", "，").replace("…", "，")
    # Keep English-English spaces, drop spaces between CJK or CJK-English
    text = re.sub(r'(?<=[^\x00-\x7f]) +| +(?=[^\x00-\x7f])', '', text)
    text = re.sub(r'—+', '，', text)
    text = re.sub(r'-+', '，', text)

    if not text:
        text = " "
    if text[-1] not in '?!。？！~：':
        text += "。"

    final_items = []
    for item in [text]:  # outer list reserved for future paragraph split
        final_items += _process_commas(item, max_length, min_length, force_cut).split("\n")

    final_items = [
        item for item in final_items
        if item.strip()
        and not (len(item.strip()) == 2 and item[0] in "?!，,。？！~：.")
        and not item.isspace()
        and item not in "?!，,。？！~：."
    ]

    if return_type == "str":
        return "\n".join(final_items)
    return final_items


__all__ = ["auto_cut_llm"]
