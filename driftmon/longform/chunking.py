"""Split long documents into chunks for embedding (D26).

Strategy: never cross a section boundary; inside a section, pack whole
paragraphs (then sentences, then words, only when a piece is too long) into
chunks of about ``size`` tokens, and start each chunk with the last
sentences of the previous one, up to ``overlap`` tokens, so a fact that
spans a boundary is still whole in one chunk.

Token counting is passed in (the embedding model's tokenizer in the
pipeline, a word count in tests), so this module stays pure and fast to test.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

CountFn = Callable[[str], int]
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass(frozen=True)
class Chunk:
    """One piece of a document, ready to embed."""

    section: str
    text: str
    tokens: int


def _pieces(paragraph: str, count: CountFn, size: int) -> list[str]:
    """Split a paragraph into pieces no longer than ``size`` tokens:
    sentences first, words only for a sentence that is itself too long."""
    if count(paragraph) <= size:
        return [paragraph]
    out = []
    for sentence in _SENTENCE.split(paragraph):
        if count(sentence) <= size:
            out.append(sentence)
            continue
        words, current = sentence.split(), []
        for w in words:
            if current and count(" ".join(current + [w])) > size:
                out.append(" ".join(current))
                current = []
            current.append(w)
        if current:
            out.append(" ".join(current))
    return out


def _tail(text: str, count: CountFn, overlap: int) -> str:
    """The last whole sentences of ``text`` that fit in ``overlap`` tokens."""
    if overlap <= 0:
        return ""
    kept: list[str] = []
    for sentence in reversed(_SENTENCE.split(text)):
        if count(" ".join([sentence] + kept)) > overlap:
            break
        kept.insert(0, sentence)
    return " ".join(kept)


def chunk_section(section: str, text: str, count: CountFn, size: int = 256, overlap: int = 32,
                  min_tokens: int = 20) -> list[Chunk]:
    """Chunk one section. Paragraphs are separated by blank lines.

    A final piece shorter than ``min_tokens`` is merged into the previous
    chunk (if that stays within 1.25 × size) rather than embedded on its own.
    """
    pieces = [p for para in re.split(r"\n\s*\n", text) if para.strip()
              for p in _pieces(" ".join(para.split()), count, size)]
    chunks: list[str] = []
    current, prefix = [], ""
    for piece in pieces:
        candidate = " ".join(([prefix] if prefix else []) + current + [piece])
        if current and count(candidate) > size:
            chunks.append(" ".join(([prefix] if prefix else []) + current))
            prefix = _tail(" ".join(current), count, overlap)
            current = []
        current.append(piece)
    if current:
        last = " ".join(([prefix] if prefix else []) + current)
        if chunks and count(" ".join(current)) < min_tokens and count(chunks[-1] + " " + " ".join(current)) <= size * 1.25:
            chunks[-1] = chunks[-1] + " " + " ".join(current)
        else:
            chunks.append(last)
    return [Chunk(section, c, count(c)) for c in chunks]


def chunk_document(sections: Sequence[tuple[str, str]], count: CountFn, size: int = 256, overlap: int = 32,
                   min_tokens: int = 20) -> list[Chunk]:
    """Chunk every (heading, text) section of a document, in order."""
    return [c for heading, text in sections for c in chunk_section(heading, text, count, size, overlap, min_tokens)]


def truncate(text: str, count: CountFn, limit: int = 512) -> str:
    """The first ``limit`` tokens' worth of whole words: what a document-level
    embedding of a long article actually reads (the model ignores the rest)."""
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:                       # longest prefix that fits
        mid = (lo + hi + 1) // 2
        if count(" ".join(words[:mid])) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


def token_counter(model_name: str) -> CountFn:
    """Token count with the embedding model's own tokenizer (cached per string)."""
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()   # counting long texts is fine; only the model is limited to 512
    tok = AutoTokenizer.from_pretrained(model_name)

    @lru_cache(maxsize=500_000)
    def count(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])
    return count
