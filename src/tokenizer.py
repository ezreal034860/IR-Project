"""Simple Chinese-aware tokenizer (no external deps)."""

from __future__ import annotations

import re
from typing import Iterable, List

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str, use_bigram: bool = True) -> List[str]:
    """Tokenize mixed Chinese/English text for sparse retrieval."""
    text = text.lower().strip()
    if not text:
        return []

    tokens: List[str] = []
    i = 0
    while i < len(text):
        m = _CJK.match(text, i)
        if m:
            run_start = i
            while i < len(text) and _CJK.match(text, i):
                i += 1
            run = text[run_start:i]
            tokens.extend(list(run))
            if use_bigram and len(run) >= 2:
                tokens.extend(run[j : j + 2] for j in range(len(run) - 1))
            continue

        m = _WORD.match(text, i)
        if m:
            tokens.append(m.group(0))
            i = m.end()
            continue

        i += 1

    return tokens


def join_tokens(tokens: Iterable[str]) -> str:
    return " ".join(tokens)
