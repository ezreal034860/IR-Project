"""BM25 sparse retrieval implementation."""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Sequence, Tuple

from .tokenizer import tokenize


class BM25Index:
  def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
    self.k1 = k1
    self.b = b
    self.doc_tokens: List[List[str]] = []
    self.doc_len: List[int] = []
    self.avgdl = 0.0
    self.df: Counter[str] = Counter()
    self.N = 0

  def fit(self, documents: Sequence[str]) -> "BM25Index":
    self.doc_tokens = [tokenize(doc) for doc in documents]
    self.doc_len = [len(toks) for toks in self.doc_tokens]
    self.N = len(documents)
    self.avgdl = sum(self.doc_len) / self.N if self.N else 0.0
    self.df = Counter()
    for toks in self.doc_tokens:
      for term in set(toks):
        self.df[term] += 1
    return self

  def _idf(self, term: str) -> float:
    n = self.df.get(term, 0)
    return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

  def score(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
    q_tokens = tokenize(query)
    if not q_tokens or not self.doc_tokens:
      return []

    scores = [0.0] * self.N
    for term in q_tokens:
      idf = self._idf(term)
      for idx, toks in enumerate(self.doc_tokens):
        tf = toks.count(term)
        if tf == 0:
          continue
        dl = self.doc_len[idx]
        denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        scores[idx] += idf * (tf * (self.k1 + 1)) / denom

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(i, s) for i, s in ranked[:top_k] if s > 0]
