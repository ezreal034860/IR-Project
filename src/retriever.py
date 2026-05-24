"""Hybrid retrieval: BM25 + dense embedding with reciprocal rank fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple

import faiss
import numpy as np
from openai import OpenAI

from .config import load_config
from .indexer import IndexBundle

RetrievalMode = Literal["bm25", "dense", "hybrid"]


@dataclass
class RetrievedChunk:
  chunk_id: str
  text: str
  chapter: str
  section: str
  score: float
  rank: int
  source: str


class HybridRetriever:
  def __init__(self, bundle: IndexBundle, rrf_k: int = 60) -> None:
    self.bundle = bundle
    self.rrf_k = rrf_k
    cfg = load_config()
    self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

  def _dense_scores(self, query: str, top_k: int) -> List[Tuple[int, float]]:
    resp = self.client.embeddings.create(
      model=self.bundle.embedding_model,
      input=query,
    )
    q_vec = np.asarray([resp.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(q_vec)
    scores, indices = self.bundle.faiss_index.search(q_vec, top_k)
    return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]

  def _rrf_fuse(
    self,
    ranked_lists: Sequence[Sequence[Tuple[int, float]]],
  ) -> List[Tuple[int, float]]:
    fused: Dict[int, float] = {}
    for ranked in ranked_lists:
      for rank, (doc_id, _score) in enumerate(ranked, start=1):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)

  def retrieve(
    self,
    query: str,
    top_k: int = 5,
    mode: RetrievalMode = "hybrid",
    candidate_k: int = 20,
  ) -> List[RetrievedChunk]:
    bm25_hits = self.bundle.bm25.score(query, top_k=candidate_k)
    dense_hits = self._dense_scores(query, top_k=candidate_k)

    if mode == "bm25":
      fused = bm25_hits
    elif mode == "dense":
      fused = dense_hits
    else:
      fused = self._rrf_fuse([bm25_hits, dense_hits])

    results: List[RetrievedChunk] = []
    for rank, (doc_idx, score) in enumerate(fused[:top_k], start=1):
      ch = self.bundle.chunks[doc_idx]
      results.append(
        RetrievedChunk(
          chunk_id=ch.chunk_id,
          text=ch.text,
          chapter=ch.chapter,
          section=ch.section,
          score=score,
          rank=rank,
          source=mode,
        )
      )
    return results
