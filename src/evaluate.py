"""Retrieval and generation evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from .generator import generate_answer
from .indexer import IndexBundle
from .retriever import HybridRetriever, RetrievalMode


@dataclass
class EvalItem:
  id: str
  question: str
  reference_answer: str
  relevance_keywords: List[str]
  min_keyword_hits: int = 2


@dataclass
class RetrievalMetrics:
  precision_at_k: float
  recall_at_k: float
  mrr: float
  ndcg_at_k: float
  hit_at_k: float


@dataclass
class GenerationMetrics:
  char_f1: float
  keyword_coverage: float


def load_eval_set(path: Path) -> List[EvalItem]:
  data = json.loads(path.read_text(encoding="utf-8"))
  return [EvalItem(**item) for item in data]


def _chunk_relevance(chunk_text: str, keywords: Sequence[str], min_hits: int) -> bool:
  hits = sum(1 for kw in keywords if kw in chunk_text)
  return hits >= min_hits


def _build_relevance_labels(bundle: IndexBundle, item: EvalItem) -> List[int]:
  labels = []
  for ch in bundle.chunks:
    labels.append(
      1
      if _chunk_relevance(ch.text, item.relevance_keywords, item.min_keyword_hits)
      else 0
    )
  return labels


def _precision_recall_at_k(retrieved: List[int], labels: List[int], k: int) -> tuple[float, float]:
  top = retrieved[:k]
  if not top:
    return 0.0, 0.0
  rel = sum(labels[i] for i in top)
  total_rel = sum(labels)
  p = rel / len(top)
  r = rel / total_rel if total_rel else 0.0
  return p, r


def _mrr(retrieved: List[int], labels: List[int]) -> float:
  for rank, idx in enumerate(retrieved, start=1):
    if labels[idx]:
      return 1.0 / rank
  return 0.0


def _ndcg_at_k(retrieved: List[int], labels: List[int], k: int) -> float:
  def dcg(scores: List[int]) -> float:
    return sum((2**s - 1) / __import__("math").log2(i + 2) for i, s in enumerate(scores))

  top = retrieved[:k]
  gains = [labels[i] for i in top]
  ideal = sorted(labels, reverse=True)[:k]
  idcg = dcg(ideal)
  if idcg == 0:
    return 0.0
  return dcg(gains) / idcg


def _char_f1(pred: str, ref: str) -> float:
  pred_chars = list(pred)
  ref_chars = list(ref)
  if not pred_chars or not ref_chars:
    return 0.0
  common = {}
  for c in pred_chars:
    common[c] = common.get(c, 0) + 1
  overlap = 0
  for c in ref_chars:
    if common.get(c, 0) > 0:
      overlap += 1
      common[c] -= 1
  if overlap == 0:
    return 0.0
  precision = overlap / len(pred_chars)
  recall = overlap / len(ref_chars)
  return 2 * precision * recall / (precision + recall)


def _keyword_coverage(text: str, keywords: Sequence[str]) -> float:
  if not keywords:
    return 0.0
  hit = sum(1 for kw in keywords if kw in text)
  return hit / len(keywords)


def evaluate_retrieval(
  retriever: HybridRetriever,
  bundle: IndexBundle,
  eval_items: Sequence[EvalItem],
  k: int = 5,
  mode: RetrievalMode = "hybrid",
) -> Dict[str, float]:
  id_to_idx = {c.chunk_id: i for i, c in enumerate(bundle.chunks)}
  p_list, r_list, mrr_list, ndcg_list, hit_list = [], [], [], [], []

  for item in eval_items:
    labels = _build_relevance_labels(bundle, item)
    hits = retriever.retrieve(item.question, top_k=max(k, 20), mode=mode)
    retrieved_ids = [id_to_idx[h.chunk_id] for h in hits if h.chunk_id in id_to_idx]
    p, r = _precision_recall_at_k(retrieved_ids, labels, k)
    p_list.append(p)
    r_list.append(r)
    mrr_list.append(_mrr(retrieved_ids, labels))
    ndcg_list.append(_ndcg_at_k(retrieved_ids, labels, k))
    hit_list.append(1.0 if any(labels[i] for i in retrieved_ids[:k]) else 0.0)

  n = len(eval_items) or 1
  return {
    f"precision@{k}": sum(p_list) / n,
    f"recall@{k}": sum(r_list) / n,
    "mrr": sum(mrr_list) / n,
    f"ndcg@{k}": sum(ndcg_list) / n,
    f"hit@{k}": sum(hit_list) / n,
    "mode": mode,
    "num_queries": len(eval_items),
  }


def evaluate_generation(
  retriever: HybridRetriever,
  eval_items: Sequence[EvalItem],
  k: int = 3,
  mode: RetrievalMode = "hybrid",
) -> Dict[str, float]:
  f1_list, cov_list = [], []
  for item in eval_items:
    chunks = retriever.retrieve(item.question, top_k=k, mode=mode)
    ans = generate_answer(item.question, chunks, use_llm=False)
    f1_list.append(_char_f1(ans.answer, item.reference_answer))
    cov_list.append(_keyword_coverage(ans.answer, item.relevance_keywords))
  n = len(eval_items) or 1
  return {
    "char_f1": sum(f1_list) / n,
    "keyword_coverage": sum(cov_list) / n,
    "mode": mode,
    "num_queries": len(eval_items),
  }


def run_full_evaluation(
  bundle: IndexBundle,
  eval_path: Path,
  k: int = 5,
  output_path: Path | None = None,
) -> Dict[str, object]:
  eval_items = load_eval_set(eval_path)
  retriever = HybridRetriever(bundle)

  retrieval_reports = {
    mode: evaluate_retrieval(retriever, bundle, eval_items, k=k, mode=mode)
    for mode in ("bm25", "dense", "hybrid")
  }
  generation_report = evaluate_generation(retriever, eval_items, k=3, mode="hybrid")

  report = {
    "retrieval": retrieval_reports,
    "generation": generation_report,
    "eval_set_size": len(eval_items),
    "top_k": k,
  }

  if output_path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
  return report
