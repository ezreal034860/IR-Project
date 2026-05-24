"""End-to-end RAG pipeline."""

from __future__ import annotations

from pathlib import Path

from .evaluate import run_full_evaluation
from .generator import GeneratedAnswer, generate_answer
from .indexer import IndexBundle, build_index, load_index, save_index
from .preprocess import build_chunks, load_chunks, save_chunks
from .retriever import HybridRetriever, RetrievalMode


class MarxismRAG:
  def __init__(self, index_dir: Path) -> None:
    self.index_dir = Path(index_dir)
    self.bundle: IndexBundle | None = None
    self.retriever: HybridRetriever | None = None

  def build_from_raw_text(self, raw_text_path: Path, chunk_path: Path | None = None) -> None:
    text = raw_text_path.read_text(encoding="utf-8")
    chunks = build_chunks(text)
    if chunk_path:
      save_chunks(chunks, chunk_path)
    self.bundle = build_index(chunks)
    save_index(self.bundle, self.index_dir)
    self.retriever = HybridRetriever(self.bundle)

  def load(self) -> None:
    self.bundle = load_index(self.index_dir)
    self.retriever = HybridRetriever(self.bundle)

  def query(
    self,
    question: str,
    top_k: int = 5,
    mode: RetrievalMode = "hybrid",
    use_llm: bool = False,
  ) -> GeneratedAnswer:
    if self.retriever is None:
      self.load()
    assert self.retriever is not None
    chunks = self.retriever.retrieve(question, top_k=top_k, mode=mode)
    return generate_answer(question, chunks, use_llm=use_llm)

  def evaluate(self, eval_path: Path, k: int = 5, output_path: Path | None = None):
    if self.bundle is None:
      self.load()
    assert self.bundle is not None
    return run_full_evaluation(self.bundle, eval_path, k=k, output_path=output_path)
