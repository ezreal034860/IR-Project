"""Build and persist BM25 + TF-IDF/FAISS indices."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .bm25 import BM25Index
from .preprocess import Chunk, load_chunks
from .tokenizer import join_tokens, tokenize


@dataclass
class IndexBundle:
  chunks: List[Chunk]
  bm25: BM25Index
  vectorizer: TfidfVectorizer
  faiss_index: faiss.Index
  dense_matrix: np.ndarray


def _tokenized_corpus(chunks: Sequence[Chunk]) -> List[str]:
  return [join_tokens(tokenize(c.text)) for c in chunks]


def _analyzer(text: str) -> List[str]:
  return text.split()


def build_index(chunks: List[Chunk]) -> IndexBundle:
  texts = [c.text for c in chunks]
  tokenized = _tokenized_corpus(chunks)

  bm25 = BM25Index().fit(texts)

  vectorizer = TfidfVectorizer(
    analyzer=_analyzer,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
  )
  matrix = vectorizer.fit_transform(tokenized).astype(np.float32)
  dense = matrix.toarray()
  faiss.normalize_L2(dense)
  index = faiss.IndexFlatIP(dense.shape[1])
  index.add(dense)

  return IndexBundle(
    chunks=chunks,
    bm25=bm25,
    vectorizer=vectorizer,
    faiss_index=index,
    dense_matrix=dense,
  )


def save_index(bundle: IndexBundle, index_dir: Path) -> None:
  index_dir.mkdir(parents=True, exist_ok=True)
  chunk_path = index_dir / "chunks.json"
  chunk_path.write_text(
    json.dumps([c.__dict__ for c in bundle.chunks], ensure_ascii=False, indent=2),
    encoding="utf-8",
  )
  with open(index_dir / "bm25.pkl", "wb") as f:
    pickle.dump(bundle.bm25, f)
  with open(index_dir / "vectorizer.pkl", "wb") as f:
    pickle.dump(bundle.vectorizer, f)
  faiss.write_index(bundle.faiss_index, str(index_dir / "faiss.index"))
  np.save(index_dir / "dense.npy", bundle.dense_matrix)


def load_index(index_dir: Path) -> IndexBundle:
  chunks = load_chunks(index_dir / "chunks.json")
  with open(index_dir / "bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)
  with open(index_dir / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
  faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
  dense = np.load(index_dir / "dense.npy")
  return IndexBundle(
    chunks=chunks,
    bm25=bm25,
    vectorizer=vectorizer,
    faiss_index=faiss_index,
    dense_matrix=dense,
  )
