"""Answer generation from retrieved context."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from .config import load_config
from .retriever import RetrievedChunk


@dataclass
class GeneratedAnswer:
  question: str
  answer: str
  citations: List[str]
  mode: str


def _extractive_answer(question: str, chunks: List[RetrievedChunk], max_chars: int = 1200) -> GeneratedAnswer:
  parts: List[str] = []
  citations: List[str] = []
  used = 0
  for ch in chunks:
    header = f"【{ch.chapter or '未知章节'} / {ch.section or ch.chunk_id}】"
    snippet = ch.text.strip()
    block = f"{header}\n{snippet}"
    if used + len(block) > max_chars:
      remain = max_chars - used
      if remain > 120:
        parts.append(block[:remain] + "…")
        citations.append(ch.chunk_id)
      break
    parts.append(block)
    citations.append(ch.chunk_id)
    used += len(block)

  answer = (
    f"问题：{question}\n\n"
    "根据教材相关内容，检索到的要点如下：\n\n"
    + "\n\n".join(parts)
  )
  return GeneratedAnswer(
    question=question,
    answer=answer,
    citations=citations,
    mode="extractive",
  )


def _llm_answer(question: str, chunks: List[RetrievedChunk]) -> Optional[GeneratedAnswer]:
  try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
  except ImportError:
    return None
  cfg = load_config()

  context = "\n\n".join(
    f"[{i + 1}] ({c.chapter}/{c.section})\n{c.text}" for i, c in enumerate(chunks)
  )
  llm = ChatOpenAI(api_key=cfg.api_key, model=cfg.llm_model, temperature=0.2)
  messages = [
    SystemMessage(
      content=(
        "你是马克思主义基本原理课程助教。仅依据给定教材片段回答问题，"
        "不要编造。若片段不足以回答，请明确说明。回答简洁、条理清晰，"
        "并在句末标注引用编号如[1]。"
      )
    ),
    HumanMessage(content=f"教材片段：\n{context}\n\n问题：{question}"),
  ]
  resp = llm.invoke(messages)
  return GeneratedAnswer(
    question=question,
    answer=str(resp.content),
    citations=[c.chunk_id for c in chunks],
    mode="llm",
  )


def generate_answer(
  question: str,
  chunks: List[RetrievedChunk],
  use_llm: bool = False,
) -> GeneratedAnswer:
  if use_llm:
    llm_result = _llm_answer(question, chunks)
    if llm_result is not None:
      return llm_result
  return _extractive_answer(question, chunks)
