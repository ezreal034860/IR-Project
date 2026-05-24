"""Document cleaning and hierarchical chunking."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

CHAPTER_RE = re.compile(r"第[一二三四五六七八九十百]+章[^\r\n\t]{0,40}")
SECTION_RE = re.compile(r"第[一二三四五六七八九十百]+节[^\r\n\t]{0,40}")
PAGE_NUM_RE = re.compile(r"\t\d+\s*$", re.MULTILINE)
NOISE_RE = re.compile(r"[\x0c\r]+")


@dataclass
class Chunk:
  chunk_id: str
  chapter: str
  section: str
  title: str
  text: str
  char_start: int
  char_end: int


def clean_text(text: str) -> str:
  text = NOISE_RE.sub("\n", text)
  text = PAGE_NUM_RE.sub("", text)
  text = re.sub(r"\n{3,}", "\n\n", text)
  return text.strip()


def _split_sections(text: str) -> List[tuple[str, str, str, int]]:
  """Return (chapter, section, body, start_offset) blocks."""
  markers: List[tuple[int, str, str]] = []
  for m in CHAPTER_RE.finditer(text):
    markers.append((m.start(), "chapter", m.group(0).strip()))
  for m in SECTION_RE.finditer(text):
    markers.append((m.start(), "section", m.group(0).strip()))
  markers.sort(key=lambda x: x[0])

  if not markers:
    return [("", "", text, 0)]

  blocks: List[tuple[str, str, str, int]] = []
  chapter = ""
  section = ""
  for i, (pos, kind, label) in enumerate(markers):
    end = markers[i + 1][0] if i + 1 < len(markers) else len(text)
    body = text[pos:end].strip()
    if kind == "chapter":
      chapter = label
      section = ""
    else:
      section = label
    blocks.append((chapter, section, body, pos))
  return blocks


def _window_chunks(
  body: str,
  chunk_size: int,
  overlap: int,
  base_offset: int,
  chapter: str,
  section: str,
  prefix: str,
) -> List[Chunk]:
  if not body.strip():
    return []

  chunks: List[Chunk] = []
  start = 0
  idx = 0
  while start < len(body):
    end = min(start + chunk_size, len(body))
  # extend to sentence boundary when possible
    if end < len(body):
      for sep in "。！？；\n":
        cut = body.rfind(sep, start, end)
        if cut > start + chunk_size // 2:
          end = cut + 1
          break
    piece = body[start:end].strip()
    if len(piece) >= 80 and not _is_learning_objective_chunk(piece):
      title = section or chapter or prefix
      cid = f"{prefix}_{idx:04d}"
      chunks.append(
        Chunk(
          chunk_id=cid,
          chapter=chapter,
          section=section,
          title=title,
          text=piece,
          char_start=base_offset + start,
          char_end=base_offset + end,
        )
      )
      idx += 1
    if end >= len(body):
      break
    start = max(end - overlap, start + 1)
  return chunks


def _find_content_start(text: str, max_scan: int = 15000) -> int:
  """Skip cover + table of contents; keep 导论 and main chapters."""
  scan = text[:max_scan]
  last_toc = 0
  for m in PAGE_NUM_RE.finditer(scan):
    last_toc = m.end()
  if last_toc:
    return last_toc
  for marker in ("\x0c导\u3000论", "导\u3000论", "一、什么是马克思主义"):
    idx = text.find(marker)
    if idx != -1:
      return idx
  return 0


def _is_learning_objective_chunk(text: str) -> bool:
  stripped = text.strip()
  if stripped.startswith("学习目标") or stripped.startswith("学习要点"):
    return True
  head = stripped[:120]
  return head.startswith("第") and "学习目标" in head[:80]


def build_chunks(
  text: str,
  chunk_size: int = 500,
  overlap: int = 100,
  skip_toc_chars: int | None = None,
) -> List[Chunk]:
  """Build retrieval chunks; skip front matter / table of contents."""
  text = clean_text(text)
  start = skip_toc_chars if skip_toc_chars is not None else _find_content_start(text)
  body = text[start:]
  offset = start

  all_chunks: List[Chunk] = []
  blocks = _split_sections(body)
  for chapter, section, block_text, block_start in blocks:
    if _is_learning_objective_chunk(block_text):
      continue
    prefix = re.sub(r"\W+", "_", (chapter or "intro")[:12]) or "intro"
    all_chunks.extend(
      _window_chunks(
        block_text,
        chunk_size=chunk_size,
        overlap=overlap,
        base_offset=offset + block_start,
        chapter=chapter,
        section=section,
        prefix=prefix,
      )
    )

  # reassign sequential ids
  for i, ch in enumerate(all_chunks):
    ch.chunk_id = f"chunk_{i:04d}"
  return all_chunks


def save_chunks(chunks: List[Chunk], path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = [asdict(c) for c in chunks]
  path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks(path: Path) -> List[Chunk]:
  data = json.loads(path.read_text(encoding="utf-8"))
  return [Chunk(**item) for item in data]
