"""Extract text from legacy .doc files using Microsoft Word COM (Windows)."""

from __future__ import annotations

import argparse
from pathlib import Path


def extract_doc(doc_path: Path, output_path: Path) -> None:
  import win32com.client

  word = win32com.client.Dispatch("Word.Application")
  word.Visible = False
  doc = word.Documents.Open(str(doc_path.resolve()))
  text = doc.Content.Text
  doc.Close(False)
  word.Quit()

  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(text, encoding="utf-8")
  print(f"Saved {len(text)} chars -> {output_path}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Extract .doc text via Word COM")
  parser.add_argument("doc_path", type=Path)
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=Path("data/raw_text.txt"),
  )
  args = parser.parse_args()
  extract_doc(args.doc_path, args.output)


if __name__ == "__main__":
  main()
