"""CLI for Marxism textbook RAG system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rag import MarxismRAG

ROOT = Path(__file__).resolve().parent
DEFAULT_RAW = ROOT / "data" / "raw_text.txt"
DEFAULT_INDEX = ROOT / "indices"
DEFAULT_EVAL = ROOT / "data" / "eval_qa.json"
DEFAULT_REPORT = ROOT / "results" / "eval_report.json"


def cmd_build(args: argparse.Namespace) -> None:
  rag = MarxismRAG(args.index_dir)
  rag.build_from_raw_text(args.raw_text, chunk_path=args.chunk_path)
  print(f"Index built at {args.index_dir}")


def cmd_query(args: argparse.Namespace) -> None:
  rag = MarxismRAG(args.index_dir)
  rag.load()
  result = rag.query(
    args.question,
    top_k=args.top_k,
    mode=args.mode,
    use_llm=args.llm,
  )
  print(result.answer)
  if args.show_citations:
    print("\n--- citations ---")
    print(", ".join(result.citations))


def cmd_evaluate(args: argparse.Namespace) -> None:
  rag = MarxismRAG(args.index_dir)
  rag.load()
  report = rag.evaluate(args.eval_path, k=args.top_k, output_path=args.output)
  print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_demo(args: argparse.Namespace) -> None:
  rag = MarxismRAG(args.index_dir)
  if not (args.index_dir / "chunks.json").exists():
    print("Building index...")
    rag.build_from_raw_text(args.raw_text)
  else:
    rag.load()

  questions = [
    "什么是马克思主义的基本特征？",
    "如何理解剩余价值？",
    "实践和认识的关系是什么？",
  ]
  for q in questions:
    print("=" * 60)
    print(f"Q: {q}")
    print("=" * 60)
    ans = rag.query(q, top_k=3, mode="hybrid")
    print(ans.answer[:800])
    print("...")

  report = rag.evaluate(args.eval_path, k=5, output_path=args.output)
  print("\n" + "=" * 60)
  print("Evaluation summary")
  print("=" * 60)
  for mode, metrics in report["retrieval"].items():
    print(
      f"[{mode}] P@5={metrics['precision@5']:.3f} "
      f"R@5={metrics['recall@5']:.3f} MRR={metrics['mrr']:.3f} "
      f"nDCG@5={metrics['ndcg@5']:.3f}"
    )
  gen = report["generation"]
  print(
    f"[generation] char_f1={gen['char_f1']:.3f} "
    f"keyword_coverage={gen['keyword_coverage']:.3f}"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description="马克思主义基本原理 RAG 系统")
  parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
  parser.add_argument("--raw-text", type=Path, default=DEFAULT_RAW)
  parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL)
  sub = parser.add_subparsers(dest="command", required=True)

  p_build = sub.add_parser("build", help="Build retrieval index from raw text")
  p_build.add_argument("--chunk-path", type=Path, default=ROOT / "data" / "chunks.json")
  p_build.set_defaults(func=cmd_build)

  p_query = sub.add_parser("query", help="Ask a question")
  p_query.add_argument("question")
  p_query.add_argument("--top-k", type=int, default=5)
  p_query.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
  p_query.add_argument("--llm", action="store_true", help="Use OpenAI LLM if API key set")
  p_query.add_argument("--show-citations", action="store_true")
  p_query.set_defaults(func=cmd_query)

  p_eval = sub.add_parser("evaluate", help="Run retrieval/generation evaluation")
  p_eval.add_argument("--top-k", type=int, default=5)
  p_eval.add_argument("--output", type=Path, default=DEFAULT_REPORT)
  p_eval.set_defaults(func=cmd_evaluate)

  p_demo = sub.add_parser("demo", help="Build (if needed), sample queries, and evaluate")
  p_demo.add_argument("--output", type=Path, default=DEFAULT_REPORT)
  p_demo.set_defaults(func=cmd_demo)

  args = parser.parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
