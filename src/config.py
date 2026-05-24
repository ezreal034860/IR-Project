"""Central project configuration loaded from config.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"


@dataclass(frozen=True)
class AppConfig:
  api_key: str
  base_url: str
  embedding_model: str
  llm_model: str


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
  data = json.loads(path.read_text(encoding="utf-8"))
  return AppConfig(
    api_key=data["api_key"],
    base_url=data["base_url"],
    embedding_model=data["embedding_model"],
    llm_model=data["llm_model"],
  )
