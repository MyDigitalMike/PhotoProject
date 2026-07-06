from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemeCandidate:
    key: str
    query: str
    image_url: str
    provider: str
    title: str = ""
    source_url: str = ""
    score: float = 0.0
