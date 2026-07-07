from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemeMatchCandidate:
    key: str
    score: float
