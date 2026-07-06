from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemeProfile:
    key: str
    priority: float = 0.0
    min_total_score: float = 0.0

    emotion_weights: dict[str, float] = field(default_factory=dict)
    signal_weights: dict[str, float] = field(default_factory=dict)

    min_scores: dict[str, float] = field(default_factory=dict)
    any_min_scores: dict[str, float] = field(default_factory=dict)

    required_signals: tuple[str, ...] = ()
    any_signals: tuple[str, ...] = ()
    blocked_signals: tuple[str, ...] = ()

    search_terms: tuple[str, ...] = ()

    description: str = ""
