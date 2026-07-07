from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable

from src.application.meme_match_candidate import MemeMatchCandidate


class MemeVarietySelector:
    def __init__(
        self,
        max_ranked_candidates: int = 6,
        score_window: float = 140.0,
        relative_score_floor: float = 0.82,
        repeat_penalty: float = 75.0,
        stable_key_penalty: float = 35.0,
        neutral_penalty: float = 50.0,
        recent_history_size: int = 10,
        recent_history_seconds: float = 120.0,
    ) -> None:
        self.max_ranked_candidates = max(1, max_ranked_candidates)
        self.score_window = max(0.0, score_window)
        self.relative_score_floor = min(max(relative_score_floor, 0.0), 1.0)
        self.repeat_penalty = max(0.0, repeat_penalty)
        self.stable_key_penalty = max(0.0, stable_key_penalty)
        self.neutral_penalty = max(0.0, neutral_penalty)
        self.recent_history_seconds = max(0.0, recent_history_seconds)
        self._recent_keys: deque[tuple[float, str]] = deque(
            maxlen=max(1, recent_history_size),
        )
        self._last_recorded_key: str | None = None

    def select(
        self,
        candidates: Iterable[MemeMatchCandidate],
        stable_key: str = "neutral",
        now: float | None = None,
    ) -> str:
        now = time.monotonic() if now is None else now
        ranked_candidates = tuple(candidates)

        if not ranked_candidates:
            return stable_key or "neutral"

        candidate_pool = self._candidate_pool(ranked_candidates)
        self._trim_recent_keys(now)

        selected_candidate = max(
            candidate_pool,
            key=lambda candidate: self._adjusted_score(
                candidate=candidate,
                stable_key=stable_key,
                now=now,
            ),
        )

        return selected_candidate.key

    def record_displayed_key(
        self,
        meme_key: str,
        now: float | None = None,
    ) -> None:
        meme_key = meme_key or "neutral"

        if meme_key == self._last_recorded_key:
            return

        now = time.monotonic() if now is None else now
        self._recent_keys.append((now, meme_key))
        self._last_recorded_key = meme_key
        self._trim_recent_keys(now)

    def _candidate_pool(
        self,
        ranked_candidates: tuple[MemeMatchCandidate, ...],
    ) -> tuple[MemeMatchCandidate, ...]:
        best_score = ranked_candidates[0].score
        minimum_score = max(
            best_score - self.score_window,
            best_score * self.relative_score_floor,
        )

        candidate_pool = [
            candidate
            for candidate in ranked_candidates[: self.max_ranked_candidates]
            if candidate.score >= minimum_score
        ]

        return tuple(candidate_pool or (ranked_candidates[0],))

    def _adjusted_score(
        self,
        candidate: MemeMatchCandidate,
        stable_key: str,
        now: float,
    ) -> float:
        recent_count = self._recent_count(candidate.key, now)
        adjusted_score = candidate.score - (recent_count * self.repeat_penalty)

        if candidate.key == stable_key:
            adjusted_score -= self.stable_key_penalty

        if candidate.key == "neutral":
            adjusted_score -= self.neutral_penalty

        return adjusted_score

    def _recent_count(
        self,
        meme_key: str,
        now: float,
    ) -> int:
        return sum(
            1
            for recorded_at, recorded_key in self._recent_keys
            if recorded_key == meme_key
            and now - recorded_at <= self.recent_history_seconds
        )

    def _trim_recent_keys(self, now: float) -> None:
        if self.recent_history_seconds <= 0:
            self._recent_keys.clear()
            return

        while self._recent_keys:
            recorded_at, _recorded_key = self._recent_keys[0]

            if now - recorded_at <= self.recent_history_seconds:
                break

            self._recent_keys.popleft()
