from __future__ import annotations

import time
from collections.abc import Iterable


class MemeDecisionSmoother:
    DEFAULT_FAST_KEYS = frozenset(
        {
            "thumbs_up",
            "kiss",
            "chef_kiss",
            "facepalm",
            "thinking",
            "hand_on_mouth",
            "shocked",
            "wtf_bro",
            "stop",
            "nope",
            "absolute_cinema",
        }
    )

    def __init__(
        self,
        min_candidate_seconds: float = 1.15,
        fast_candidate_seconds: float = 0.60,
        minimum_display_seconds: float = 2.0,
        fast_keys: Iterable[str] = DEFAULT_FAST_KEYS,
        initial_key: str = "neutral",
    ) -> None:
        self.min_candidate_seconds = max(0.0, min_candidate_seconds)
        self.fast_candidate_seconds = max(0.0, fast_candidate_seconds)
        self.minimum_display_seconds = max(0.0, minimum_display_seconds)
        self.fast_keys = {key.lower() for key in fast_keys}

        self.stable_key = initial_key
        self.stable_started_at = -self.minimum_display_seconds
        self.candidate_key: str | None = None
        self.candidate_started_at = 0.0

    def update(
        self,
        suggested_key: str,
        now: float | None = None,
    ) -> str:
        now = time.monotonic() if now is None else now
        suggested_key = suggested_key or "neutral"

        if suggested_key == self.stable_key:
            self.candidate_key = None
            self.candidate_started_at = 0.0
            return self.stable_key

        if suggested_key != self.candidate_key:
            self.candidate_key = suggested_key
            self.candidate_started_at = now

        candidate_age = now - self.candidate_started_at
        stable_age = now - self.stable_started_at

        if (
            candidate_age >= self._candidate_seconds_for(suggested_key)
            and stable_age >= self.minimum_display_seconds
        ):
            self.stable_key = suggested_key
            self.stable_started_at = now
            self.candidate_key = None
            self.candidate_started_at = 0.0

        return self.stable_key

    def _candidate_seconds_for(self, suggested_key: str) -> float:
        if suggested_key.lower() in self.fast_keys:
            return self.fast_candidate_seconds

        return self.min_candidate_seconds
