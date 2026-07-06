from __future__ import annotations

import time

from src.domain.emotion_result import EmotionResult


class EmotionSmoother:
    def __init__(
        self,
        required_repeats: int = 2,
        min_stable_seconds: float = 0.0,
    ) -> None:
        self.required_repeats = required_repeats
        self.min_stable_seconds = min_stable_seconds
        self.candidate_emotion: str | None = None
        self.candidate_count = 0
        self.candidate_started_at = 0.0
        self.stable_emotion = "neutral"

    def update(
        self,
        emotion_result: EmotionResult,
        now: float | None = None,
    ) -> str:
        now = time.monotonic() if now is None else now
        candidate_emotion = emotion_result.label or "neutral"

        if candidate_emotion == self.candidate_emotion:
            self.candidate_count += 1
        else:
            self.candidate_emotion = candidate_emotion
            self.candidate_count = 1
            self.candidate_started_at = now

        repeated_enough = self.candidate_count >= self.required_repeats
        held_long_enough = (
            now - self.candidate_started_at
        ) >= self.min_stable_seconds

        if candidate_emotion == self.stable_emotion:
            self.stable_emotion = candidate_emotion
        elif repeated_enough and held_long_enough:
            self.stable_emotion = candidate_emotion

        return self.stable_emotion
