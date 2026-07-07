from __future__ import annotations

import time
from collections import deque

from src.domain.emotion_result import EmotionResult


class EmotionResultSmoother:
    def __init__(
        self,
        window_seconds: float = 1.6,
        max_samples: int = 5,
        recency_bias: float = 0.30,
    ) -> None:
        self.window_seconds = max(0.1, window_seconds)
        self.max_samples = max(1, max_samples)
        self.recency_bias = min(max(recency_bias, 0.0), 0.95)
        self._samples: deque[tuple[float, EmotionResult]] = deque(
            maxlen=self.max_samples,
        )

    def update(
        self,
        emotion_result: EmotionResult,
        now: float | None = None,
    ) -> EmotionResult:
        now = time.monotonic() if now is None else now
        self._samples.append((now, emotion_result))
        self._trim_old_samples(now)

        smoothed_scores = self._weighted_average_scores(now)

        if not smoothed_scores:
            return emotion_result

        label, confidence = max(
            smoothed_scores.items(),
            key=lambda item: item[1],
        )

        return EmotionResult(
            label=label,
            confidence=confidence,
            scores=smoothed_scores,
        )

    def _trim_old_samples(self, now: float) -> None:
        while self._samples:
            sample_time, _sample = self._samples[0]

            if now - sample_time <= self.window_seconds:
                break

            self._samples.popleft()

    def _weighted_average_scores(self, now: float) -> dict[str, float]:
        weighted_scores: dict[str, float] = {}
        total_weight = 0.0

        for sample_time, sample in self._samples:
            age = max(0.0, now - sample_time)
            age_ratio = min(age / self.window_seconds, 1.0)
            weight = 1.0 - (age_ratio * self.recency_bias)
            total_weight += weight

            for emotion, score in sample.scores.items():
                weighted_scores[emotion] = (
                    weighted_scores.get(emotion, 0.0)
                    + (score * weight)
                )

        if total_weight <= 0:
            return {}

        return {
            emotion: score / total_weight
            for emotion, score in weighted_scores.items()
        }
