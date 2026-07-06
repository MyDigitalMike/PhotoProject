from __future__ import annotations

import numpy as np

from src.domain.emotion_result import EmotionResult
from src.domain.ports import EmotionAnalyzer


class FallbackEmotionAnalyzer:
    def __init__(
        self,
        primary: EmotionAnalyzer,
        fallback: EmotionAnalyzer,
        min_primary_confidence: float = 15.0,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.min_primary_confidence = min_primary_confidence

    def analyze(self, frame: np.ndarray) -> EmotionResult:
        primary_result = self.primary.analyze(frame)

        if primary_result.confidence >= self.min_primary_confidence:
            return primary_result

        fallback_result = self.fallback.analyze(frame)

        if fallback_result.confidence > primary_result.confidence:
            return fallback_result

        return primary_result
