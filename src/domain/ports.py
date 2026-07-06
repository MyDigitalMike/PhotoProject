from __future__ import annotations

from typing import Protocol

import numpy as np

from src.domain.emotion_result import EmotionResult
from src.domain.visual_context import VisualContext


class EmotionAnalyzer(Protocol):
    def analyze(self, frame: np.ndarray) -> EmotionResult:
        ...


class VisualSignalAnalyzer(Protocol):
    def analyze(self, frame: np.ndarray) -> VisualContext:
        ...


class MemeRepository(Protocol):
    def get_meme(self, meme_key: str) -> np.ndarray | None:
        ...