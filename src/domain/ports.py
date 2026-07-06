from __future__ import annotations

from typing import Protocol

import numpy as np

from src.domain.emotion_result import EmotionResult
from src.domain.meme_candidate import MemeCandidate
from src.domain.meme_media import MemeMedia
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


class MemeProvider(Protocol):
    name: str

    def is_enabled(self) -> bool:
        ...

    def search(
        self,
        meme_key: str,
        query: str,
        limit: int,
    ) -> list[MemeCandidate]:
        ...


class MemeImageLoader(Protocol):
    def load_image(self, image_url: str) -> np.ndarray | None:
        ...


class MemeMediaLoader(Protocol):
    def load_media(self, image_url: str) -> MemeMedia | None:
        ...
