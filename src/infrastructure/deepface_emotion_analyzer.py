from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import cv2
import numpy as np
from deepface import DeepFace

from src.domain.emotion_result import EmotionResult


class DeepFaceEmotionAnalyzer:
    def __init__(
        self,
        detector_backend: str = "opencv",
        frame_width: int = 720,
    ) -> None:
        self.detector_backend = detector_backend
        self.frame_width = frame_width

    def analyze(self, frame: np.ndarray) -> EmotionResult:
        resized_frame = self._resize_for_analysis(frame)

        analysis_result: Any = DeepFace.analyze(
            img_path=resized_frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend=self.detector_backend,
            align=True,
            silent=True,
        )

        result = self._normalize_deepface_result(analysis_result)

        label = str(result.get("dominant_emotion", "neutral")).lower()

        raw_scores = result.get("emotion", {})

        if not isinstance(raw_scores, Mapping):
            raw_scores = {}

        scores = {
            str(emotion).lower(): float(score)
            for emotion, score in raw_scores.items()
        }

        confidence = float(scores.get(label, 0.0))

        return EmotionResult(
            label=label,
            confidence=confidence,
            scores=scores,
        )

    def _resize_for_analysis(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]

        if width <= self.frame_width:
            return frame

        scale = self.frame_width / width
        target_height = int(height * scale)

        return cv2.resize(frame, (self.frame_width, target_height))

    @staticmethod
    def _normalize_deepface_result(analysis_result: Any) -> Mapping[str, Any]:
        """
        DeepFace can return:
        - dict
        - list[dict]

        Pylance does not know this reliably, so we normalize the result here.
        """

        if isinstance(analysis_result, list):
            if len(analysis_result) == 0:
                return {}

            first_result = analysis_result[0]

            if isinstance(first_result, Mapping):
                return cast(Mapping[str, Any], first_result)

            return {}

        if isinstance(analysis_result, Mapping):
            return cast(Mapping[str, Any], analysis_result)

        return {}