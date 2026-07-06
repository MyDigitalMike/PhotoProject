from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.domain.emotion_result import EmotionResult
from src.infrastructure.http_json_client import HttpJsonClient


class HuggingFaceEmotionAnalyzer:
    API_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{model_id}"

    EMOTION_ALIASES = {
        "anger": "angry",
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happy",
        "happiness": "happy",
        "joy": "happy",
        "neutral": "neutral",
        "sad": "sad",
        "sadness": "sad",
        "surprise": "surprise",
        "surprised": "surprise",
    }

    def __init__(
        self,
        api_token: str,
        model_id: str,
        http_client: HttpJsonClient,
        frame_width: int = 720,
    ) -> None:
        self.api_token = api_token
        self.model_id = model_id
        self.http_client = http_client
        self.frame_width = frame_width

    def analyze(self, frame: np.ndarray) -> EmotionResult:
        image_bytes = self._encode_frame(frame)

        if image_bytes is None:
            return EmotionResult(
                label="neutral",
                confidence=0.0,
                scores={},
            )

        try:
            payload = self.http_client.post_json_bytes(
                self.API_URL_TEMPLATE.format(model_id=self.model_id),
                body=image_bytes,
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "image/jpeg",
                },
            )
        except Exception as error:
            print(f"Hugging Face emotion analysis failed: {error}")
            return EmotionResult(
                label="neutral",
                confidence=0.0,
                scores={},
            )

        scores = self._scores_from_payload(payload)

        if not scores:
            return EmotionResult(
                label="neutral",
                confidence=0.0,
                scores={},
            )

        label, confidence = max(
            scores.items(),
            key=lambda item: item[1],
        )

        return EmotionResult(
            label=label,
            confidence=confidence,
            scores=scores,
        )

    def _encode_frame(self, frame: np.ndarray) -> bytes | None:
        resized_frame = self._resize_for_analysis(frame)
        success, encoded_frame = cv2.imencode(".jpg", resized_frame)

        if not success:
            return None

        return encoded_frame.tobytes()

    def _resize_for_analysis(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]

        if width <= self.frame_width:
            return frame

        scale = self.frame_width / width
        target_height = int(height * scale)

        return cv2.resize(frame, (self.frame_width, target_height))

    @classmethod
    def _scores_from_payload(cls, payload: Any) -> dict[str, float]:
        raw_items = cls._normalize_payload(payload)
        scores: dict[str, float] = {}

        for item in raw_items:
            if not isinstance(item, dict):
                continue

            raw_label = str(item.get("label", "")).lower().strip()
            score = float(item.get("score", 0.0)) * 100.0
            label = cls.EMOTION_ALIASES.get(raw_label)

            if label is None:
                continue

            scores[label] = max(scores.get(label, 0.0), score)

        return scores

    @staticmethod
    def _normalize_payload(payload: Any) -> list[Any]:
        if isinstance(payload, list):
            if payload and isinstance(payload[0], list):
                return list(payload[0])

            return payload

        return []
