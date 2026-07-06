from __future__ import annotations

import cv2
import numpy as np

from src.domain.emotion_result import EmotionResult
from src.domain.visual_context import VisualContext


class OpenCvRenderer:
    def render(
        self,
        frame: np.ndarray,
        meme: np.ndarray | None,
        emotion_result: EmotionResult,
        stable_meme_key: str,
        visual_context: VisualContext,
    ) -> np.ndarray:
        frame = frame.copy()

        self._draw_text(
            frame,
            f"Detected: {emotion_result.label} ({emotion_result.confidence:.1f}%)",
            y=35,
        )

        self._draw_text(
            frame,
            f"Meme: {stable_meme_key}",
            y=75,
        )

        visual_debug = (
            f"Hands: {visual_context.hands_detected} | "
            f"Mouth open: {visual_context.mouth_open} | "
            f"Thumbs up: {visual_context.thumbs_up}"
        )

        self._draw_text(
            frame,
            visual_debug,
            y=115,
            scale=0.55,
        )

        if meme is None:
            return frame

        meme_resized = self._resize_to_height(meme, frame.shape[0])

        return np.hstack((frame, meme_resized))

    def show(self, window_name: str, frame: np.ndarray) -> bool:
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        return key != ord("q")

    @staticmethod
    def close_all() -> None:
        cv2.destroyAllWindows()

    @staticmethod
    def _draw_text(
        frame: np.ndarray,
        text: str,
        y: int,
        scale: float = 0.8,
    ) -> None:
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
        height, width = image.shape[:2]

        if height == 0:
            raise ValueError("Invalid image height.")

        scale = target_height / height
        target_width = int(width * scale)

        return cv2.resize(image, (target_width, target_height))