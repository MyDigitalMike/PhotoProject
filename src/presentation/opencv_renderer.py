from __future__ import annotations

import time

import cv2
import numpy as np

from src.domain.emotion_result import EmotionResult
from src.domain.visual_context import VisualContext


class OpenCvRenderer:
    def __init__(self, transition_seconds: float = 0.25) -> None:
        self.transition_seconds = transition_seconds
        self._displayed_meme_key: str | None = None
        self._displayed_meme: np.ndarray | None = None
        self._transition_from: np.ndarray | None = None
        self._transition_started_at = 0.0

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
            f"Eyes wide: {visual_context.eyes_wide} | "
            f"Peace: {visual_context.peace_sign} | "
            f"Point: {visual_context.finger_pointing} | "
            f"2 palms: {visual_context.two_open_palms} | "
            f"Wave: {visual_context.hand_wave}"
        )

        self._draw_text(
            frame,
            visual_debug,
            y=115,
            scale=0.55,
        )

        if meme is None:
            return frame

        meme_panel = self._resize_to_fit(
            meme,
            target_width=frame.shape[1],
            target_height=frame.shape[0],
        )
        meme_panel = self._apply_meme_transition(
            meme_key=stable_meme_key,
            meme_panel=meme_panel,
        )

        return np.hstack((frame, meme_panel))

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

    def _apply_meme_transition(
        self,
        meme_key: str,
        meme_panel: np.ndarray,
    ) -> np.ndarray:
        if self._displayed_meme is None or self._displayed_meme_key is None:
            self._displayed_meme_key = meme_key
            self._displayed_meme = meme_panel

            return meme_panel

        if meme_key != self._displayed_meme_key:
            self._transition_from = self._displayed_meme.copy()
            self._transition_started_at = time.monotonic()
            self._displayed_meme_key = meme_key

        if self._transition_from is None:
            self._displayed_meme = meme_panel

            return meme_panel

        if self.transition_seconds <= 0:
            progress = 1.0
        else:
            progress = min(
                (time.monotonic() - self._transition_started_at)
                / self.transition_seconds,
                1.0,
            )

        eased_progress = self._ease_in_out(progress)
        blended_panel = cv2.addWeighted(
            self._transition_from,
            1.0 - eased_progress,
            meme_panel,
            eased_progress,
            0,
        )

        if progress >= 1.0:
            self._transition_from = None
            self._displayed_meme = meme_panel
        else:
            self._displayed_meme = blended_panel

        return self._displayed_meme

    @staticmethod
    def _resize_to_fit(
        image: np.ndarray,
        target_width: int,
        target_height: int,
    ) -> np.ndarray:
        image = OpenCvRenderer._ensure_bgr(image)
        height, width = image.shape[:2]

        if height == 0 or width == 0:
            raise ValueError("Invalid image size.")

        scale = min(target_width / width, target_height / height)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))

        resized = cv2.resize(image, (resized_width, resized_height))

        panel = np.zeros(
            (target_height, target_width, 3),
            dtype=resized.dtype,
        )

        x = (target_width - resized_width) // 2
        y = (target_height - resized_height) // 2

        panel[y : y + resized_height, x : x + resized_width] = resized

        return panel

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        return image

    @staticmethod
    def _ease_in_out(progress: float) -> float:
        progress = min(max(progress, 0.0), 1.0)

        return progress * progress * (3.0 - 2.0 * progress)
