from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np

from src.domain.meme_media import MemeMedia

try:
    from PIL import Image, ImageSequence
except ImportError:
    Image = None
    ImageSequence = None


class MemeMediaDecoder:
    DEFAULT_FRAME_DURATION_MS = 100
    MAX_ANIMATION_FRAMES = 120

    def decode(self, media_bytes: bytes) -> MemeMedia | None:
        animated_media = self._decode_animation_with_pillow(media_bytes)

        if animated_media is not None:
            return animated_media

        static_frame = self._decode_static_with_opencv(media_bytes)

        if static_frame is not None:
            return MemeMedia.from_frame(static_frame)

        static_frame = self._decode_static_with_pillow(media_bytes)

        if static_frame is not None:
            return MemeMedia.from_frame(static_frame)

        return None

    @staticmethod
    def _decode_static_with_opencv(media_bytes: bytes) -> np.ndarray | None:
        image_array = np.frombuffer(media_bytes, dtype=np.uint8)

        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    def _decode_animation_with_pillow(
        self,
        media_bytes: bytes,
    ) -> MemeMedia | None:
        if Image is None or ImageSequence is None:
            return None

        try:
            image = Image.open(BytesIO(media_bytes))
        except Exception:
            return None

        if not getattr(image, "is_animated", False):
            return None

        frames: list[np.ndarray] = []
        durations_ms: list[int] = []

        try:
            for frame_index, frame in enumerate(ImageSequence.Iterator(image)):
                if frame_index >= self.MAX_ANIMATION_FRAMES:
                    break

                frames.append(self._pil_frame_to_bgr(frame))
                durations_ms.append(
                    self._frame_duration_ms(frame.info.get("duration"))
                )
        except Exception:
            return None

        if not frames:
            return None

        return MemeMedia(
            frames=tuple(frames),
            durations_ms=tuple(durations_ms),
        )

    @staticmethod
    def _decode_static_with_pillow(media_bytes: bytes) -> np.ndarray | None:
        if Image is None:
            return None

        try:
            image = Image.open(BytesIO(media_bytes))
            return MemeMediaDecoder._pil_frame_to_bgr(image)
        except Exception:
            return None

    @staticmethod
    def _pil_frame_to_bgr(frame: Image.Image) -> np.ndarray:
        rgb_frame = frame.convert("RGB")

        return cv2.cvtColor(np.array(rgb_frame), cv2.COLOR_RGB2BGR)

    def _frame_duration_ms(self, raw_duration: object) -> int:
        if not isinstance(raw_duration, int | float):
            return self.DEFAULT_FRAME_DURATION_MS

        duration_ms = int(raw_duration)

        if duration_ms <= 0:
            return self.DEFAULT_FRAME_DURATION_MS

        return duration_ms
