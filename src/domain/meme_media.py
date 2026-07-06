from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MemeMedia:
    frames: tuple[np.ndarray, ...]
    durations_ms: tuple[int, ...]

    @classmethod
    def from_frame(cls, frame: np.ndarray) -> MemeMedia:
        return cls(
            frames=(frame,),
            durations_ms=(1000,),
        )

    @property
    def is_animated(self) -> bool:
        return len(self.frames) > 1

    def frame_at(self, elapsed_seconds: float) -> np.ndarray:
        if not self.frames:
            raise ValueError("Meme media has no frames.")

        if len(self.frames) == 1:
            return self.frames[0].copy()

        total_duration_ms = sum(self.durations_ms)

        if total_duration_ms <= 0:
            return self.frames[0].copy()

        elapsed_ms = int(elapsed_seconds * 1000) % total_duration_ms
        accumulated_ms = 0

        for frame, duration_ms in zip(self.frames, self.durations_ms):
            accumulated_ms += duration_ms

            if elapsed_ms < accumulated_ms:
                return frame.copy()

        return self.frames[-1].copy()
