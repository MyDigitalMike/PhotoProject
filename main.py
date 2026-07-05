from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class EmotionResult:
    label: str
    confidence: float


class DeepFaceEmotionAnalyzer:
    def __init__(self, minimum_confidence: float = 35.0) -> None:
        self.minimum_confidence = minimum_confidence

    def analyze(self, frame: np.ndarray) -> EmotionResult:
        result = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )

        if isinstance(result, list):
            result = result[0]

        label = str(result["dominant_emotion"]).lower()
        scores = result["emotion"]
        confidence = float(scores.get(label, 0.0))

        if confidence < self.minimum_confidence:
            return EmotionResult(label="neutral", confidence=confidence)

        return EmotionResult(label=label, confidence=confidence)


class EmotionSmoother:
    """
    Prevents the meme from changing too fast because the model can flicker
    between emotions like neutral, happy, and surprise.
    """

    def __init__(self, required_repeats: int = 2) -> None:
        self.required_repeats = required_repeats
        self.candidate_emotion: str | None = None
        self.candidate_count = 0
        self.stable_emotion = "neutral"

    def update(self, emotion_result: EmotionResult) -> str:
        if emotion_result.label == self.candidate_emotion:
            self.candidate_count += 1
        else:
            self.candidate_emotion = emotion_result.label
            self.candidate_count = 1

        if self.candidate_count >= self.required_repeats:
            self.stable_emotion = emotion_result.label

        return self.stable_emotion


class MemeSelector:
    def __init__(self, meme_root: Path) -> None:
        self.meme_root = meme_root
        self.memes_by_emotion = self._load_meme_paths()

        self.current_emotion: str | None = None
        self.current_meme: np.ndarray | None = None

    def _load_meme_paths(self) -> dict[str, list[Path]]:
        if not self.meme_root.exists():
            raise FileNotFoundError(f"Meme folder not found: {self.meme_root}")

        memes_by_emotion: dict[str, list[Path]] = {}

        for folder in self.meme_root.iterdir():
            if not folder.is_dir():
                continue

            emotion = folder.name.lower()

            image_paths = [
                path
                for path in folder.iterdir()
                if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ]

            memes_by_emotion[emotion] = image_paths

        return memes_by_emotion

    def select(self, emotion: str) -> np.ndarray | None:
        image_paths = self.memes_by_emotion.get(emotion)

        if not image_paths:
            image_paths = self.memes_by_emotion.get("neutral", [])

        if not image_paths:
            return None

        emotion_changed = emotion != self.current_emotion

        if emotion_changed or self.current_meme is None:
            selected_path = random.choice(image_paths)
            image = cv2.imread(str(selected_path))

            if image is None:
                print(f"Could not load image: {selected_path}")
                return None

            self.current_emotion = emotion
            self.current_meme = image

            print(f"Meme changed to: {emotion} -> {selected_path.name}")

        return self.current_meme.copy()


class EmotionMemeApp:
    def __init__(
        self,
        analyzer: DeepFaceEmotionAnalyzer,
        meme_selector: MemeSelector,
        camera_index: int = 0,
        analysis_interval_seconds: float = 0.4,
    ) -> None:
        self.analyzer = analyzer
        self.meme_selector = meme_selector
        self.camera_index = camera_index
        self.analysis_interval_seconds = analysis_interval_seconds
        self.smoother = EmotionSmoother(required_repeats=1)

    def run(self) -> None:
        camera = cv2.VideoCapture(self.camera_index)

        if not camera.isOpened():
            raise RuntimeError("Could not open the camera. Try camera_index=1.")

        current_emotion = "neutral"
        current_confidence = 0.0
        last_analysis_time = 0.0

        print("Emotion Meme App is running. Press Q to quit.")

        while True:
            success, frame = camera.read()

            if not success:
                raise RuntimeError("Could not read a frame from the camera.")

            now = time.time()
            should_analyze = now - last_analysis_time >= self.analysis_interval_seconds

            if should_analyze:
                try:
                    emotion_result = self.analyzer.analyze(frame)
                    current_emotion = self.smoother.update(emotion_result)
                    current_confidence = emotion_result.confidence

                    print(
                        f"Detected: {emotion_result.label} "
                        f"({current_confidence:.1f}%) | "
                        f"Stable: {current_emotion}"
                    )

                except Exception as error:
                    print(f"Emotion analysis failed: {error}")
                    current_emotion = "neutral"
                    current_confidence = 0.0

                last_analysis_time = now

            meme = self.meme_selector.select(current_emotion)
            output = self._create_output_frame(
                frame=frame,
                meme=meme,
                emotion=current_emotion,
                confidence=current_confidence,
            )

            cv2.imshow("Emotion Meme App - Press Q to quit", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        camera.release()
        cv2.destroyAllWindows()

    def _create_output_frame(
        self,
        frame: np.ndarray,
        meme: np.ndarray | None,
        emotion: str,
        confidence: float,
    ) -> np.ndarray:
        frame = frame.copy()

        text = f"Expression: {emotion} ({confidence:.1f}%)"

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if meme is None:
            return frame

        meme_resized = self._resize_to_height(meme, frame.shape[0])

        return np.hstack((frame, meme_resized))

    @staticmethod
    def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
        height, width = image.shape[:2]

        if height == 0:
            raise ValueError("Invalid image height.")

        scale = target_height / height
        target_width = int(width * scale)

        return cv2.resize(image, (target_width, target_height))


def main() -> None:
    meme_root = Path("assets") / "memes"

    analyzer = DeepFaceEmotionAnalyzer(minimum_confidence=35.0)
    meme_selector = MemeSelector(meme_root=meme_root)

    app = EmotionMemeApp(
        analyzer=analyzer,
        meme_selector=meme_selector,
        camera_index=0,
        analysis_interval_seconds=0.75,
    )

    app.run()


if __name__ == "__main__":
    main()