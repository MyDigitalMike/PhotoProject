from __future__ import annotations

import time

from src.application.emotion_smoother import EmotionSmoother
from src.application.meme_matcher import MemeMatcher
from src.domain.emotion_result import EmotionResult
from src.domain.ports import EmotionAnalyzer, MemeRepository, VisualSignalAnalyzer
from src.domain.visual_context import VisualContext
from src.infrastructure.camera import Camera
from src.presentation.opencv_renderer import OpenCvRenderer


class EmotionMemeApp:
    def __init__(
        self,
        camera: Camera,
        emotion_analyzer: EmotionAnalyzer,
        visual_signal_analyzer: VisualSignalAnalyzer,
        meme_repository: MemeRepository,
        renderer: OpenCvRenderer,
        analysis_interval_seconds: float = 0.5,
    ) -> None:
        self.camera = camera
        self.emotion_analyzer = emotion_analyzer
        self.visual_signal_analyzer = visual_signal_analyzer
        self.meme_repository = meme_repository
        self.renderer = renderer
        self.analysis_interval_seconds = analysis_interval_seconds

        self.matcher = MemeMatcher()
        self.smoother = EmotionSmoother(required_repeats=1)

    def run(self) -> None:
        last_analysis_time = 0.0

        current_emotion_result = EmotionResult(
            label="neutral",
            confidence=0.0,
            scores={},
        )

        current_visual_context = VisualContext()
        stable_meme_key = "neutral"

        print("Emotion Meme App is running. Press Q to quit.")

        try:
            while True:
                frame = self.camera.read()
                now = time.time()

                if now - last_analysis_time >= self.analysis_interval_seconds:
                    try:
                        current_emotion_result = self.emotion_analyzer.analyze(frame)
                        current_visual_context = self.visual_signal_analyzer.analyze(frame)

                        raw_meme_key = self.matcher.match(
                            emotion_result=current_emotion_result,
                            visual_context=current_visual_context,
                        )

                        stable_meme_key = self.smoother.update(
                            EmotionResult(
                                label=raw_meme_key,
                                confidence=current_emotion_result.confidence,
                                scores=current_emotion_result.scores,
                            )
                        )

                        self._print_debug(
                            emotion_result=current_emotion_result,
                            visual_context=current_visual_context,
                            meme_key=stable_meme_key,
                        )

                    except Exception as error:
                        print(f"Analysis failed: {error}")
                        stable_meme_key = "neutral"

                    last_analysis_time = now

                meme = self.meme_repository.get_meme(stable_meme_key)

                output = self.renderer.render(
                    frame=frame,
                    meme=meme,
                    emotion_result=current_emotion_result,
                    stable_meme_key=stable_meme_key,
                    visual_context=current_visual_context,
                )

                should_continue = self.renderer.show(
                    "Emotion Meme App - Press Q to quit",
                    output,
                )

                if not should_continue:
                    break

        finally:
            self.camera.release()
            self.renderer.close_all()

    @staticmethod
    def _print_debug(
        emotion_result: EmotionResult,
        visual_context: VisualContext,
        meme_key: str,
    ) -> None:
        scores_text = " | ".join(
            f"{emotion}: {score:.2f}%"
            for emotion, score in sorted(
                emotion_result.scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        visual_text = (
            f"hands={visual_context.hands_detected}, "
            f"mouth_open={visual_context.mouth_open}, "
            f"thumbs_up={visual_context.thumbs_up}, "
            f"forehead={visual_context.hand_near_forehead}, "
            f"chin={visual_context.hand_near_chin}, "
            f"mouth={visual_context.hand_near_mouth}, "
            f"cheeks={visual_context.hands_near_cheeks}"
        )

        print(
            f"Detected: {emotion_result.label} "
            f"({emotion_result.confidence:.2f}%) -> Meme: {meme_key} | "
            f"{visual_text} | {scores_text}"
        )