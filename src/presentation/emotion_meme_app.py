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
        meme_matcher: MemeMatcher | None = None,
        analysis_interval_seconds: float = 0.5,
        visual_analysis_interval_seconds: float = 0.10,
    ) -> None:
        self.camera = camera
        self.emotion_analyzer = emotion_analyzer
        self.visual_signal_analyzer = visual_signal_analyzer
        self.meme_repository = meme_repository
        self.renderer = renderer
        self.analysis_interval_seconds = analysis_interval_seconds
        self.visual_analysis_interval_seconds = visual_analysis_interval_seconds

        self.matcher = meme_matcher or MemeMatcher()
        self.smoother = EmotionSmoother(
            required_repeats=2,
            min_stable_seconds=0.45,
        )

    def run(self) -> None:
        last_analysis_time = 0.0
        last_visual_analysis_time = 0.0

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
                now = time.monotonic()
                should_print_debug = False

                if (
                    now - last_visual_analysis_time
                    >= self.visual_analysis_interval_seconds
                ):
                    try:
                        current_visual_context = self.visual_signal_analyzer.analyze(frame)
                    except Exception as error:
                        print(f"Visual analysis failed: {error}")

                    last_visual_analysis_time = now

                if now - last_analysis_time >= self.analysis_interval_seconds:
                    try:
                        current_emotion_result = self.emotion_analyzer.analyze(frame)
                        should_print_debug = True
                    except Exception as error:
                        print(f"Emotion analysis failed: {error}")

                    last_analysis_time = now

                try:
                    raw_meme_key = self.matcher.match(
                        emotion_result=current_emotion_result,
                        visual_context=current_visual_context,
                    )

                    stable_meme_key = self.smoother.update(
                        EmotionResult(
                            label=raw_meme_key,
                            confidence=current_emotion_result.confidence,
                            scores=current_emotion_result.scores,
                        ),
                        now=now,
                    )

                    if should_print_debug:
                        self._print_debug(
                            emotion_result=current_emotion_result,
                            visual_context=current_visual_context,
                            meme_key=stable_meme_key,
                        )

                except Exception as error:
                    print(f"Meme matching failed: {error}")

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
            f"mouth_closed={visual_context.mouth_closed}, "
            f"mouth_ratio={visual_context.debug.get('mouth_open_ratio', 0.0):.3f}, "
            f"mouth_wide={visual_context.mouth_wide_open}, "
            f"thumbs_up={visual_context.thumbs_up}, "
            f"thumbs_count={visual_context.debug.get('thumbs_up_count', 0.0):.0f}, "
            f"any_hands={visual_context.any_hands}, "
            f"two_palms={visual_context.two_open_palms}, "
            f"palms_count={visual_context.debug.get('open_palm_count', 0.0):.0f}, "
            f"near_face={visual_context.hand_near_face}, "
            f"forehead={visual_context.hand_near_forehead}, "
            f"chin={visual_context.hand_near_chin}, "
            f"temple={visual_context.hand_near_temple}, "
            f"mouth={visual_context.hand_near_mouth}, "
            f"cheeks={visual_context.hands_near_cheeks}"
        )

        print(
            f"Detected: {emotion_result.label} "
            f"({emotion_result.confidence:.2f}%) -> Meme: {meme_key} | "
            f"{visual_text} | {scores_text}"
        )
