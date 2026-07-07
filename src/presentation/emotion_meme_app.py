from __future__ import annotations

import time

from src.application.emotion_result_smoother import EmotionResultSmoother
from src.application.meme_decision_smoother import MemeDecisionSmoother
from src.application.meme_match_candidate import MemeMatchCandidate
from src.application.meme_matcher import MemeMatcher
from src.application.meme_variety_selector import MemeVarietySelector
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
        emotion_smoothing_window_seconds: float = 1.6,
        meme_candidate_seconds: float = 1.15,
        fast_meme_candidate_seconds: float = 0.60,
        meme_minimum_display_seconds: float = 2.0,
    ) -> None:
        self.camera = camera
        self.emotion_analyzer = emotion_analyzer
        self.visual_signal_analyzer = visual_signal_analyzer
        self.meme_repository = meme_repository
        self.renderer = renderer
        self.analysis_interval_seconds = analysis_interval_seconds
        self.visual_analysis_interval_seconds = visual_analysis_interval_seconds

        self.matcher = meme_matcher or MemeMatcher()
        self.emotion_smoother = EmotionResultSmoother(
            window_seconds=emotion_smoothing_window_seconds,
            max_samples=5,
            recency_bias=0.30,
        )
        self.meme_decision_smoother = MemeDecisionSmoother(
            min_candidate_seconds=meme_candidate_seconds,
            fast_candidate_seconds=fast_meme_candidate_seconds,
            minimum_display_seconds=meme_minimum_display_seconds,
        )
        self.meme_variety_selector = MemeVarietySelector()

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
                        raw_emotion_result = self.emotion_analyzer.analyze(frame)
                        current_emotion_result = self.emotion_smoother.update(
                            raw_emotion_result,
                            now=now,
                        )
                        should_print_debug = True
                    except Exception as error:
                        print(f"Emotion analysis failed: {error}")

                    last_analysis_time = now

                try:
                    ranked_meme_candidates = self.matcher.rank(
                        emotion_result=current_emotion_result,
                        visual_context=current_visual_context,
                    )
                    raw_meme_key = self.meme_variety_selector.select(
                        ranked_meme_candidates,
                        stable_key=stable_meme_key,
                        now=now,
                    )

                    stable_meme_key = self.meme_decision_smoother.update(
                        raw_meme_key,
                        now=now,
                    )
                    self.meme_variety_selector.record_displayed_key(
                        stable_meme_key,
                        now=now,
                    )

                    if should_print_debug:
                        self._print_debug(
                            emotion_result=current_emotion_result,
                            visual_context=current_visual_context,
                            meme_key=stable_meme_key,
                            suggested_meme_key=raw_meme_key,
                            ranked_meme_candidates=ranked_meme_candidates,
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
        suggested_meme_key: str,
        ranked_meme_candidates: tuple[MemeMatchCandidate, ...],
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
            f"mouth_width={visual_context.debug.get('mouth_width_ratio', 0.0):.3f}, "
            f"mouth_wide={visual_context.mouth_wide_open}, "
            f"smile={visual_context.mouth_smile}, "
            f"puckered={visual_context.mouth_puckered}, "
            f"eyes_wide={visual_context.eyes_wide}, "
            f"squint={visual_context.eyes_squint}, "
            f"wink={visual_context.wink}, "
            f"eye_ratio={visual_context.debug.get('eye_open_ratio', 0.0):.3f}, "
            f"brows={visual_context.eyebrows_raised}, "
            f"tilt_l={visual_context.head_tilt_left}, "
            f"tilt_r={visual_context.head_tilt_right}, "
            f"look_l={visual_context.looking_left}, "
            f"look_r={visual_context.looking_right}, "
            f"thumbs_up={visual_context.thumbs_up}, "
            f"thumbs_count={visual_context.debug.get('thumbs_up_count', 0.0):.0f}, "
            f"peace={visual_context.peace_sign}, "
            f"point={visual_context.finger_pointing}, "
            f"fist={visual_context.fist}, "
            f"wave={visual_context.hand_wave}, "
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
            f"({emotion_result.confidence:.2f}%) -> Meme: {meme_key} "
            f"(suggested={suggested_meme_key}) | "
            f"top={EmotionMemeApp._format_candidates(ranked_meme_candidates)} | "
            f"{visual_text} | {scores_text}"
        )

    @staticmethod
    def _format_candidates(
        candidates: tuple[MemeMatchCandidate, ...],
    ) -> str:
        if not candidates:
            return "none"

        return ", ".join(
            f"{candidate.key}:{candidate.score:.0f}"
            for candidate in candidates[:4]
        )
