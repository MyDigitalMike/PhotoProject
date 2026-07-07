from __future__ import annotations

from collections.abc import Iterable

from src.application.meme_match_candidate import MemeMatchCandidate
from src.application.meme_profiles import build_default_meme_profiles
from src.domain.emotion_result import EmotionResult
from src.domain.meme_profile import MemeProfile
from src.domain.visual_context import VisualContext


class MemeMatcher:
    """
    Scores meme profiles using DeepFace emotion scores plus MediaPipe signals.

    The matcher intentionally works with generic profiles instead of hard-coded
    folders. That makes it cheap to add reactions such as absolute_cinema,
    wtf_bro, confused, side_eye, etc. as soon as there are images for them.
    """

    SIGNAL_ALIASES = {
        "close_mouth": "mouth_closed",
        "closed_mouth": "mouth_closed",
        "mouth_close": "mouth_closed",
        "smile": "mouth_smile",
        "smiling": "mouth_smile",
        "kiss_lips": "mouth_puckered",
        "puckered_lips": "mouth_puckered",
        "duck_face": "mouth_puckered",
        "kiss_mouth": "mouth_puckered",
        "wide_eyes": "eyes_wide",
        "big_eyes": "eyes_wide",
        "squint": "eyes_squint",
        "squinting": "eyes_squint",
        "closed_eyes": "eyes_closed",
        "eye_closed": "eyes_closed",
        "raised_eyebrows": "eyebrows_raised",
        "brows_up": "eyebrows_raised",
        "look_left": "looking_left",
        "look_right": "looking_right",
        "wide_open_mouth": "mouth_wide_open",
        "big_mouth_open": "mouth_wide_open",
        "hands_visible": "any_hands",
        "hand_visible": "any_hands",
        "any_hand": "any_hands",
        "hand_near_head": "hand_near_temple",
        "hand_face": "hand_near_face",
        "no_hand": "no_hands",
        "single_hand": "one_hand",
        "both_hands": "two_hands",
        "single_open_palm": "one_open_palm",
        "peace": "peace_sign",
        "victory_sign": "peace_sign",
        "pointing": "finger_pointing",
        "point": "finger_pointing",
        "closed_fist": "fist",
        "wave": "hand_wave",
        "waving": "hand_wave",
        "palm_forward": "open_palm",
    }

    def __init__(
        self,
        profiles: Iterable[MemeProfile] | None = None,
        available_keys: Iterable[str] | None = None,
    ) -> None:
        self.profiles = tuple(profiles or build_default_meme_profiles())
        self.available_keys = (
            {key.lower() for key in available_keys}
            if available_keys is not None
            else None
        )

    def match(
        self,
        emotion_result: EmotionResult,
        visual_context: VisualContext,
    ) -> str:
        candidates = self.rank(
            emotion_result=emotion_result,
            visual_context=visual_context,
        )

        if not candidates:
            return emotion_result.label or "neutral"

        return candidates[0].key

    def rank(
        self,
        emotion_result: EmotionResult,
        visual_context: VisualContext,
    ) -> tuple[MemeMatchCandidate, ...]:
        candidates: list[MemeMatchCandidate] = []

        for profile in self.profiles:
            if not self._is_available(profile.key):
                continue

            score = self._score_profile(
                profile=profile,
                emotion_result=emotion_result,
                visual_context=visual_context,
            )

            if score is None:
                continue

            candidates.append(
                MemeMatchCandidate(
                    key=profile.key,
                    score=score,
                )
            )

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)

        return tuple(candidates)

    def _is_available(self, meme_key: str) -> bool:
        if self.available_keys is None:
            return True

        return meme_key.lower() in self.available_keys

    def _score_profile(
        self,
        profile: MemeProfile,
        emotion_result: EmotionResult,
        visual_context: VisualContext,
    ) -> float | None:
        if not self._signals_are_valid(profile, visual_context):
            return None

        if not self._emotion_thresholds_are_valid(profile, emotion_result):
            return None

        score = profile.priority

        for emotion, weight in profile.emotion_weights.items():
            score += emotion_result.scores.get(emotion, 0.0) * weight

        for signal_name, weight in profile.signal_weights.items():
            if self._has_signal(visual_context, signal_name):
                score += weight

        if score < profile.min_total_score:
            return None

        return score

    def _signals_are_valid(
        self,
        profile: MemeProfile,
        visual_context: VisualContext,
    ) -> bool:
        for signal_name in profile.required_signals:
            if not self._has_signal(visual_context, signal_name):
                return False

        if profile.any_signals and not any(
            self._has_signal(visual_context, signal_name)
            for signal_name in profile.any_signals
        ):
            return False

        for signal_name in profile.blocked_signals:
            if self._has_signal(visual_context, signal_name):
                return False

        return True

    @staticmethod
    def _emotion_thresholds_are_valid(
        profile: MemeProfile,
        emotion_result: EmotionResult,
    ) -> bool:
        scores = emotion_result.scores

        for emotion, minimum_score in profile.min_scores.items():
            if scores.get(emotion, 0.0) < minimum_score:
                return False

        if profile.any_min_scores and not any(
            scores.get(emotion, 0.0) >= minimum_score
            for emotion, minimum_score in profile.any_min_scores.items()
        ):
            return False

        return True

    @staticmethod
    def _has_signal(
        visual_context: VisualContext,
        signal_name: str,
    ) -> bool:
        if signal_name == "head_tilt":
            return (
                visual_context.head_tilt_left
                or visual_context.head_tilt_right
            )

        if signal_name == "looking_side":
            return visual_context.looking_left or visual_context.looking_right

        resolved_signal_name = MemeMatcher.SIGNAL_ALIASES.get(
            signal_name,
            signal_name,
        )

        return bool(getattr(visual_context, resolved_signal_name, False))
