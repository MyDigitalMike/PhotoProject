from __future__ import annotations

from src.domain.emotion_result import EmotionResult
from src.domain.visual_context import VisualContext


class MemeMatcher:
    """
    Combines DeepFace emotion scores with MediaPipe face/hand signals.

    DeepFace:
    - good enough for basic happy / sad / neutral / surprise
    - less reliable for disgust and sometimes fear

    MediaPipe:
    - useful for meme poses
    - hand near face, mouth open, facepalm, thinking pose, etc.
    """

    def match(
        self,
        emotion_result: EmotionResult,
        visual_context: VisualContext,
    ) -> str:
        scores = emotion_result.scores

        happy = scores.get("happy", 0.0)
        sad = scores.get("sad", 0.0)
        angry = scores.get("angry", 0.0)
        surprise = scores.get("surprise", 0.0)
        fear = scores.get("fear", 0.0)
        disgust = scores.get("disgust", 0.0)
        neutral = scores.get("neutral", 0.0)

        # 1. High-priority hand/meme poses.
        # These should win over basic emotion labels.
        if visual_context.thumbs_up:
            return "thumbs_up"

        if visual_context.hand_near_forehead:
            return "facepalm"

        if visual_context.hand_near_chin and not visual_context.hand_near_mouth:
            return "thinking"

        if visual_context.hands_near_cheeks and (
            visual_context.mouth_open or surprise >= 20.0 or fear >= 20.0 or sad >= 60
        ):
            return "shocked"

        if visual_context.hand_near_mouth and (
            not visual_context.mouth_open or surprise >= 15.0 or fear >= 15.0 or sad >= 60
        ):
            return "hand_on_mouth"

        # 2. Strong happy rule.
        # Happy is usually easier to detect, but neutral can still dominate.
        if happy >= 35.0:
            return "happy"

        if happy >= 20.0 and neutral <= 80.0:
            return "happy"

        # 3. Better fear/shocked behavior.
        # In practice, fear is often confused with surprise or neutral.
        if fear >= 25.0:
            return "fear"

        if surprise >= 25.0 and visual_context.mouth_open:
            return "fear"

        if surprise >= 35.0:
            return "shocked"

        # 4. Sad rule.
        # Sad can be subtle, so use lower thresholds than angry.
        if sad >= 25.0:
            return "sad"

        if sad >= 12.0 and happy <= 5.0 and neutral <= 85.0:
            return "sad"

        # 5. Disgust rule.
        # Your logs show disgust stays at 0.0, so do not depend on it.
        # Use it only if it appears clearly.
        if disgust >= 8.0:
            return "disgust"

        # 6. Angry rule.
        # Keep angry stricter because you said it feels weird.
        if angry >= 45.0:
            return "angry"

        if angry >= 25.0 and neutral <= 65.0 and happy <= 5.0:
            return "angry"

        # 7. Open palm can be a neutral/stop meme.
        if visual_context.open_palm:
            return "open_palm"

        # 8. Fallback.
        if neutral >= 50.0:
            return "neutral"

        return emotion_result.label or "neutral"