from src.domain.emotion_result import EmotionResult
class EmotionSmoother:
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