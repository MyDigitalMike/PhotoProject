from dataclasses import dataclass
@dataclass(frozen=True)
class EmotionResult:
    label: str
    confidence: float
    scores: dict[str, float]