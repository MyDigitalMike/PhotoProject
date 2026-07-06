from __future__ import annotations

from src.domain.meme_profile import MemeProfile


def build_default_meme_profiles() -> tuple[MemeProfile, ...]:
    return (
        MemeProfile(
            key="thumbs_up",
            priority=1000.0,
            min_total_score=1000.0,
            required_signals=("thumbs_up",),
            search_terms=("thumbs up reaction meme", "approval meme"),
            description="Thumbs-up pose.",
        ),
        MemeProfile(
            key="facepalm",
            priority=950.0,
            min_total_score=950.0,
            required_signals=("hand_near_forehead",),
            search_terms=("facepalm reaction meme", "face palm meme"),
            description="Hand near forehead.",
        ),
        MemeProfile(
            key="thinking",
            priority=900.0,
            min_total_score=900.0,
            required_signals=("hand_near_chin",),
            blocked_signals=("hand_near_mouth",),
            search_terms=("thinking reaction meme", "thinking face meme"),
            description="Hand on chin.",
        ),
        MemeProfile(
            key="shocked",
            priority=520.0,
            min_total_score=580.0,
            required_signals=("hands_near_cheeks",),
            emotion_weights={
                "surprise": 3.0,
                "fear": 3.0,
                "sad": 1.0,
            },
            signal_weights={
                "mouth_open": 100.0,
            },
            search_terms=("shocked reaction meme", "surprised shocked meme"),
            description="Both hands near cheeks with shock/fear/open mouth.",
        ),
        MemeProfile(
            key="hand_on_mouth",
            priority=850.0,
            min_total_score=850.0,
            required_signals=("hand_near_mouth",),
            blocked_signals=("mouth_open",),
            search_terms=("hand on mouth reaction meme", "gasp reaction meme"),
            description="Covered mouth.",
        ),
        MemeProfile(
            key="hand_on_mouth",
            priority=620.0,
            min_total_score=665.0,
            required_signals=("hand_near_mouth",),
            emotion_weights={
                "surprise": 3.0,
                "fear": 3.0,
                "sad": 1.0,
            },
            search_terms=("hand on mouth reaction meme", "gasp reaction meme"),
            description="Hand near mouth with surprise/fear/sadness.",
        ),
        MemeProfile(
            key="wtf_bro",
            priority=610.0,
            min_total_score=660.0,
            required_signals=("two_open_palms",),
            emotion_weights={
                "surprise": 2.5,
                "angry": 1.5,
                "disgust": 4.0,
                "fear": 1.0,
            },
            signal_weights={
                "mouth_open": 70.0,
            },
            search_terms=("wtf reaction meme", "what bro reaction meme"),
            description="Two-palms questioning pose.",
        ),
        MemeProfile(
            key="confused",
            priority=540.0,
            min_total_score=590.0,
            required_signals=("hand_near_temple",),
            emotion_weights={
                "surprise": 2.0,
                "neutral": 0.5,
                "sad": 0.8,
            },
            search_terms=("confused reaction meme", "confused meme"),
            description="Hand near temple/head-scratch style pose.",
        ),
        MemeProfile(
            key="absolute_cinema",
            priority=210.0,
            min_total_score=290.0,
            any_signals=("open_palm", "two_open_palms"),
            emotion_weights={
                "happy": 1.2,
                "neutral": 0.8,
                "surprise": 0.4,
            },
            signal_weights={
                "open_palm": 85.0,
                "two_open_palms": 120.0,
            },
            search_terms=("absolute cinema meme", "cinema reaction meme"),
            description="Admiring/open-hand reaction.",
        ),
        MemeProfile(
            key="smiling",
            priority=760.0,
            min_total_score=785.0,
            required_signals=("mouth_smile", "mouth_closed"),
            blocked_signals=("mouth_puckered",),
            min_scores={
                "happy": 20.0,
            },
            emotion_weights={
                "happy": 1.4,
            },
            search_terms=("smiling reaction gif", "happy smile reaction meme"),
            description="Clear closed-mouth smile.",
        ),
        MemeProfile(
            key="kiss",
            priority=780.0,
            min_total_score=780.0,
            required_signals=("mouth_puckered",),
            blocked_signals=("mouth_smile",),
            emotion_weights={
                "happy": 0.4,
                "neutral": 0.2,
            },
            search_terms=("blowing kiss reaction gif", "mwah reaction gif"),
            description="Puckered lips / kiss-face expression.",
        ),
        MemeProfile(
            key="happy",
            priority=0.0,
            min_total_score=40.0,
            blocked_signals=("mouth_puckered",),
            emotion_weights={
                "happy": 2.0,
            },
            signal_weights={
                "mouth_smile": 30.0,
            },
            search_terms=("happy reaction meme", "laughing reaction meme"),
            description="Happy emotion.",
        ),
        MemeProfile(
            key="fear",
            priority=0.0,
            min_total_score=50.0,
            emotion_weights={
                "fear": 2.0,
            },
            signal_weights={
                "mouth_open": 15.0,
            },
            search_terms=("scared reaction meme", "fear reaction meme"),
            description="Fear emotion.",
        ),
        MemeProfile(
            key="fear",
            priority=0.0,
            min_total_score=65.0,
            min_scores={
                "surprise": 25.0,
            },
            signal_weights={
                "mouth_open": 65.0,
            },
            search_terms=("scared reaction meme", "surprised scared meme"),
            description="Surprise with open mouth, mapped to fear.",
        ),
        MemeProfile(
            key="shocked",
            priority=0.0,
            min_total_score=70.0,
            emotion_weights={
                "surprise": 2.0,
            },
            search_terms=("shocked reaction meme", "surprised reaction meme"),
            description="Strong surprise.",
        ),
        MemeProfile(
            key="surprise",
            priority=0.0,
            min_total_score=45.0,
            emotion_weights={
                "surprise": 1.5,
            },
            signal_weights={
                "mouth_open": 20.0,
            },
            search_terms=("surprised reaction meme", "surprise meme"),
            description="Moderate surprise.",
        ),
        MemeProfile(
            key="sad",
            priority=0.0,
            min_total_score=35.0,
            emotion_weights={
                "sad": 2.0,
            },
            search_terms=("sad reaction meme", "crying reaction meme"),
            description="Sad emotion.",
        ),
        MemeProfile(
            key="disgust",
            priority=0.0,
            min_total_score=24.0,
            emotion_weights={
                "disgust": 3.0,
            },
            search_terms=("disgust reaction meme", "gross reaction meme"),
            description="Disgust emotion.",
        ),
        MemeProfile(
            key="angry",
            priority=0.0,
            min_total_score=80.0,
            emotion_weights={
                "angry": 2.0,
            },
            search_terms=("angry reaction meme", "mad reaction meme"),
            description="Angry emotion.",
        ),
        MemeProfile(
            key="open_palm",
            priority=100.0,
            min_total_score=100.0,
            required_signals=("open_palm",),
            search_terms=("stop reaction meme", "open palm reaction meme"),
            description="Open palm/stop gesture.",
        ),
        MemeProfile(
            key="neutral",
            priority=0.0,
            min_total_score=30.0,
            emotion_weights={
                "neutral": 1.0,
            },
            search_terms=("neutral reaction meme", "blank stare meme"),
            description="Neutral fallback.",
        ),
    )
