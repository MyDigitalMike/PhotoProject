from pathlib import Path

from src.infrastructure.camera import Camera
from src.infrastructure.deepface_emotion_analyzer import DeepFaceEmotionAnalyzer
from src.infrastructure.folder_meme_repository import FolderMemeRepository
from src.infrastructure.mediapipe_visual_signal_analyzer import MediaPipeVisualSignalAnalyzer
from src.presentation.emotion_meme_app import EmotionMemeApp
from src.presentation.opencv_renderer import OpenCvRenderer


def main() -> None:
    meme_root = Path("assets") / "memes"

    camera = Camera(camera_index=0)

    emotion_analyzer = DeepFaceEmotionAnalyzer(
        detector_backend="opencv",
        frame_width=720,
    )

    visual_signal_analyzer = MediaPipeVisualSignalAnalyzer()

    meme_repository = FolderMemeRepository(meme_root=meme_root)
    renderer = OpenCvRenderer()

    app = EmotionMemeApp(
        camera=camera,
        emotion_analyzer=emotion_analyzer,
        visual_signal_analyzer=visual_signal_analyzer,
        meme_repository=meme_repository,
        renderer=renderer,
        analysis_interval_seconds=0.5,
    )

    app.run()


if __name__ == "__main__":
    main()