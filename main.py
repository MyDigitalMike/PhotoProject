import os
from pathlib import Path

from src.application.fallback_emotion_analyzer import FallbackEmotionAnalyzer
from src.application.meme_matcher import MemeMatcher
from src.application.meme_profiles import build_default_meme_profiles
from src.infrastructure.api_meme_repository import ApiMemeRepository
from src.infrastructure.camera import Camera
from src.infrastructure.deepface_emotion_analyzer import DeepFaceEmotionAnalyzer
from src.infrastructure.env_file_loader import EnvFileLoader
from src.infrastructure.folder_meme_repository import FolderMemeRepository
from src.infrastructure.giphy_meme_provider import GiphyMemeProvider
from src.infrastructure.http_json_client import HttpJsonClient
from src.infrastructure.huggingface_emotion_analyzer import HuggingFaceEmotionAnalyzer
from src.infrastructure.hybrid_meme_repository import HybridMemeRepository
from src.infrastructure.imgflip_meme_provider import ImgflipMemeProvider
from src.infrastructure.json_meme_profile_catalog import JsonMemeProfileCatalog
from src.infrastructure.meme_api_config import MemeApiConfigLoader
from src.infrastructure.mediapipe_visual_signal_analyzer import MediaPipeVisualSignalAnalyzer
from src.infrastructure.remote_image_loader import RemoteImageLoader
from src.presentation.emotion_meme_app import EmotionMemeApp
from src.presentation.opencv_renderer import OpenCvRenderer


def main() -> None:
    EnvFileLoader(
        env_paths=(
            Path(".env"),
            Path(".env.local"),
        ),
    ).load()

    meme_root = Path("assets") / "memes"
    catalog_path = meme_root / "catalog.json"
    api_config_path = meme_root / "api_providers.json"

    camera = Camera(camera_index=0)
    http_client = HttpJsonClient()

    deepface_emotion_analyzer = DeepFaceEmotionAnalyzer(
        detector_backend="opencv",
        frame_width=720,
    )
    emotion_analyzer = deepface_emotion_analyzer
    huggingface_token = (
        os.environ.get("HUGGINGFACE_API_TOKEN")
        or os.environ.get("HF_TOKEN")
    )
    huggingface_model = os.environ.get("HF_EMOTION_MODEL")

    if huggingface_token and huggingface_model:
        emotion_analyzer = FallbackEmotionAnalyzer(
            primary=deepface_emotion_analyzer,
            fallback=HuggingFaceEmotionAnalyzer(
                api_token=huggingface_token,
                model_id=huggingface_model,
                http_client=http_client,
            ),
            min_primary_confidence=15.0,
        )

    visual_signal_analyzer = MediaPipeVisualSignalAnalyzer()

    meme_profile_catalog = JsonMemeProfileCatalog(catalog_path=catalog_path)
    meme_profiles = (
        meme_profile_catalog.load_profiles()
        or build_default_meme_profiles()
    )
    api_config = MemeApiConfigLoader(config_path=api_config_path).load()
    giphy_api_key = (
        os.environ.get(api_config.giphy.api_key_env)
        if api_config.giphy.enabled
        else None
    )

    _print_api_config(
        api_enabled=api_config.enabled,
        prefer_remote=api_config.prefer_remote,
        log_api_calls=api_config.log_api_calls,
        giphy_enabled=api_config.giphy.enabled,
        giphy_api_key_env=api_config.giphy.api_key_env,
        giphy_api_key=giphy_api_key,
        imgflip_enabled=api_config.imgflip.enabled,
        minimum_remote_display_seconds=api_config.minimum_remote_display_seconds,
        remote_request_cooldown_seconds=api_config.remote_request_cooldown_seconds,
        media_cache_seconds=api_config.media_cache_seconds,
    )

    local_meme_repository = FolderMemeRepository(
        meme_root=meme_root,
        fallback_to_neutral=False,
    )
    meme_providers = []

    if api_config.enabled:
        meme_providers.extend(
            [
                GiphyMemeProvider(
                    api_key=giphy_api_key,
                    http_client=http_client,
                    rating=api_config.giphy.rating,
                    language=api_config.giphy.language,
                    log_api_calls=api_config.log_api_calls,
                ),
                ImgflipMemeProvider(
                    http_client=http_client,
                    enabled=api_config.imgflip.enabled,
                    log_api_calls=api_config.log_api_calls,
                ),
            ]
        )

    api_meme_repository = ApiMemeRepository(
        profiles=meme_profiles,
        providers=meme_providers,
        image_loader=RemoteImageLoader(),
        max_results_per_query=api_config.max_results_per_query,
        log_api_calls=api_config.log_api_calls,
        candidate_cache_seconds=api_config.candidate_cache_seconds,
        media_cache_seconds=api_config.media_cache_seconds,
        minimum_display_seconds=api_config.minimum_remote_display_seconds,
        request_cooldown_seconds=api_config.remote_request_cooldown_seconds,
    )
    meme_repository = HybridMemeRepository(
        local_repository=local_meme_repository,
        api_repository=api_meme_repository,
        prefer_remote=api_config.prefer_remote,
        log_api_calls=api_config.log_api_calls,
    )
    meme_matcher = MemeMatcher(
        profiles=meme_profiles,
        available_keys=meme_repository.get_available_keys(),
    )
    renderer = OpenCvRenderer()

    app = EmotionMemeApp(
        camera=camera,
        emotion_analyzer=emotion_analyzer,
        visual_signal_analyzer=visual_signal_analyzer,
        meme_repository=meme_repository,
        renderer=renderer,
        meme_matcher=meme_matcher,
        analysis_interval_seconds=0.5,
    )

    app.run()


def _print_api_config(
    api_enabled: bool,
    prefer_remote: bool,
    log_api_calls: bool,
    giphy_enabled: bool,
    giphy_api_key_env: str,
    giphy_api_key: str | None,
    imgflip_enabled: bool,
    minimum_remote_display_seconds: float,
    remote_request_cooldown_seconds: float,
    media_cache_seconds: float,
) -> None:
    print(
        "Meme API config: "
        f"enabled={api_enabled}, "
        f"prefer_remote={prefer_remote}, "
        f"log_api_calls={log_api_calls}"
    )
    print(
        "Meme API provider: "
        f"giphy enabled={giphy_enabled}, "
        f"env={giphy_api_key_env}, "
        f"key={_mask_secret(giphy_api_key)}"
    )
    if giphy_enabled and not giphy_api_key:
        print(
            "Meme API provider warning: GIPHY is configured but disabled at "
            f"runtime because {giphy_api_key_env} is missing in this process."
        )
    print(f"Meme API provider: imgflip enabled={imgflip_enabled}")
    print(
        "Meme API pacing: "
        f"minimum_display={minimum_remote_display_seconds:.1f}s, "
        f"request_cooldown={remote_request_cooldown_seconds:.1f}s, "
        f"media_cache={media_cache_seconds:.1f}s"
    )


def _mask_secret(secret: str | None) -> str:
    if not secret:
        return "missing"

    if len(secret) <= 8:
        return "present"

    return f"{secret[:4]}...{secret[-4:]}"


if __name__ == "__main__":
    main()
