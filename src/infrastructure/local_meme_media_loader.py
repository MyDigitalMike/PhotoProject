from __future__ import annotations

from pathlib import Path

from src.domain.meme_media import MemeMedia
from src.infrastructure.meme_media_decoder import MemeMediaDecoder


class LocalMemeMediaLoader:
    def __init__(
        self,
        media_decoder: MemeMediaDecoder | None = None,
    ) -> None:
        self.media_decoder = media_decoder or MemeMediaDecoder()

    def load_media(self, media_path: Path) -> MemeMedia | None:
        try:
            media_bytes = media_path.read_bytes()
        except Exception as error:
            print(f"Could not read media file: {media_path} ({error})")
            return None

        media = self.media_decoder.decode(media_bytes)

        if media is None:
            print(f"Could not decode media file: {media_path}")
            return None

        if media.is_animated:
            print(
                "Local animated media loaded: "
                f"frames={len(media.frames)} path={media_path.name}"
            )

        return media
