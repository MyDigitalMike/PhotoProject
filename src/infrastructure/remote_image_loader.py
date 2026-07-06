from __future__ import annotations

from urllib.request import Request, urlopen

import numpy as np

from src.domain.meme_media import MemeMedia
from src.infrastructure.meme_media_decoder import MemeMediaDecoder


class RemoteImageLoader:
    def __init__(
        self,
        timeout_seconds: float = 7.0,
        media_decoder: MemeMediaDecoder | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.media_decoder = media_decoder or MemeMediaDecoder()

    def load_image(self, image_url: str) -> np.ndarray | None:
        media = self.load_media(image_url)

        if media is None:
            return None

        return media.frame_at(0.0)

    def load_media(self, image_url: str) -> MemeMedia | None:
        request = Request(
            image_url,
            headers={
                "User-Agent": "EmotionMemeApp/1.0",
            },
            method="GET",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                media_bytes = response.read()
        except Exception as error:
            print(f"Remote media download failed: {error}")
            return None

        media = self.media_decoder.decode(media_bytes)

        if media is None:
            print(f"Remote media could not be decoded: {image_url}")
            return None

        if media.is_animated:
            print(
                "Remote animated media loaded: "
                f"frames={len(media.frames)} url={image_url}"
            )

        return media
