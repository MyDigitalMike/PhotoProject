from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np

from src.domain.meme_media import MemeMedia
from src.infrastructure.local_meme_media_loader import LocalMemeMediaLoader


class FolderMemeRepository:
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

    def __init__(
        self,
        meme_root: Path,
        fallback_to_neutral: bool = True,
        media_loader: LocalMemeMediaLoader | None = None,
    ) -> None:
        self.meme_root = meme_root
        self.fallback_to_neutral = fallback_to_neutral
        self.media_loader = media_loader or LocalMemeMediaLoader()
        self.memes_by_key = self._load_meme_paths()

        self.current_key: str | None = None
        self.current_media: MemeMedia | None = None
        self.current_media_started_at = 0.0

    def _load_meme_paths(self) -> dict[str, list[Path]]:
        if not self.meme_root.exists():
            raise FileNotFoundError(f"Meme folder not found: {self.meme_root}")

        memes_by_key: dict[str, list[Path]] = {}

        for folder in self.meme_root.iterdir():
            if not folder.is_dir():
                continue

            meme_key = folder.name.lower()

            image_paths = [
                path
                for path in folder.iterdir()
                if path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]

            memes_by_key[meme_key] = image_paths

        return memes_by_key

    def get_available_keys(self) -> set[str]:
        return {
            meme_key
            for meme_key, image_paths in self.memes_by_key.items()
            if image_paths
        }

    def get_meme(self, meme_key: str) -> np.ndarray | None:
        image_paths = self.memes_by_key.get(meme_key)

        if not image_paths and self.fallback_to_neutral:
            image_paths = self.memes_by_key.get("neutral", [])

        if not image_paths:
            return None

        should_change = meme_key != self.current_key or self.current_media is None

        if should_change:
            selected_path = random.choice(image_paths)
            media = self.media_loader.load_media(selected_path)

            if media is None:
                return None

            self.current_key = meme_key
            self.current_media = media
            self.current_media_started_at = time.monotonic()

            print(
                f"Meme changed: {meme_key} -> {selected_path.name} "
                f"animated={media.is_animated}"
            )

        if self.current_media is None:
            return None

        return self.current_media.frame_at(
            time.monotonic() - self.current_media_started_at
        )
