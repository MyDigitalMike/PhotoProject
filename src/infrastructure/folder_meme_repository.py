from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np


class FolderMemeRepository:
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, meme_root: Path) -> None:
        self.meme_root = meme_root
        self.memes_by_key = self._load_meme_paths()

        self.current_key: str | None = None
        self.current_meme: np.ndarray | None = None

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

    def get_meme(self, meme_key: str) -> np.ndarray | None:
        image_paths = self.memes_by_key.get(meme_key)

        if not image_paths:
            image_paths = self.memes_by_key.get("neutral", [])

        if not image_paths:
            return None

        should_change = meme_key != self.current_key or self.current_meme is None

        if should_change:
            selected_path = random.choice(image_paths)
            image = cv2.imread(str(selected_path))

            if image is None:
                print(f"Could not load image: {selected_path}")
                return None

            self.current_key = meme_key
            self.current_meme = image

            print(f"Meme changed: {meme_key} -> {selected_path.name}")

        if self.current_meme is None:
            return None

        return self.current_meme.copy()