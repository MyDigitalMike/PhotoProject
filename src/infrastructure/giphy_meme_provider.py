from __future__ import annotations

from typing import Any

from src.domain.meme_candidate import MemeCandidate
from src.infrastructure.http_json_client import HttpJsonClient


class GiphyMemeProvider:
    name = "giphy"
    SEARCH_URL = "https://api.giphy.com/v1/gifs/search"

    def __init__(
        self,
        api_key: str | None,
        http_client: HttpJsonClient,
        rating: str = "pg-13",
        language: str = "en",
        log_api_calls: bool = True,
    ) -> None:
        self.api_key = api_key
        self.http_client = http_client
        self.rating = rating
        self.language = language
        self.log_api_calls = log_api_calls

    def is_enabled(self) -> bool:
        return bool(self.api_key)

    def search(
        self,
        meme_key: str,
        query: str,
        limit: int,
    ) -> list[MemeCandidate]:
        if not self.is_enabled():
            return []

        if self.log_api_calls:
            print(f"API search: provider=giphy key={meme_key} query='{query}'")

        params = {
            "api_key": self.api_key or "",
            "q": query,
            "limit": limit,
            "rating": self.rating,
            "lang": self.language,
        }

        try:
            payload = self.http_client.get_json(
                self.SEARCH_URL,
                params=params,
            )
        except Exception as error:
            print(f"GIPHY search failed: {error}")
            return []

        if not isinstance(payload, dict):
            return []

        raw_items = payload.get("data", [])

        if not isinstance(raw_items, list):
            return []

        candidates: list[MemeCandidate] = []

        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue

            image_url = self._extract_image_url(item)

            if not image_url:
                continue

            candidates.append(
                MemeCandidate(
                    key=meme_key,
                    query=query,
                    image_url=image_url,
                    provider=self.name,
                    title=str(item.get("title", "")),
                    source_url=str(item.get("url", "")),
                    score=float(limit - index),
                )
            )

        if self.log_api_calls:
            print(
                "API search result: "
                f"provider=giphy key={meme_key} raw_items={len(raw_items)} "
                f"candidates={len(candidates)}"
            )

            if raw_items and not candidates:
                sample_item = raw_items[0]

                if isinstance(sample_item, dict):
                    sample_images = sample_item.get("images", {})

                    if isinstance(sample_images, dict):
                        print(
                            "GIPHY debug: no extractable image URL. "
                            f"available_renditions={list(sample_images)[:12]}"
                        )

        return candidates

    @staticmethod
    def _extract_image_url(item: dict[str, Any]) -> str:
        images = item.get("images", {})

        if not isinstance(images, dict):
            return ""

        for rendition_name in (
            "fixed_height_small",
            "fixed_width_small",
            "fixed_height",
            "fixed_width",
            "downsized",
            "downsized_small",
            "original",
            "preview_gif",
            "fixed_height_small_still",
            "fixed_height_still",
            "downsized_still",
            "original_still",
        ):
            rendition = images.get(rendition_name, {})

            if not isinstance(rendition, dict):
                continue

            image_url = rendition.get("url")

            if image_url:
                return str(image_url)

        return ""
