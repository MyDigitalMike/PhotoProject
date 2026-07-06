from __future__ import annotations

import re
from typing import Any

from src.domain.meme_candidate import MemeCandidate
from src.infrastructure.http_json_client import HttpJsonClient


class ImgflipMemeProvider:
    name = "imgflip"
    GET_MEMES_URL = "https://api.imgflip.com/get_memes"

    def __init__(
        self,
        http_client: HttpJsonClient,
        enabled: bool = True,
        log_api_calls: bool = True,
    ) -> None:
        self.http_client = http_client
        self.enabled = enabled
        self.log_api_calls = log_api_calls
        self._cached_templates: list[dict[str, Any]] | None = None

    def is_enabled(self) -> bool:
        return self.enabled

    def search(
        self,
        meme_key: str,
        query: str,
        limit: int,
    ) -> list[MemeCandidate]:
        if not self.is_enabled():
            return []

        if self.log_api_calls:
            print(f"API search: provider=imgflip key={meme_key} query='{query}'")

        templates = self._get_templates()
        scored_templates = [
            (self._score_template(query, template), template)
            for template in templates
        ]
        scored_templates = [
            (score, template)
            for score, template in scored_templates
            if score > 0
        ]
        scored_templates.sort(key=lambda item: item[0], reverse=True)

        candidates: list[MemeCandidate] = []

        for score, template in scored_templates[:limit]:
            image_url = str(template.get("url", ""))

            if not image_url:
                continue

            candidates.append(
                MemeCandidate(
                    key=meme_key,
                    query=query,
                    image_url=image_url,
                    provider=self.name,
                    title=str(template.get("name", "")),
                    source_url=f"https://imgflip.com/memegenerator/{template.get('id', '')}",
                    score=score,
                )
            )

        if self.log_api_calls:
            print(
                "API search result: "
                f"provider=imgflip key={meme_key} candidates={len(candidates)}"
            )

        return candidates

    def _get_templates(self) -> list[dict[str, Any]]:
        if self._cached_templates is not None:
            return self._cached_templates

        if self.log_api_calls:
            print("API fetch: provider=imgflip endpoint=get_memes")

        try:
            payload = self.http_client.get_json(self.GET_MEMES_URL)
        except Exception as error:
            print(f"Imgflip get_memes failed: {error}")
            self._cached_templates = []
            return self._cached_templates

        if not isinstance(payload, dict):
            self._cached_templates = []
            return self._cached_templates

        data = payload.get("data", {})

        if not isinstance(data, dict):
            self._cached_templates = []
            return self._cached_templates

        memes = data.get("memes", [])

        if not isinstance(memes, list):
            self._cached_templates = []
            return self._cached_templates

        self._cached_templates = [
            meme
            for meme in memes
            if isinstance(meme, dict)
        ]

        return self._cached_templates

    @staticmethod
    def _score_template(
        query: str,
        template: dict[str, Any],
    ) -> float:
        query_tokens = set(ImgflipMemeProvider._tokens(query))
        title = str(template.get("name", ""))
        title_tokens = set(ImgflipMemeProvider._tokens(title))

        if not query_tokens or not title_tokens:
            return 0.0

        overlap = query_tokens & title_tokens

        if not overlap:
            return 0.0

        captions = float(template.get("captions", 0) or 0)
        popularity_bonus = min(captions / 100000.0, 2.0)

        return float(len(overlap)) + popularity_bonus

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2
        ]
