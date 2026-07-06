from __future__ import annotations

import random
import time
from collections.abc import Iterable

import numpy as np

from src.domain.meme_candidate import MemeCandidate
from src.domain.meme_media import MemeMedia
from src.domain.meme_profile import MemeProfile
from src.domain.ports import MemeMediaLoader, MemeProvider


class ApiMemeRepository:
    MAX_DOWNLOAD_ATTEMPTS = 4

    def __init__(
        self,
        profiles: Iterable[MemeProfile],
        providers: Iterable[MemeProvider],
        image_loader: MemeMediaLoader,
        max_results_per_query: int = 6,
        log_api_calls: bool = True,
        candidate_cache_seconds: float = 60.0,
    ) -> None:
        self.profiles_by_key = self._build_profiles_by_key(profiles)
        self.providers = tuple(providers)
        self.media_loader = image_loader
        self.max_results_per_query = max_results_per_query
        self.log_api_calls = log_api_calls
        self.candidate_cache_seconds = candidate_cache_seconds

        self.current_key: str | None = None
        self.current_media: MemeMedia | None = None
        self.current_media_started_at = 0.0
        self.current_candidate: MemeCandidate | None = None
        self._candidate_cache: dict[str, tuple[float, MemeCandidate | None]] = {}

    def has_enabled_provider(self) -> bool:
        return any(provider.is_enabled() for provider in self.providers)

    def get_available_keys(self) -> set[str]:
        if not self.has_enabled_provider():
            if self.log_api_calls:
                print("API providers: no enabled meme provider")

            return set()

        return set(self.profiles_by_key)

    def get_meme(self, meme_key: str) -> np.ndarray | None:
        if meme_key == self.current_key and self.current_media is not None:
            return self.current_media.frame_at(
                time.monotonic() - self.current_media_started_at
            )

        skipped_urls: set[str] = set()

        for _attempt in range(self.MAX_DOWNLOAD_ATTEMPTS):
            candidate = self._find_candidate(
                meme_key,
                skipped_urls=skipped_urls,
            )

            if candidate is None:
                return None

            media = self.media_loader.load_media(candidate.image_url)

            if media is None:
                skipped_urls.add(candidate.image_url)
                self._clear_cached_candidate(meme_key, candidate)

                if self.log_api_calls:
                    print(
                        "API repository: candidate failed, trying another "
                        f"key={meme_key} provider={candidate.provider}"
                    )

                continue

            self.current_key = meme_key
            self.current_media = media
            self.current_media_started_at = time.monotonic()
            self.current_candidate = candidate

            print(
                "Remote meme changed: "
                f"{meme_key} -> {candidate.provider}: "
                f"{candidate.title or candidate.query} "
                f"animated={media.is_animated}"
            )

            return media.frame_at(0.0)

        return None

    def _find_candidate(
        self,
        meme_key: str,
        skipped_urls: set[str] | None = None,
    ) -> MemeCandidate | None:
        skipped_urls = skipped_urls or set()
        cached_candidate = self._get_cached_candidate(meme_key)

        if (
            cached_candidate is not None
            and cached_candidate.image_url not in skipped_urls
        ):
            return cached_candidate

        if cached_candidate is not None:
            self._clear_cached_candidate(meme_key, cached_candidate)

        if self._has_recent_negative_cache(meme_key):
            if self.log_api_calls:
                print(f"API repository cache: no candidates for key={meme_key}")

            return None

        profile = self.profiles_by_key.get(meme_key)

        if profile is None:
            if self.log_api_calls:
                print(f"API repository: no profile for key={meme_key}")

            return None

        candidates: list[MemeCandidate] = []
        queries = self._queries_for_profile(profile)

        if self.log_api_calls:
            provider_names = [
                provider.name
                for provider in self.providers
                if provider.is_enabled()
            ]
            print(
                "API repository search: "
                f"key={meme_key} providers={provider_names} queries={list(queries)}"
            )

        for query in queries:
            for provider in self.providers:
                if not provider.is_enabled():
                    continue

                candidates.extend(
                    provider.search(
                        meme_key=meme_key,
                        query=query,
                        limit=self.max_results_per_query,
                    )
                )

        if not candidates:
            if self.log_api_calls:
                print(f"API repository: no remote candidates for key={meme_key}")

            self._set_cached_candidate(meme_key, None)
            return None

        candidates = [
            candidate
            for candidate in candidates
            if candidate.image_url not in skipped_urls
        ]

        if not candidates:
            return None

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        top_candidates = candidates[: min(len(candidates), 8)]

        selected_candidate = random.choice(top_candidates)

        if self.log_api_calls:
            print(
                "API repository selected: "
                f"key={meme_key} provider={selected_candidate.provider} "
                f"title='{selected_candidate.title or selected_candidate.query}'"
            )

        self._set_cached_candidate(meme_key, selected_candidate)

        return selected_candidate

    def _get_cached_candidate(self, meme_key: str) -> MemeCandidate | None:
        cached_item = self._candidate_cache.get(meme_key)

        if cached_item is None:
            return None

        cached_at, cached_candidate = cached_item

        if time.monotonic() - cached_at > self.candidate_cache_seconds:
            self._candidate_cache.pop(meme_key, None)
            return None

        if cached_candidate is not None and self.log_api_calls:
            print(f"API repository cache: candidate hit for key={meme_key}")

        return cached_candidate

    def _has_recent_negative_cache(self, meme_key: str) -> bool:
        cached_item = self._candidate_cache.get(meme_key)

        if cached_item is None:
            return False

        cached_at, cached_candidate = cached_item

        if cached_candidate is not None:
            return False

        if time.monotonic() - cached_at > self.candidate_cache_seconds:
            self._candidate_cache.pop(meme_key, None)
            return False

        return True

    def _set_cached_candidate(
        self,
        meme_key: str,
        candidate: MemeCandidate | None,
    ) -> None:
        self._candidate_cache[meme_key] = (time.monotonic(), candidate)

    def _clear_cached_candidate(
        self,
        meme_key: str,
        candidate: MemeCandidate,
    ) -> None:
        cached_item = self._candidate_cache.get(meme_key)

        if cached_item is None:
            return

        _cached_at, cached_candidate = cached_item

        if cached_candidate == candidate:
            self._candidate_cache.pop(meme_key, None)

    @staticmethod
    def _build_profiles_by_key(
        profiles: Iterable[MemeProfile],
    ) -> dict[str, MemeProfile]:
        profiles_by_key: dict[str, MemeProfile] = {}

        for profile in profiles:
            profiles_by_key.setdefault(profile.key, profile)

        return profiles_by_key

    @staticmethod
    def _queries_for_profile(profile: MemeProfile) -> tuple[str, ...]:
        if profile.search_terms:
            return profile.search_terms

        readable_key = profile.key.replace("_", " ")

        return (
            f"{readable_key} reaction meme",
            f"{readable_key} meme",
        )
