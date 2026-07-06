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
        media_cache_seconds: float = 900.0,
        minimum_display_seconds: float = 5.0,
        request_cooldown_seconds: float = 8.0,
    ) -> None:
        self.profiles_by_key = self._build_profiles_by_key(profiles)
        self.providers = tuple(providers)
        self.media_loader = image_loader
        self.max_results_per_query = max_results_per_query
        self.log_api_calls = log_api_calls
        self.candidate_cache_seconds = max(0.0, candidate_cache_seconds)
        self.media_cache_seconds = max(0.0, media_cache_seconds)
        self.minimum_display_seconds = max(0.0, minimum_display_seconds)
        self.request_cooldown_seconds = max(0.0, request_cooldown_seconds)

        self.current_key: str | None = None
        self.current_media: MemeMedia | None = None
        self.current_media_started_at = 0.0
        self.current_candidate: MemeCandidate | None = None
        self._candidate_cache: dict[str, tuple[float, MemeCandidate | None]] = {}
        self._media_cache: dict[str, tuple[float, MemeMedia, MemeCandidate]] = {}
        self._last_provider_request_at = 0.0
        self._last_hold_log: tuple[str | None, str] | None = None
        self._last_cooldown_log: tuple[str, float] | None = None
        self._last_cooldown_hold_log: tuple[str | None, str, float] | None = None

    def has_enabled_provider(self) -> bool:
        return any(provider.is_enabled() for provider in self.providers)

    def get_available_keys(self) -> set[str]:
        if not self.has_enabled_provider():
            if self.log_api_calls:
                print("API providers: no enabled meme provider")

            return set()

        return set(self.profiles_by_key)

    def get_meme(self, meme_key: str) -> np.ndarray | None:
        now = time.monotonic()

        if meme_key == self.current_key and self.current_media is not None:
            return self.current_media.frame_at(
                now - self.current_media_started_at
            )

        if self._should_hold_current_media(meme_key=meme_key, now=now):
            return self.current_media.frame_at(now - self.current_media_started_at)

        self._last_hold_log = None

        cached_media = self._get_cached_media(meme_key=meme_key, now=now)

        if cached_media is not None:
            media, candidate = cached_media
            self._use_media(
                meme_key=meme_key,
                media=media,
                candidate=candidate,
                now=now,
            )

            return media.frame_at(0.0)

        if self._should_hold_for_request_cooldown(meme_key=meme_key, now=now):
            return self.current_media.frame_at(now - self.current_media_started_at)

        skipped_urls: set[str] = set()

        for _attempt in range(self.MAX_DOWNLOAD_ATTEMPTS):
            candidate = self._find_candidate(
                meme_key,
                skipped_urls=skipped_urls,
                ignore_request_cooldown=bool(skipped_urls),
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

            self._use_media(
                meme_key=meme_key,
                media=media,
                candidate=candidate,
                now=time.monotonic(),
            )
            self._set_cached_media(meme_key, media, candidate)

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
        ignore_request_cooldown: bool = False,
    ) -> MemeCandidate | None:
        now = time.monotonic()
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

        cooldown_remaining = self._provider_cooldown_remaining(now)

        if not ignore_request_cooldown and cooldown_remaining > 0:
            self._log_provider_cooldown(meme_key, cooldown_remaining)
            return None

        candidates: list[MemeCandidate] = []
        queries = self._queries_for_profile(profile)
        self._last_provider_request_at = now
        self._last_cooldown_log = None
        self._last_cooldown_hold_log = None

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

    def _use_media(
        self,
        meme_key: str,
        media: MemeMedia,
        candidate: MemeCandidate,
        now: float,
    ) -> None:
        self.current_key = meme_key
        self.current_media = media
        self.current_media_started_at = now
        self.current_candidate = candidate
        self._last_hold_log = None

    def _should_hold_current_media(
        self,
        meme_key: str,
        now: float,
    ) -> bool:
        if self.current_media is None:
            return False

        if self.current_key == meme_key:
            return False

        if self.minimum_display_seconds <= 0:
            return False

        elapsed_seconds = now - self.current_media_started_at

        if elapsed_seconds >= self.minimum_display_seconds:
            return False

        if self.log_api_calls:
            log_key = (self.current_key, meme_key)

            if log_key != self._last_hold_log:
                remaining_seconds = self.minimum_display_seconds - elapsed_seconds
                print(
                    "API repository hold: "
                    f"keeping key={self.current_key} requested={meme_key} "
                    f"remaining={remaining_seconds:.1f}s"
                )
                self._last_hold_log = log_key

        return True

    def _should_hold_for_request_cooldown(
        self,
        meme_key: str,
        now: float,
    ) -> bool:
        if self.current_media is None:
            return False

        cooldown_remaining = self._provider_cooldown_remaining(now)

        if cooldown_remaining <= 0:
            return False

        if self.log_api_calls:
            log_key = (
                self.current_key,
                meme_key,
                self._last_provider_request_at,
            )

            if log_key != self._last_cooldown_hold_log:
                print(
                    "API repository cooldown: "
                    f"keeping key={self.current_key} requested={meme_key} "
                    f"remaining={cooldown_remaining:.1f}s"
                )
                self._last_cooldown_hold_log = log_key

        return True

    def _get_cached_media(
        self,
        meme_key: str,
        now: float,
    ) -> tuple[MemeMedia, MemeCandidate] | None:
        cached_item = self._media_cache.get(meme_key)

        if cached_item is None:
            return None

        cached_at, media, candidate = cached_item

        if now - cached_at > self.media_cache_seconds:
            self._media_cache.pop(meme_key, None)
            return None

        if self.log_api_calls:
            print(f"API repository media cache: hit for key={meme_key}")

        return media, candidate

    def _set_cached_media(
        self,
        meme_key: str,
        media: MemeMedia,
        candidate: MemeCandidate,
    ) -> None:
        if self.media_cache_seconds <= 0:
            return

        self._media_cache[meme_key] = (time.monotonic(), media, candidate)

    def _provider_cooldown_remaining(self, now: float) -> float:
        if self.request_cooldown_seconds <= 0:
            return 0.0

        if self._last_provider_request_at <= 0:
            return 0.0

        elapsed_seconds = now - self._last_provider_request_at

        return max(0.0, self.request_cooldown_seconds - elapsed_seconds)

    def _log_provider_cooldown(
        self,
        meme_key: str,
        remaining_seconds: float,
    ) -> None:
        if not self.log_api_calls:
            return

        log_key = (meme_key, self._last_provider_request_at)

        if log_key == self._last_cooldown_log:
            return

        print(
            "API repository cooldown: "
            f"skipping provider search key={meme_key} "
            f"remaining={remaining_seconds:.1f}s"
        )
        self._last_cooldown_log = log_key

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
