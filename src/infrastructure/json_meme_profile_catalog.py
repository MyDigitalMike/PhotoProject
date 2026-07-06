from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.domain.meme_profile import MemeProfile


class JsonMemeProfileCatalog:
    def __init__(self, catalog_path: Path) -> None:
        self.catalog_path = catalog_path

    def load_profiles(self) -> tuple[MemeProfile, ...]:
        if not self.catalog_path.exists():
            return ()

        with self.catalog_path.open("r", encoding="utf-8") as catalog_file:
            raw_catalog = json.load(catalog_file)

        raw_profiles = raw_catalog.get("profiles", [])

        if not isinstance(raw_profiles, list):
            return ()

        profiles = [
            self._profile_from_mapping(raw_profile)
            for raw_profile in raw_profiles
            if isinstance(raw_profile, dict)
        ]

        return tuple(profiles)

    @staticmethod
    def _profile_from_mapping(raw_profile: dict[str, Any]) -> MemeProfile:
        return MemeProfile(
            key=str(raw_profile.get("key", "neutral")).lower(),
            priority=float(raw_profile.get("priority", 0.0)),
            min_total_score=float(raw_profile.get("min_total_score", 0.0)),
            emotion_weights=JsonMemeProfileCatalog._float_dict(
                raw_profile.get("emotion_weights", {})
            ),
            signal_weights=JsonMemeProfileCatalog._float_dict(
                raw_profile.get("signal_weights", {})
            ),
            min_scores=JsonMemeProfileCatalog._float_dict(
                raw_profile.get("min_scores", {})
            ),
            any_min_scores=JsonMemeProfileCatalog._float_dict(
                raw_profile.get("any_min_scores", {})
            ),
            required_signals=JsonMemeProfileCatalog._string_tuple(
                raw_profile.get("required_signals", [])
            ),
            any_signals=JsonMemeProfileCatalog._string_tuple(
                raw_profile.get("any_signals", [])
            ),
            blocked_signals=JsonMemeProfileCatalog._string_tuple(
                raw_profile.get("blocked_signals", [])
            ),
            search_terms=JsonMemeProfileCatalog._string_tuple(
                raw_profile.get("search_terms", [])
            ),
            description=str(raw_profile.get("description", "")),
        )

    @staticmethod
    def _float_dict(raw_value: Any) -> dict[str, float]:
        if not isinstance(raw_value, dict):
            return {}

        return {
            str(key).lower(): float(value)
            for key, value in raw_value.items()
        }

    @staticmethod
    def _string_tuple(raw_value: Any) -> tuple[str, ...]:
        if not isinstance(raw_value, list):
            return ()

        return tuple(str(value) for value in raw_value)
