from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GiphyProviderConfig:
    enabled: bool = False
    api_key_env: str = "GIPHY_API_KEY"
    rating: str = "pg-13"
    language: str = "en"


@dataclass(frozen=True)
class ImgflipProviderConfig:
    enabled: bool = True


@dataclass(frozen=True)
class MemeApiConfig:
    enabled: bool = True
    prefer_remote: bool = False
    log_api_calls: bool = True
    max_results_per_query: int = 6
    candidate_cache_seconds: float = 60.0
    media_cache_seconds: float = 900.0
    minimum_remote_display_seconds: float = 5.0
    remote_request_cooldown_seconds: float = 8.0
    giphy: GiphyProviderConfig = field(default_factory=GiphyProviderConfig)
    imgflip: ImgflipProviderConfig = field(default_factory=ImgflipProviderConfig)


class MemeApiConfigLoader:
    VALID_GIPHY_RATINGS = {"g", "pg", "pg-13", "r"}

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path

    def load(self) -> MemeApiConfig:
        if not self.config_path.exists():
            return MemeApiConfig()

        with self.config_path.open("r", encoding="utf-8") as config_file:
            raw_config = json.load(config_file)

        if not isinstance(raw_config, dict):
            return MemeApiConfig()

        return MemeApiConfig(
            enabled=bool(raw_config.get("enabled", True)),
            prefer_remote=bool(raw_config.get("prefer_remote", False)),
            log_api_calls=bool(raw_config.get("log_api_calls", True)),
            max_results_per_query=int(raw_config.get("max_results_per_query", 6)),
            candidate_cache_seconds=float(
                raw_config.get("candidate_cache_seconds", 60.0)
            ),
            media_cache_seconds=float(raw_config.get("media_cache_seconds", 900.0)),
            minimum_remote_display_seconds=float(
                raw_config.get("minimum_remote_display_seconds", 5.0)
            ),
            remote_request_cooldown_seconds=float(
                raw_config.get("remote_request_cooldown_seconds", 8.0)
            ),
            giphy=self._giphy_config(raw_config.get("giphy", {})),
            imgflip=self._imgflip_config(raw_config.get("imgflip", {})),
        )

    @staticmethod
    def _giphy_config(raw_config: Any) -> GiphyProviderConfig:
        if not isinstance(raw_config, dict):
            return GiphyProviderConfig()

        rating = str(raw_config.get("rating", "pg-13")).lower()

        if rating == "pg-18":
            print("GIPHY config warning: rating 'pg-18' is invalid; using 'r'.")
            rating = "r"
        elif rating not in MemeApiConfigLoader.VALID_GIPHY_RATINGS:
            print(
                "GIPHY config warning: "
                f"rating '{rating}' is invalid; using 'pg-13'."
            )
            rating = "pg-13"

        return GiphyProviderConfig(
            enabled=bool(raw_config.get("enabled", False)),
            api_key_env=str(raw_config.get("api_key_env", "GIPHY_API_KEY")),
            rating=rating,
            language=str(raw_config.get("language", "en")),
        )

    @staticmethod
    def _imgflip_config(raw_config: Any) -> ImgflipProviderConfig:
        if not isinstance(raw_config, dict):
            return ImgflipProviderConfig()

        return ImgflipProviderConfig(
            enabled=bool(raw_config.get("enabled", True)),
        )
