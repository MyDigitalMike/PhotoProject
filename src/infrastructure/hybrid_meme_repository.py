from __future__ import annotations

import numpy as np

from src.infrastructure.api_meme_repository import ApiMemeRepository
from src.infrastructure.folder_meme_repository import FolderMemeRepository


class HybridMemeRepository:
    def __init__(
        self,
        local_repository: FolderMemeRepository,
        api_repository: ApiMemeRepository,
        prefer_remote: bool = False,
        log_api_calls: bool = True,
    ) -> None:
        self.local_repository = local_repository
        self.api_repository = api_repository
        self.prefer_remote = prefer_remote
        self.log_api_calls = log_api_calls
        self._last_logged_lookup_key: str | None = None
        self._last_logged_hit: tuple[str, str] | None = None

    def get_available_keys(self) -> set[str]:
        return (
            self.local_repository.get_available_keys()
            | self.api_repository.get_available_keys()
        )

    def get_meme(self, meme_key: str) -> np.ndarray | None:
        repositories = self._repositories_in_order()

        if self.log_api_calls and meme_key != self._last_logged_lookup_key:
            order = [
                repository_name
                for repository_name, _repository in repositories
            ]
            print(f"Meme repository lookup: key={meme_key} order={order}")
            self._last_logged_lookup_key = meme_key

        for repository_name, repository in repositories:
            image = repository.get_meme(meme_key)

            if image is not None:
                current_hit = (meme_key, repository_name)

                if self.log_api_calls and current_hit != self._last_logged_hit:
                    print(
                        "Meme repository hit: "
                        f"key={meme_key} source={repository_name}"
                    )
                    self._last_logged_hit = current_hit

                return image

        if meme_key != "neutral":
            return self.local_repository.get_meme("neutral")

        return None

    def _repositories_in_order(
        self,
    ) -> tuple[
        tuple[str, ApiMemeRepository | FolderMemeRepository],
        tuple[str, ApiMemeRepository | FolderMemeRepository],
    ]:
        if self.prefer_remote:
            return (
                ("api", self.api_repository),
                ("local", self.local_repository),
            )

        return (
            ("local", self.local_repository),
            ("api", self.api_repository),
        )
