from __future__ import annotations

import os
from pathlib import Path


class EnvFileLoader:
    def __init__(
        self,
        env_paths: tuple[Path, ...],
    ) -> None:
        self.env_paths = env_paths

    def load(self) -> None:
        for env_path in self.env_paths:
            if not env_path.exists():
                continue

            self._load_env_file(env_path)

    @staticmethod
    def _load_env_file(env_path: Path) -> None:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value
