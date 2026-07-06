from pathlib import Path

from src.infrastructure.env_file_loader import EnvFileLoader
from src.infrastructure.giphy_meme_provider import GiphyMemeProvider
from src.infrastructure.http_json_client import HttpJsonClient
from src.infrastructure.meme_api_config import MemeApiConfigLoader


def main() -> None:
    EnvFileLoader(
        env_paths=(
            Path(".env"),
            Path(".env.local"),
        ),
    ).load()

    config = MemeApiConfigLoader(
        Path("assets") / "memes" / "api_providers.json"
    ).load()
    api_key = _get_env(config.giphy.api_key_env)

    print(
        "GIPHY test config: "
        f"enabled={config.giphy.enabled}, "
        f"env={config.giphy.api_key_env}, "
        f"key={_mask_secret(api_key)}, "
        f"rating={config.giphy.rating}"
    )

    provider = GiphyMemeProvider(
        api_key=api_key,
        http_client=HttpJsonClient(),
        rating=config.giphy.rating,
        language=config.giphy.language,
        log_api_calls=True,
    )
    candidates = provider.search(
        meme_key="neutral",
        query="neutral reaction meme",
        limit=6,
    )

    print(f"GIPHY test candidates: {len(candidates)}")

    for candidate in candidates[:3]:
        print(
            f"- {candidate.title or candidate.query} | "
            f"{candidate.image_url}"
        )


def _get_env(key: str) -> str | None:
    import os

    return os.environ.get(key)


def _mask_secret(secret: str | None) -> str:
    if not secret:
        return "missing"

    if len(secret) <= 8:
        return "present"

    return f"{secret[:4]}...{secret[-4:]}"


if __name__ == "__main__":
    main()
