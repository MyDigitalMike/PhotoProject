from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class HttpJsonClient:
    def __init__(self, timeout_seconds: float = 5.0) -> None:
        self.timeout_seconds = timeout_seconds

    def get_json(
        self,
        url: str,
        params: dict[str, str | int | float] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        request_url = self._build_url(url, params or {})
        request = Request(
            request_url,
            headers=headers or {},
            method="GET",
        )

        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = response.read().decode("utf-8")

        return json.loads(payload)

    def post_json_bytes(
        self,
        url: str,
        body: bytes,
        headers: dict[str, str] | None = None,
    ) -> Any:
        request = Request(
            url,
            data=body,
            headers=headers or {},
            method="POST",
        )

        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = response.read().decode("utf-8")

        return json.loads(payload)

    @staticmethod
    def _build_url(
        url: str,
        params: dict[str, str | int | float],
    ) -> str:
        if not params:
            return url

        separator = "&" if "?" in url else "?"

        return f"{url}{separator}{urlencode(params)}"
