import json
from pathlib import Path
from typing import Any

import httpx


async def fetch_data_from_api(url: str) -> list[dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=10.0)
        response.raise_for_status()
        return response.json()


def load_local_data(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with path.open("r") as f:
        return json.load(f)
