# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "httpx",
#    "numpy",
# ]
# ///
"""Send concurrent forecast requests to the Timestral forecasting API.

python forecast_client.py https://<your-app-url> --requests 200
"""

import argparse
import asyncio
import time

import httpx
import numpy as np


# {{docs-fragment client}}
def synthetic_history(rng: np.random.Generator, length: int) -> list[float]:
    """An hourly series with daily and weekly cycles plus noise."""
    t = np.arange(length)
    daily = 10 * np.sin(2 * np.pi * t / 24)
    weekly = 5 * np.sin(2 * np.pi * t / 168)
    return (100 + daily + weekly + rng.normal(0, 2, length)).round(3).tolist()


async def main(url: str, n_requests: int, horizon: int) -> None:
    rng = np.random.default_rng(0)
    async with httpx.AsyncClient(base_url=url, timeout=60) as client:
        print((await client.get("/health")).json())

        async def one() -> float:
            body = {
                "values": synthetic_history(rng, int(rng.integers(500, 3000))),
                "horizon": horizon,
                "quantiles": [0.1, 0.5, 0.9],
            }
            start = time.perf_counter()
            response = await client.post("/forecast", json=body)
            response.raise_for_status()
            return time.perf_counter() - start

        start = time.perf_counter()
        latencies = await asyncio.gather(*(one() for _ in range(n_requests)))
        elapsed = time.perf_counter() - start

    print(f"{n_requests} forecasts in {elapsed:.2f}s ({n_requests / elapsed:.0f}/s)")
    print(
        f"p50 latency {1000 * np.percentile(latencies, 50):.0f} ms, p95 {1000 * np.percentile(latencies, 95):.0f} ms"
    )


# {{/docs-fragment client}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()
    asyncio.run(main(args.url, args.requests, args.horizon))
