import pathlib
from typing import Any

import flyte
from pyproject_package.data import loader, processor
from pyproject_package.models import analyzer

UV_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.parent

env = flyte.TaskEnvironment(
    name="data_pipeline",
    image=flyte.Image.from_debian_base().with_uv_project(pyproject_file=UV_PROJECT_ROOT / "pyproject.toml"),
    resources=flyte.Resources(memory="512Mi", cpu="500m"),
)


@env.task
async def fetch_task(url: str) -> list[dict[str, Any]]:
    print(f"Fetching data from: {url}")
    data = await loader.fetch_data_from_api(url)
    print(f"Fetched {len(data)} top-level keys")
    return data


@env.task
async def pipeline(api_url: str) -> dict:
    raw_data = await loader.fetch_data_from_api(api_url)
    clean_data_result = processor.clean_data(raw_data)
    transformed = processor.transform_data(clean_data_result)
    return analyzer.analyze_results(transformed)


@env.task
async def process_task(raw_data: dict[str, Any]) -> list[dict[str, Any]]:
    print("Cleaning data...")
    cleaned = processor.clean_data(raw_data)

    print("Transforming data...")
    transformed = processor.transform_data(cleaned)

    print(f"Processed {len(transformed)} items")
    return transformed


@env.task
async def analyze_task(processed_data: list[dict[str, Any]]) -> str:
    print("Aggregating data...")
    aggregated = await processor.aggregate_data(processed_data)

    print("Calculating statistics...")
    stats = analyzer.calculate_statistics(processed_data)

    print("Generating report...")
    report = analyzer.generate_report({"basic": stats, "aggregated": aggregated})

    print("\n" + report)
    return report


@env.task
async def pipeline(api_url: str) -> str:
    raw_data = await fetch_task(url=api_url)
    processed_data = await process_task(raw_data=raw_data[0])
    report = await analyze_task(processed_data=processed_data)

    return report
