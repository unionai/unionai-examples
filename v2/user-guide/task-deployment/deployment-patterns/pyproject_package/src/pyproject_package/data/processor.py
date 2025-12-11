import asyncio
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DataItem(BaseModel):
    id: int = Field(gt=0, description="Item ID must be positive")
    value: float = Field(description="Item value")
    category: str = Field(min_length=1, description="Item category")

    @field_validator("category")
    @classmethod
    def category_must_be_lowercase(cls, v: str) -> str:
        """Ensure category is lowercase."""
        return v.lower()


def clean_data(raw_data: dict[str, Any]) -> dict[str, Any]:
    cleaned = {k: v for k, v in raw_data.items() if v is not None}

    if "items" in cleaned:
        validated_items = []
        for item in cleaned["items"]:
            try:
                validated = DataItem(**item)
                validated_items.append(validated.model_dump())
            except Exception as e:
                print(f"Skipping invalid item {item}: {e}")
                continue
        cleaned["items"] = validated_items

    return cleaned


def transform_data(data: dict[str, Any]) -> list[dict[str, Any]]:
    items = data.get("items", [])

    transformed = []
    for item in items:
        transformed_item = {
            **item,
            "value_squared": item["value"] ** 2,
            "category_upper": item["category"].upper(),
        }
        transformed.append(transformed_item)

    return transformed


async def aggregate_data(items: list[dict[str, Any]]) -> dict[str, Any]:
    await asyncio.sleep(0.1)

    aggregated: dict[str, dict[str, Any]] = {}

    for item in items:
        category = item["category"]

        if category not in aggregated:
            aggregated[category] = {
                "count": 0,
                "total_value": 0.0,
                "values": [],
            }

        aggregated[category]["count"] += 1
        aggregated[category]["total_value"] += item["value"]
        aggregated[category]["values"].append(item["value"])

    for category, v in aggregated.items():
        total = v["total_value"]
        count = v["count"]
        v["average_value"] = total / count if count > 0 else 0.0

    return {"categories": aggregated, "total_items": len(items)}
