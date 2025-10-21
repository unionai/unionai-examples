# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b0",
#    "pydantic",
# ]
# main = "main"
# params = ""
# ///

import asyncio
from dataclasses import dataclass
from typing import List

from pydantic import BaseModel
import flyte

env = flyte.TaskEnvironment(name="ex-mixed-structures")


@dataclass
class InferenceRequest:
    feature_a: float
    feature_b: float


@dataclass
class BatchRequest:
    requests: List[InferenceRequest]
    batch_id: str = "default"


class PredictionSummary(BaseModel):
    predictions: List[float]
    average: float
    count: int
    batch_id: str


@env.task
async def predict_one(request: InferenceRequest) -> float:
    """
    A dummy linear model: prediction = 2 * feature_a + 3 * feature_b + bias(=1.0)
    """
    return 2.0 * request.feature_a + 3.0 * request.feature_b + 1.0


@env.task
async def process_batch(batch: BatchRequest) -> PredictionSummary:
    """
    Processes a batch of inference requests and returns summary statistics.
    """
    # Process all requests concurrently
    tasks = [predict_one(request=req) for req in batch.requests]
    predictions = await asyncio.gather(*tasks)

    # Calculate statistics
    average = sum(predictions) / len(predictions) if predictions else 0.0

    return PredictionSummary(
        predictions=predictions,
        average=average,
        count=len(predictions),
        batch_id=batch.batch_id
    )


@env.task
async def summarize_results(summary: PredictionSummary) -> str:
    """
    Creates a text summary from the prediction results.
    """
    return (
        f"Batch {summary.batch_id}: "
        f"Processed {summary.count} predictions, "
        f"average value: {summary.average:.2f}"
    )


@env.task
async def main() -> str:
    batch = BatchRequest(
        requests=[
            InferenceRequest(feature_a=1.0, feature_b=2.0),
            InferenceRequest(feature_a=3.0, feature_b=4.0),
            InferenceRequest(feature_a=5.0, feature_b=6.0),
        ],
        batch_id="demo_batch_001"
    )
    summary = await process_batch(batch)
    result = await summarize_results(summary)
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()