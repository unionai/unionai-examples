# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = ""
# ///

import asyncio
import logging

import flyte

# Mock API client for the example
class MockAPIClient:
    async def post(self, endpoint: str, json: dict):
        """Mock API client that simulates failure for invalid data."""
        if json.get("invalid") == "data":
            raise ValueError("Invalid data provided to API")
        return {"result": "success", "processed": json}

    def json(self):
        """Mock response method"""
        return self

api_client = MockAPIClient()
logger = logging.getLogger(__name__)

env = flyte.TaskEnvironment("env")

# {{docs-fragment all}}
@flyte.trace
async def risky_api_call(endpoint: str, data: dict) -> dict:
    """API call that might fail - traces capture errors."""
    try:
        response = await api_client.post(endpoint, json=data)
        return response.json()
    except Exception as e:
        # Error is automatically captured in trace
        logger.error(f"API call failed: {e}")
        raise

@env.task
async def error_handling() -> dict:
    try:
        result = await risky_api_call("/process", {"invalid": "data"})
        return {"status": "success", "result": result}
    except Exception as e:
        # The error is recorded in the trace for debugging
        return {"status": "error", "message": str(e)}
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(error_handling)
    print(r.name)
    print(r.url)
    r.wait()
