# {{docs-fragment import}}
import asyncio
import time
from typing import List

import flyte
from async_lru import alru_cache
# {{/docs-fragment import}}

# {{docs-fragment mock}}
# Mock expensive model that takes time to "load"
class ExpensiveModel:
    def __init__(self):
        self.loaded_at = time.time()
        print(f"✅ Model loaded successfully at {self.loaded_at}")

    @classmethod
    async def create(cls):
        """Async factory method to create the expensive model"""
        print("🔄 Loading expensive model... (this takes 5 seconds)")
        await asyncio.sleep(5)  # Simulate expensive model loading
        return cls()

    def predict(self, data: List[float]) -> float:
        # Simple mock prediction: return sum of inputs
        result = sum(data) * 1.5  # Some "AI" calculation
        print(f"🧠 Model prediction: {data} -> {result}")
        return result


@alru_cache(maxsize=1)
async def load_expensive_model() -> ExpensiveModel:
    """Async factory function to create the expensive model with caching"""
    return await ExpensiveModel.create()
# {{/docs-fragment mock}}

# {{docs-fragment env}}
# Currently required to enable reusable containers
reusable_image = flyte.Image.from_debian_base().with_pip_packages("unionai-reuse>=0.1.3")


env = flyte.TaskEnvironment(
    name="ml_env",
    resources=flyte.Resources(memory="2Gi", cpu="1"),
    reusable=flyte.ReusePolicy(
        replicas=1,         # Single container to clearly see reuse
        concurrency=3,      # Allow 3 concurrent predictions
        scaledown_ttl=300,  # Keep container alive for 5 minutes
        idle_ttl=1800       # Keep environment alive for 30 minutes
    ),
    image=reusable_image
)
# {{/docs-fragment env}}

# {{docs-fragment do_predict}}
# Model loaded once per container
model = None
model_lock = asyncio.Lock()

@env.task
async def do_predict(data: List[float]) -> float:
    """
    Prediction task that loads the model once per container
    and reuses it for subsequent predictions.
    """
    global model

    print(f"🚀 Task started with data: {data}")

    # Thread-safe lazy loading of the expensive model
    if model is None:
        async with model_lock:
            if model is None:  # Double-check pattern
                print("📦 No model found, loading expensive model...")
                # Load the model asynchronously with caching
                model = await load_expensive_model()
            else:
                print("⚡ Another task already loaded the model while we waited")
    else:
        print("⚡ Model already loaded, reusing existing model")

    # Use the model for prediction
    result = model.predict(data)
    print(f"✨ Task completed: {data} -> {result}")
    return result
# {{/docs-fragment do_predict}}

# {{docs-fragment main}}
@env.task
async def main() -> List[float]:
    """
    Main workflow that calls do_predict multiple times.
    The first call will load the model, subsequent calls will reuse it.
    """
    print("🎯 Starting ML inference workflow with reusable containers")

    # Test data for predictions
    test_data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0]
    ]

    print(f"📊 Running {len(test_data)} predictions...")

    # Run predictions - these may execute concurrently due to concurrency=3
    # but they'll all reuse the same model once it's loaded
    results = []
    for i, data in enumerate(test_data):
        print(f"📤 Submitting prediction {i+1}/{len(test_data)}")
        result = await do_predict(data)
        results.append(result)

        # Small delay to see the timing more clearly
        await asyncio.sleep(1)

    print("🏁 All predictions completed!")
    print(f"📈 Results: {results}")
    return results
# {{/docs-fragment main}}

# {{docs-fragment run}}
if __name__ == "__main__":
    # Establish a remote connection from within your script.
    flyte.init_from_config()

    # Run your tasks remotely inline and pass parameter data.
    run = flyte.run(main)

    # Print various attributes of the run.
    print(run.name)
    print(run.url)

    # Stream the logs from the remote run to the terminal.
    run.wait(run)
# {{/docs-fragment run}}