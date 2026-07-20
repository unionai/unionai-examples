# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "scikit-learn",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment all}}
import asyncio

import flyte
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

env = flyte.TaskEnvironment(
    name="hpo",
    image=flyte.Image.from_debian_base().with_pip_packages("scikit-learn"),
)


@env.task
async def train_eval(max_depth: int) -> float:
    data = load_iris()
    model = RandomForestClassifier(max_depth=max_depth, random_state=42)
    scores = cross_val_score(model, data.data, data.target, cv=3)
    return float(scores.mean())


@env.task
async def main() -> dict:
    grid = [2, 4, 8, 16]
    # Fan out one training run per hyperparameter value...
    scores = await asyncio.gather(*[train_eval(d) for d in grid])
    # ...then pick the best in plain Python (impossible in a Flyte 1 workflow).
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return {"best_max_depth": grid[best_idx], "best_score": scores[best_idx]}
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
