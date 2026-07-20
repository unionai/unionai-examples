from flytekit import task, workflow, map_task
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


@task
def get_grid() -> list[int]:
    return [2, 4, 8, 16]


@task
def train_eval(max_depth: int) -> float:
    data = load_iris()
    model = RandomForestClassifier(max_depth=max_depth, random_state=42)
    scores = cross_val_score(model, data.data, data.target, cv=3)
    return float(scores.mean())


@task
def best_score(scores: list[float]) -> float:
    return max(scores)


@workflow
def main() -> float:
    grid = get_grid()
    # Fan out one training run per hyperparameter value.
    scores = map_task(train_eval)(max_depth=grid)
    return best_score(scores=scores)
