# # Track XGBoost Training with Neptune
#
# {{run-on-union}}
#
# Neptune is an MLOps tool for experiment tracking.
# It provides a centralized location to log, compare, store, and collaborate on experiments and models.
# This plugin enables seamless integration of Neptune with Flyte by configuring connections between the two platforms.
# In this example, we demonstrate how to scale the training of multiple XGBoost models while using Neptune for tracking.

# First, we need to import the necessary libraries.

from typing import List, Tuple

import numpy as np
import flytekit
from flytekitplugins.neptune import neptune_scale_run

# We then specify the Neptune project that was created on Neptune's platform.
# Please update `NEPTUNE_PROJECT` to the value associated with your account.

NEPTUNE_PROJECT = (
    "username/project"  # TODO: Update this to your Neptune "workspace/project"
)

# Neptune requires an API key for authentication. You can securely provide this key to Flyte
# by creating a secret using the [secrets manager](https://www.union.ai/docs/flyte/deployment/flyte-configuration/secrets/).

NEPTUNE_API_KEY = flytekit.Secret(group="neptune-api-group", key="neptune-api-token")

# We use `ImageSpec` to build a container image with the dependencies required for the XGBoost training task.
# Make sure to set `REGISTRY` to a container registry accessible by your cluster.

REGISTRY = "localhost:30000"

image = flytekit.ImageSpec(
    name="flytekit-xgboost",
    packages=[
        "neptune",
        "neptune-xgboost",
        "flytekitplugins-neptune",
        "scikit-learn==1.5.1",
        "numpy==1.26.1",
        "matplotlib==3.9.2",
    ],
    builder="default",
    registry=REGISTRY,
)

# To train and track our XGBoost model, we'll first need a dataset.
# In this example, we use the California housing dataset, which provides a clean and structured foundation for
# regression tasks. We can easily download it using the `fetch_california_housing` function from `sklearn.datasets`.


@flytekit.task(
    container_image=image,
    cache=True,
    cache_version="v1",
    requests=flytekit.Resources(cpu="2", mem="2Gi"),
)
def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True, as_frame=False)
    return X, y


# Next, we integrate Neptune into our training workflow using the `neptune_scale_run` decorator.
# This decorator initializes a [Neptune run](https://docs.neptune.ai/run/) and handles authentication using the `NEPTUNE_API_KEY` secret,
# which must be both defined in the decorator and explicitly requested in the task definition.
# Once initialized, the Neptune Run object becomes available via `current_context().neptune_run`.

# In this example, we log metadata to the Neptune run, such as model parameters and evaluation metrics.


@flytekit.task(
    container_image=image,
    secret_requests=[NEPTUNE_API_KEY],
    requests=flytekit.Resources(cpu="2", mem="4Gi"),
)
@neptune_scale_run(project=NEPTUNE_PROJECT, secret=NEPTUNE_API_KEY)
def train_model(max_depth: int, X: np.ndarray, y: np.ndarray):
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_test, label=y_test)

    model_params = {
        "tree_method": "hist",
        "eta": 0.7,
        "gamma": 0.001,
        "max_depth": max_depth,
        "objective": "reg:squarederror",
        "eval_metric": ["mae", "rmse"],
    }
    evals = [(dtrain, "train"), (dval, "valid")]

    ctx = flytekit.current_context()
    run = ctx.neptune_run
    run.log_configs(
        {
            "parameters/eta": 0.7,
            "parameters/gamma": 0.001,
            "parameters/tree_method": "hist",
            "parameters/eval_metric": ["mae", "rmse"],
            "parameters/max_depth": max_depth,
            "parameters/objective": "reg:squarederror",
        }
    )

    # Train the model and log metadata to the run in Neptune
    xgb.train(
        params=model_params,
        dtrain=dtrain,
        num_boost_round=57,
        evals=evals,
        callbacks=[
            xgb.callback.LearningRateScheduler(lambda epoch: 0.99**epoch),
            xgb.callback.EarlyStopping(rounds=30),
        ],
    )


# You can also specify `run_id` and `experiment_name` in the `neptune_scale_run` decorator.

# Using Flyte's dynamic workflows, we scale out multiple training jobs with different `max_depth` values.


@flytekit.dynamic(container_image=image)
def train_multiple_models(max_depths: List[int], X: np.ndarray, y: np.ndarray):
    for max_depth in max_depths:
        train_model(max_depth=max_depth, X=X, y=y)


@flytekit.workflow
def train_wf(max_depths: List[int] = [2, 4, 10]):
    X, y = get_dataset()
    train_multiple_models(max_depths=max_depths, X=X, y=y)


# To run this workflow on a remote Flyte cluster, run the following command:
#
# ```shell
# $ pyflyte run --remote neptune_example.py train_wf
# ```
