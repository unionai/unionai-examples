# # Credit Default Prediction on GPUs with XGBoost & NVIDIA RAPIDS
#
# In this tutorial, we will use NVIDIA RAPIDS `cudf` DataFrame library for preprocessing
# data and XGBoost, an optimized gradient boosting library, for credit default prediction.
# We'll learn how to declare NVIDIA  `A100` for our training function and `ImageSpec`
# for specifying our python dependencies.

# ## Declaring workflow dependencies
#
# First, we start by importing all the dependencies that is required by this workflow:

import os
import gc
from pathlib import Path
from typing import Tuple

import fsspec
from flytekit import task, workflow, current_context, Resources, ImageSpec, Deck
from flytekit.types.file import FlyteFile
from flytekit.extras.accelerators import A100

# We download the credit default dataset and return the dataset as two `FlyteFile`s. With
# `cache=True` the results of this task will be cached. This means that downstream
# tasks that use the data will use the cached version and do not need to re-download the
# data again. For this example, we use a subset of the data so the task completes quickly.


@task(cache=True, cache_version="v5", requests=Resources(cpu="2", mem="4Gi"))
def download_data() -> Tuple[FlyteFile, FlyteFile]:
    working_dir = Path(current_context().working_directory)

    train_data = working_dir / "train.parquet"
    train_label_data = working_dir / "train_label.parquet"
    _download_file(
        "https://github.com/thomasjpfan/credit-data/raw/main/train/part_0.parquet",
        train_data,
    )
    _download_file(
        "https://github.com/thomasjpfan/credit-data/raw/main/train_labels.parquet",
        train_label_data,
    )

    return train_data, train_label_data


# ## Defining Python Dependencies
#
# We use `flytekit`'s `ImageSpec` to specify the python packages that is required by the
# XGBoost training task and preprocessing with `cudf`.

credit_default_image = ImageSpec(
    name="credit-default",
    packages=[
        "cudf-cu12==24.6.*",
        "xgboost==2.1.0",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "union==0.1.56",
        "cuml-cu12==24.6.*",
        "scikit-learn==1.4.*",
    ],
    python_version="3.11",
    pip_index="https://pypi.nvidia.com",
    registry=os.environ.get("IMAGE_SPEC_REGISTRY"),
)


# ## Training with XGBoost with NVIDIA's A100
#
# Next, we define our XGBoost training job using `flytekit`'s `@task`. We set the
# `container_image` to `credit_default_image` so that this task runs with the required python
# packages. We easily configure this task to run on `A100`s by setting `accelerator=A100`.
# The training task uses RAPID's `cudf` DataFrame library for preprocessing and passed to
# `XGBoost` for training.


@task(
    requests=Resources(gpu="1"),
    accelerator=A100,
    container_image=credit_default_image,
    cache=True,
    cache_version="v0",
)
def train_xgboost(
    train_data: FlyteFile, train_labels: FlyteFile
) -> Tuple[FlyteFile, float]:
    _configure_nvidia_libs()

    import cudf

    train_data.download()
    train_labels.download()

    train_df = cudf.read_parquet(train_data.path)
    train_labels = cudf.read_parquet(train_labels.path)
    train = train_df.merge(train_labels, on="customer_ID", how="left")

    X_train, y_train, X_test, y_test = prepare_for_training(train)

    model = get_xgb_model()

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    ctx = current_context()
    working_dir = Path(ctx.working_directory)
    model_path = working_dir / "model.ubj"
    model.save_model(model_path)

    return model_path, model.best_score


# ## Plot feature importances
#
# XGBoost's feature importances represent the relative importance of each feature in
# making predictions. In this next task, we use a Flyte Deck to plot the feature
# importances with matplotlib:

matplotlib_image = ImageSpec(
    "plot-xgboost",
    packages=[
        "xgboost==2.1.0",
        "matplotlib==3.9.1",
        "union==0.1.56",
        "scikit-learn==1.4.*",
    ],
)


@task(container_image=matplotlib_image, enable_deck=True)
def plot_feature_importances(model: FlyteFile):
    import matplotlib.pyplot as plt
    from xgboost import plot_importance

    model.download()

    xgb_model = get_xgb_model()
    xgb_model.load_model(model.path)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_importance(xgb_model, max_num_features=15, ax=ax)

    importances_deck = Deck("Feature Importance", _fig_to_html(fig))
    decks = current_context().decks
    decks.insert(0, importances_deck)


# ## Full Workflow
#
# Finally, we define the workflow that calls `download_data` and passes it's output
# to `train_xgboost`. We run the workflow by:
#
# ```bash
# union run --remote credit_default.py credit_default_wf
# ````


@workflow
def credit_default_wf() -> Tuple[FlyteFile, float]:
    train_data, train_labels = download_data()
    model, best_score = train_xgboost(train_data=train_data, train_labels=train_labels)
    plot_feature_importances(model=model)
    return model, best_score


# ## Appendix
#
# The following are helper functions used by our Flyte tasks. They include functions
# that download files, configure nvidia library, preprocessing data, and configuring
# the XGBoost model.


def _download_file(src, dest):
    """Download file from src to dest."""
    with fsspec.open(src, mode="rb") as r:
        with dest.open("wb") as w:
            w.write(r.read())


def _configure_nvidia_libs():
    """Configures NVIDIA RAPIDS .so files to be accessible by `cudf`."""
    import site
    import os

    site_packages = Path(site.getsitepackages()[0])

    libs_to_link = [
        (
            site_packages / "nvidia" / "cuda_runtime" / "lib" / "libcudart.so.12",
            "/usr/lib/libcudart.so",
        ),
        (
            site_packages / "nvidia" / "cuda_nvrtc" / "lib" / "libnvrtc.so.12",
            "/usr/lib/libnvrtc.so.12",
        ),
        (
            site_packages
            / "nvidia"
            / "cuda_nvrtc"
            / "lib"
            / "libnvrtc-builtins.so.12.5",
            "/usr/lib/libnvrtc-builtins.so.12.5",
        ),
    ]

    for src, dst in libs_to_link:
        os.symlink(src, dst)


def preprocess(df):
    """Preprocessing dataframe by dropping duplicates based on cid."""
    return (
        df.sort_values(["cid", "S_2"])
        .drop_duplicates("cid", keep="last")
        .sort_values("cid")
        .reset_index(drop=True)
    )


def prepare_for_training(train):
    """Split data into training and validation."""
    train["cid"], _ = train.customer_ID.factorize()
    mask = train["cid"] % 4 == 0

    tr, va = train.loc[~mask], train.loc[mask]

    tr = preprocess(tr)
    va = preprocess(va)

    # prepare for training
    not_used = [
        i for i in tr.columns if i in ["cid", "target", "S_2"] or tr[i].dtype == "O"
    ]
    not_used += [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]

    X_train = tr.drop(not_used, axis=1)
    y_train = tr["target"]

    X_test = va.drop(not_used, axis=1)
    y_test = va["target"]

    for i in X_train.columns:
        X_train[i] = X_train[i].astype("float32")
        X_test[i] = X_test[i].astype("float32")

    del train, tr, va
    gc.collect()
    return X_train, y_train, X_test, y_test


def amex_metric_np(target, preds) -> float:
    """Custom metric based on the evaluation metric from
    https://www.kaggle.com/competitions/amex-default-prediction/overview."""
    import numpy as np

    indices = np.argsort(preds)[::-1]
    preds, target = preds[indices], target[indices]

    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04
    d = np.sum(target[four_pct_mask]) / np.sum(target)

    weighted_target = target * weight
    lorentz = (weighted_target / weighted_target.sum()).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    n_pos = np.sum(target)
    n_neg = target.shape[0] - n_pos
    gini_max = 10 * n_neg * (n_pos + 20 * n_neg - 19) / (n_pos + 20 * n_neg)

    g = gini / gini_max
    return 0.5 * (g + d)


def get_xgb_model():
    """Create XGBoost model with amex_metric evaluation."""
    import xgboost as xgb

    max_depth = 7
    num_trees = 1000
    min_child_weight = 50
    early_stop = xgb.callback.EarlyStopping(
        rounds=10, maximize=True, metric_name="amex_metric_np", data_name="validation_0"
    )
    model = xgb.XGBClassifier(
        tree_method="hist",
        objective="binary:logistic",
        max_depth=max_depth,
        n_estimators=num_trees,
        min_child_weight=min_child_weight,
        eval_metric=amex_metric_np,
        callbacks=[early_stop],
        device="cuda",
    )
    return model


def _fig_to_html(fig) -> str:
    """Convert matplotlib figure to html."""
    import io
    import base64

    fig_bytes = io.BytesIO()
    fig.savefig(fig_bytes, format="jpg")
    fig_bytes.seek(0)
    image_base64 = base64.b64encode(fig_bytes.read()).decode()
    return f'<img src="data:image/png;base64,{image_base64}" alt="Rendered Image" />'
