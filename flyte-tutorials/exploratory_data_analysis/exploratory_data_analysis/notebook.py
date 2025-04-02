# # Flyte Pipeline in One Jupyter Notebook
#
# In this example, we will implement a simple pipeline that takes hyperparameters, does EDA, feature engineering, and measures the Gradient
# Boosting model's performance using mean absolute error (MAE), all in one notebook.

# First, let's import the libraries we will use in this example.
import pathlib

from flytekit import Resources, kwtypes, workflow
from flytekitplugins.papermill import NotebookTask

# We define a `NotebookTask` to run the [Jupyter notebook](https://github.com/flyteorg/flytesnacks/blob/master/examples/exploratory_data_analysis/exploratory_data_analysis/supermarket_regression.ipynb).
#
#
# This notebook returns `mae_score` as the output.
nb = NotebookTask(
    name="pipeline-nb",
    notebook_path=str(pathlib.Path(__file__).parent.absolute() / "supermarket_regression.ipynb"),
    inputs=kwtypes(
        n_estimators=int,
        max_depth=int,
        max_features=str,
        min_samples_split=int,
        random_state=int,
    ),
    outputs=kwtypes(mae_score=float),
    requests=Resources(mem="500Mi"),
)


# Since a task need not be defined, we create a `workflow` and return the MAE score.


@workflow
def notebook_wf(
    n_estimators: int = 150,
    max_depth: int = 3,
    max_features: str = "sqrt",
    min_samples_split: int = 4,
    random_state: int = 2,
) -> float:
    output = nb(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )
    return output.mae_score


# We can now run the notebook locally.
#
if __name__ == "__main__":
    print(notebook_wf())
