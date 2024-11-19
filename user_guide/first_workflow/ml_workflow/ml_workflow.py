"""Run ML workflow

1. Get dataset from openml
2. Train a HistGradientBoostingClassifier model
3. Evaluate the model and create a confusion matrix deck

Usage:
```
pip install -r guides/01_getting_started/ml_workflow/requirements.txt
union run --remote guides/01_getting_started/ml_workflow/ml_workflow.py main
```
"""

import base64
import io
import html
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from flytekit import task, workflow, ImageSpec, current_context, Deck, Resources

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier


image = ImageSpec(
    requirements=Path(__file__).parent / "requirements.txt",
)

enable_cache = True


@task(
    cache=enable_cache,
    cache_version="4",
    container_image=image,
    requests=Resources(cpu="2", mem="2Gi"),
)
def get_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = fetch_openml(name="penguins", version=1, as_frame=True)
    train_dataset, test_dataset = train_test_split(
        dataset.frame, random_state=0, stratify=dataset.target
    )
    return train_dataset, test_dataset


@task(
    container_image=image,
    requests=Resources(cpu="3", mem="2Gi"),
)
def train_model(dataset: pd.DataFrame, max_bins: int) -> BaseEstimator:
    X_train, y_train = dataset.drop("species", axis="columns"), dataset["species"]
    hist = HistGradientBoostingClassifier(
        random_state=0, max_bins=max_bins, categorical_features="from_dtype"
    )
    return hist.fit(X_train, y_train)


@task(
    container_image=image,
    enable_deck=True,
    requests=Resources(cpu="2", mem="2Gi"),
)
def evaluate_model(model: BaseEstimator, dataset: pd.DataFrame) -> float:
    ctx = current_context()

    X_test, y_test = dataset.drop("species", axis="columns"), dataset["species"]
    y_pred = model.predict(X_test)

    # Plot confusion matrix in deck
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)

    metrics_deck = Deck("Metrics")
    metrics_deck.append(_convert_fig_into_html(fig))

    # Add classification report
    report = html.escape(classification_report(y_test, y_pred))
    html_report = dedent(
        f"""\
    <h2>Classification report</h2>
    <pre>{report}</pre>"""
    )
    metrics_deck.append(html_report)

    ctx.decks.insert(0, metrics_deck)

    return accuracy_score(y_test, y_pred)


@workflow
def main(max_bins: int = 64) -> float:
    train, test = get_dataset()
    model = train_model(dataset=train, max_bins=max_bins)
    return evaluate_model(model=model, dataset=test)


def _convert_fig_into_html(fig: mpl.figure.Figure) -> str:
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    img_base64 = base64.b64encode(img_buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_base64}" alt="Rendered Image" />'
