"""Shared pieces for the Timestral tutorials: the container image, model loading,
and forecast-accuracy metrics."""

from __future__ import annotations

import functools
import os

import flyte

# {{docs-fragment models}}
# Timestral is a family of four checkpoints with the same architecture. All of them
# run on CPU; larger ones are more accurate and slower.
TIMESTRAL_MODELS = {
    "5M": "mistralai/Timestral-5M",
    "28M": "mistralai/Timestral-28M",
    "123M": "mistralai/Timestral-123M",
    "209M": "mistralai/Timestral-209M",
}

# Every Timestral forecast carries these 21 quantile levels: 0.01, 0.05 to 0.95 in
# steps of 0.05, and 0.99.
QUANTILE_LEVELS = (0.01, *(round(0.05 * k, 2) for k in range(1, 20)), 0.99)
# {{/docs-fragment models}}

# {{docs-fragment image}}
# Timestral installs from GitHub, pinned to a commit so every task in every run
# loads the same code. A CPU build of torch keeps the image small: Timestral does
# not need a GPU.
TIMESTRAL_COMMIT = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"

timestral_image = (
    flyte.Image.from_debian_base(name="timestral", python_version=(3, 12))
    .with_apt_packages("git")
    .with_pip_packages("torch", index_url="https://download.pytorch.org/whl/cpu")
    .with_pip_packages(
        f"timestral @ git+https://github.com/mistralai/timestral.git@{TIMESTRAL_COMMIT}",
        "numpy",
        "pandas",
        "pyarrow",
        "httpx",
    )
)
# {{/docs-fragment image}}


# {{docs-fragment load_forecaster}}
@functools.lru_cache(maxsize=4)
def load_forecaster(model_id: str):
    """Load a Timestral checkpoint once per process.

    The cache matters when a container serves many forecasts: an app replica, or a
    task environment whose containers are reused across tasks.
    """
    import torch
    from timestral import TimestralForecaster

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return TimestralForecaster.from_pretrained(model_id, device="cpu")


# {{/docs-fragment load_forecaster}}


# {{docs-fragment metrics}}
def mase(actual, median, history, season: int) -> float:
    """Mean absolute scaled error: forecast MAE divided by the in-sample MAE of a
    seasonal-naive forecast. Below 1.0 beats "same as last season"."""
    import numpy as np

    scale = np.nanmean(np.abs(history[season:] - history[:-season]))
    if not np.isfinite(scale) or scale == 0:
        return float("nan")
    return float(np.nanmean(np.abs(actual - median)) / scale)


def weighted_quantile_loss(actual, quantiles) -> float:
    """Weighted quantile loss (WQL), the probabilistic metric used by GIFT-Eval:
    pinball loss summed over quantile levels, normalized by the actuals.

    ``actual`` is (horizon,), ``quantiles`` is (horizon, 21).
    """
    import numpy as np

    levels = np.asarray(QUANTILE_LEVELS)
    diff = actual[:, None] - quantiles
    pinball = np.maximum(levels * diff, (levels - 1) * diff)
    denom = np.nansum(np.abs(actual))
    if denom == 0:
        return float("nan")
    return float(2 * np.nansum(pinball) / denom / len(levels))


# {{/docs-fragment metrics}}


def fan_chart_svg(
    history,
    quantiles,
    actual=None,
    *,
    width: int = 640,
    height: int = 200,
    title: str = "",
) -> str:
    """Render history, the 10-90% and 1-99% forecast bands, the median, and
    (optionally) the actuals as a small inline SVG for Flyte reports."""
    import numpy as np

    history = np.asarray(history, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    lo99, lo90, med, hi90, hi99 = (q[:, i] for i in (0, 2, 10, 18, 20))
    n_hist, n_fc = len(history), len(med)
    values = [history, lo99, hi99] + (
        [np.asarray(actual, dtype=float)] if actual is not None else []
    )
    ymin = float(np.nanmin(np.concatenate(values)))
    ymax = float(np.nanmax(np.concatenate(values)))
    if ymax == ymin:
        ymax = ymin + 1.0
    pad = 24

    def x(i):
        return pad + (width - 2 * pad) * i / max(1, n_hist + n_fc - 1)

    def y(v):
        return height - pad - (height - 2 * pad) * (v - ymin) / (ymax - ymin)

    def path(xs, ys):
        return " ".join(
            f"{'M' if k == 0 else 'L'}{x(i):.1f},{y(v):.1f}"
            for k, (i, v) in enumerate(zip(xs, ys))
        )

    def band(lo, hi):
        idx = list(range(n_hist, n_hist + n_fc))
        upper = " ".join(f"L{x(i):.1f},{y(v):.1f}" for i, v in zip(idx, hi))
        lower = " ".join(
            f"L{x(i):.1f},{y(v):.1f}" for i, v in zip(reversed(idx), reversed(lo))
        )
        return f"M{x(idx[0]):.1f},{y(hi[0]):.1f} {upper} {lower} Z"

    fc_idx = range(n_hist, n_hist + n_fc)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="font:11px sans-serif;background:#fff">',
        f'<text x="{pad}" y="14" fill="#333">{title}</text>',
        f'<path d="{band(lo99, hi99)}" fill="#c7d2fe" stroke="none"/>',
        f'<path d="{band(lo90, hi90)}" fill="#818cf8" fill-opacity="0.6" stroke="none"/>',
        f'<path d="{path(range(n_hist), history)}" fill="none" stroke="#475569" stroke-width="1"/>',
        f'<path d="{path(fc_idx, med)}" fill="none" stroke="#3730a3" stroke-width="1.5"/>',
    ]
    if actual is not None:
        parts.append(
            f'<path d="{path(fc_idx, actual)}" fill="none" stroke="#dc2626" stroke-width="1"/>'
        )
    parts.append(
        f'<line x1="{x(n_hist):.1f}" x2="{x(n_hist):.1f}" y1="{pad}" y2="{height - pad}" '
        'stroke="#94a3b8" stroke-dasharray="3,3"/></svg>'
    )
    return "".join(parts)
