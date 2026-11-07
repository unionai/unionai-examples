"""Parse speculators' training metrics and render the live training report.

`speculators.train` logs one line per optimizer step, such as
`train.loss=1.234, train.full_acc_0=0.61, ..., epoch=0, lr=5.00e-05, global_step=12`,
and one line of validation metrics per epoch. `History` keeps both.
"""

import re
from dataclasses import dataclass, field

import report_ui as ui

_PAIR = re.compile(r"([A-Za-z0-9_./-]+)=(-?[0-9.]+(?:e[-+]?\d+)?)")


TITLES = {"eagle3": "EAGLE-3", "dflash": "DFlash", "dspark": "DSpark"}


_POSITION = re.compile(r"position_(\d+)_acc(_epoch)?")


def _normalize(key: str) -> str:
    """Block drafters log `position_1_acc`; EAGLE-3 logs `full_acc_0`. Use the latter."""
    match = _POSITION.fullmatch(key)
    if match:
        return f"full_acc_{int(match.group(1)) - 1}{match.group(2) or ''}"
    return key


@dataclass
class History:
    method: str
    config: dict[str, str]
    train: list[dict[str, float]] = field(default_factory=list)
    val: list[dict[str, float]] = field(default_factory=list)
    checkpoint: str = ""

    def parse(self, line: str) -> bool:
        """Record a metric line. Returns True if the line held metrics."""
        if "=" not in line or ("global_step=" not in line and "val" not in line):
            return False
        pairs = {k: float(v) for k, v in _PAIR.findall(line)}
        if not pairs:
            return False
        flat = {_normalize(k.split(".")[-1].split("/")[-1]): v for k, v in pairs.items()}
        if any(k.startswith(("val.", "val/")) for k in pairs):
            self.val.append(flat)
        elif "global_step" in flat:
            self.train.append(flat)
        else:
            return False
        return True


def _series(rows: list[dict[str, float]], key: str, x: str = "global_step") -> list[tuple[float, float]]:
    return [(r.get(x, i), r[key]) for i, r in enumerate(rows) if key in r]


def _smooth(points: list[tuple[float, float]], window: int = 10) -> list[tuple[float, float]]:
    out = []
    for i in range(len(points)):
        chunk = points[max(0, i - window + 1) : i + 1]
        out.append((points[i][0], sum(y for _, y in chunk) / len(chunk)))
    return out


def render(history: History, done: bool = False) -> str:
    train = history.train
    last = train[-1] if train else {}
    # Block drafters report up to 15 positions; the first few tell the story.
    positions = sorted(
        int(k.rsplit("_", 1)[1]) for k in last if re.fullmatch(r"full_acc_\d+", k)
    )[:6]
    status = f"finished · kept checkpoint {history.checkpoint}" if done else "training"

    accuracy = {
        f"position {p}": _smooth(_series(train, f"full_acc_{p}")) for p in positions
    }
    val_rows = [
        [int(r.get("epoch", i)), f"{r.get('loss_epoch', float('nan')):.4f}"]
        + [f"{r.get(f'full_acc_{p}_epoch', float('nan')):.3f}" for p in positions]
        for i, r in enumerate(history.val)
    ]
    return ui.page(
        ui.hero(
            f"{TITLES.get(history.method, history.method)} draft model training",
            "Online mode: vLLM computes Qwen3-8B hidden states on demand; the trainer never stores them.",
            ["speculators", status],
        ),
        ui.tiles({
            "step": f"{int(last.get('global_step', 0)):,}",
            "epoch": f"{int(last.get('epoch', 0))}",
            "loss": f"{last.get('loss', float('nan')):.3f}",
            **{f"accuracy, token {p + 1}": f"{last.get(f'full_acc_{p}', 0):.1%}" for p in positions[:3]},
        }),
        ui.charts(
            ui.line_chart("Training loss (smoothed)", {"loss": _smooth(_series(train, "loss"))}, y_label="steps"),
            ui.line_chart("Draft accuracy by position (smoothed)", accuracy, y_label="steps"),
        ),
        ui.section(
            "What the accuracy curves mean",
            lede=(
                "Position 0 is the first drafted token, position 1 the second, and so on. Each is "
                "accepted only if every earlier one was, so acceptance length is roughly "
                "1 + acc0 + acc0·acc1 + …. Later positions are always lower."
            ),
        ),
        ui.section(
            "Validation by epoch",
            ui.table(["epoch", "loss"] + [f"acc {p}" for p in positions], val_rows) if val_rows else "<p class='lede'>Runs at the end of each epoch.</p>",
        ),
        ui.section("Configuration", ui.table(["setting", "value"], [[k, v] for k, v in history.config.items()], numeric=set())),
    )
