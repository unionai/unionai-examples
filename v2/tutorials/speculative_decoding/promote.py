# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
#    "flyteplugins-vllm==2.13.0",
#    "httpx>=0.27",
# ]
# ///
"""Union only: version draft models and their training traces as artifacts, with cards.

`retrain` returns whichever drafter won the gate. Publishing it as a versioned artifact,
one artifact per family, gives every promotion a name, lineage back to the run that produced
it, a model card with its benchmark results, and a one-line rollback. The traces it trained
on are published as a data artifact with a data card.

    python promote.py --method dflash --run <training-run> [--benchmarks <compare-run>]
    python promote.py --traces <agent-job-run> ...       # publish agent traces
    python promote.py --method dflash --serve            # serve the latest drafter
    python promote.py --method dflash --rollback <version>
"""

import json
import sys
from collections import Counter

import flyte
import flyte.app
from flyte import artifacts
from flyte.io import Dir, File
from flyte.remote import App

import report_ui as ui
from config import TARGET_MODEL, cpu_image
from recipes import RECIPES
from serve import APPS

TRACES_ARTIFACT = "qwen3-8b-agent-traces"
FAMILY = {"eagle3": "EAGLE-3", "dflash": "DFlash", "dspark": "DSpark"}


def drafter_artifact(method: str) -> str:
    return f"qwen3-8b-{method}-drafter"

promote_env = flyte.TaskEnvironment(
    name="specdec-promote", image=cpu_image, resources=flyte.Resources(cpu=1, memory="2Gi")
)


# {{docs-fragment model-card}}
def model_card(method: str, summary: dict | None, source_run: str) -> str:
    """A model card: what the drafter is, how it was trained, and what it measured."""
    recipe = RECIPES[method]
    parts = [
        ui.hero(
            f"Qwen3-8B {FAMILY[method]} draft model",
            f"A {FAMILY[method]} drafter for Qwen3-8B. Outputs are unchanged; only decoding speed differs.",
            [FAMILY[method], "speculators", "vLLM", f"{recipe.num_speculative_tokens} speculative tokens"],
        ),
        ui.section(
            "Details",
            ui.table(
                ["field", "value"],
                [
                    ["target model", TARGET_MODEL],
                    ["initialized from", recipe.init_from or "random weights"],
                    ["target layers", ", ".join(map(str, recipe.target_layers))],
                    ["produced by run", source_run or "unknown"],
                    ["serve with", '{"method": "%s", "num_speculative_tokens": %d}' % (method, recipe.num_speculative_tokens)],
                ],
                numeric=set(),
            ),
        ),
    ]
    if summary:
        for workload, data in summary["workloads"].items():
            rows = data["rows"]
            parts.append(
                ui.section(
                    f"Benchmark: {workload}",
                    ui.charts(
                        ui.bar_chart("Speedup at concurrency 1", {n: r["speedup"]["1"] for n, r in rows.items()}, "{:.2f}x", 1.0),
                        ui.bar_chart(
                            "Acceptance length",
                            {n: r["acceptance_length"] for n, r in rows.items() if n != "baseline"},
                        ),
                    ),
                    ui.table(
                        ["config", "acceptance length", "speedup c=1", "matches baseline before divergence"],
                        [
                            [n, f"{r['acceptance_length']:.2f}", f"{r['speedup']['1']:.2f}x", f"{r['prefix_agreement']:.0%}"]
                            for n, r in rows.items()
                        ],
                    ),
                )
            )
    parts.append(
        ui.section(
            "Limitations",
            lede=(
                "Trained for Qwen3-8B with thinking disabled. Retrain after any change to the target's "
                "weights, quantization, or chat template. Speedups shrink as serving concurrency rises."
            ),
        )
    )
    return ui.page(*parts)


@promote_env.task(produces_artifacts=True)
async def publish_drafter(
    drafter: Dir, method: str = "eagle3", summary: File | None = None, source_run: str = ""
) -> Dir:
    """Publish a drafter as the next version of its family's artifact, with a model card."""
    results = None
    if summary is not None:
        with open(await summary.download()) as f:
            results = json.load(f)
    card = await artifacts.Card.create_from.aio(
        content=model_card(method, results, source_run), format="html", card_type="model"
    )
    return artifacts.new(
        drafter,
        artifacts.Metadata.create_model_metadata(
            name=drafter_artifact(method),
            description=f"{FAMILY[method]} draft model for {TARGET_MODEL}",
            card=card,
            framework="PyTorch",
            architecture=FAMILY[method],
            task="speculative-decoding",
            serial_format="safetensors",
            attrs={"target_model": TARGET_MODEL, "method": method, "source_run": source_run},
        ),
    )
# {{/docs-fragment model-card}}


# {{docs-fragment data-card}}
@promote_env.task(produces_artifacts=True)
async def publish_traces(traces: list[File]) -> File:
    """Merge agent trace files into one data artifact, with a data card."""
    rows = []
    for f in traces:
        with open(await f.download()) as fh:
            rows.extend(json.loads(line) for line in fh)
    path = "/tmp/traces.jsonl"
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    tools = Counter(
        c["function"]["name"]
        for r in rows
        for m in r["messages"]
        if m["role"] == "assistant"
        for c in m.get("tool_calls", [])
    )
    turns = [sum(m["role"] == "assistant" for m in r["messages"]) for r in rows]
    card_html = ui.page(
        ui.hero(
            "Agent traces for draft head training",
            f"Tool-calling episodes served by {TARGET_MODEL}. Every assistant turn is target output.",
            ["OpenAI chat format", "tools included", "jsonl"],
        ),
        ui.tiles({
            "episodes": f"{len(rows):,}",
            "source files": f"{len(traces):,}",
            "tool calls": f"{sum(tools.values()):,}",
            "distinct tools": f"{len(tools):,}",
        }),
        ui.charts(
            ui.histogram("Assistant turns per episode", turns, bins=6),
            ui.bar_chart("Most called tools", dict(tools.most_common(8)), "{:.0f}"),
        ),
        ui.section(
            "Schema",
            ui.pretty_json({"messages": "[OpenAI chat messages, with tool_calls]", "tools": "[OpenAI tool schemas]"}),
        ),
        ui.section("Example", ui.chat(rows[0]["messages"]) if rows else ""),
    )
    card = await artifacts.Card.create_from.aio(content=card_html, format="html", card_type="data")
    return artifacts.new(
        await File.from_local(path),
        artifacts.Metadata(
            name=TRACES_ARTIFACT,
            description="Agent traces collected from the speculative decoding app",
            card=card,
            kind="data",
            attrs={"episodes": str(len(rows)), "target_model": TARGET_MODEL},
        ),
    )
# {{/docs-fragment data-card}}


# {{docs-fragment artifact-app}}
def artifact_app(method: str, version: str | None = None):
    """The family's app, serving its drafter artifact: the latest version, or a pinned one."""
    app = APPS[method]
    return app.clone_with(
        name=f"{app.name}-promoted",
        draft_model_path=flyte.app.ArtifactValue(
            name=drafter_artifact(method), type="directory", version=version
        ),
    )


def roll_back(method: str, version: str) -> None:
    """Serve an earlier drafter. The target model and everything else stay the same."""
    pinned = artifact_app(method, version)
    flyte.deploy(pinned)
    App.get(name=pinned.name).activate()
# {{/docs-fragment artifact-app}}


def _blob_uri(literals) -> str:
    (literal,) = literals.values()
    return literal.scalar.blob.uri


if __name__ == "__main__":
    import flyte.remote as remote

    flyte.init_from_config()
    method = sys.argv[sys.argv.index("--method") + 1] if "--method" in sys.argv else "eagle3"
    if "--run" in sys.argv:
        # The training run whose drafter to publish, and optionally the compare.py run
        # whose benchmark results go on its model card.
        name = sys.argv[sys.argv.index("--run") + 1]
        drafter = Dir.from_existing_remote(_blob_uri(remote.Run.get(name).output_literals()))
        summary = None
        if "--benchmarks" in sys.argv:
            bench_run = remote.Run.get(sys.argv[sys.argv.index("--benchmarks") + 1])
            summary = File.from_existing_remote(_blob_uri(bench_run.output_literals()))
        run = flyte.run(publish_drafter, drafter=drafter, method=method, summary=summary, source_run=name)
        print(run.url)
    elif "--traces" in sys.argv:
        names = sys.argv[sys.argv.index("--traces") + 1 :]
        files = [File.from_existing_remote(_blob_uri(remote.Run.get(n).output_literals())) for n in names]
        print(flyte.run(publish_traces, traces=files).url)
    elif "--rollback" in sys.argv:
        roll_back(method, sys.argv[sys.argv.index("--rollback") + 1])
    else:
        print(flyte.with_servecontext(activate_timeout=1800.0).serve(artifact_app(method)).url)
