# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "flyteplugins-hitl>=2.0.0",
#    "fastapi",
#    "uvicorn",
#    "python-multipart",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment setup}}
import flyte
import flyteplugins.hitl as hitl

# The task environment must declare hitl.env as a dependency.
# This makes the HITL web app available during task execution.
env = flyte.TaskEnvironment(
    name="hitl-workflow",
    image=flyte.Image.from_debian_base(name="hitl").with_pip_packages(
        "flyteplugins-hitl>=2.0.0",
        "fastapi",
        "uvicorn",
        "python-multipart",
    ),
    resources=flyte.Resources(cpu="1", memory="512Mi"),
    depends_on=[hitl.env],
)
# {{/docs-fragment setup}}


# {{docs-fragment automated-task}}
@env.task(report=True)
async def analyze_data(dataset: str) -> dict:
    """Automated task that produces a result requiring human review."""
    # Simulate analysis
    result = {
        "dataset": dataset,
        "row_count": 142857,
        "anomalies_detected": 3,
        "confidence": 0.87,
    }
    await flyte.report.replace.aio(
        f"Analysis complete: {result['anomalies_detected']} anomalies detected "
        f"(confidence: {result['confidence']:.0%})"
    )
    await flyte.report.flush.aio()
    return result
# {{/docs-fragment automated-task}}


# {{docs-fragment hitl-event}}
@env.task(report=True)
async def request_human_review(analysis: dict) -> bool:
    """Pause and ask a human whether to proceed with the flagged records."""
    event = await hitl.new_event.aio(
        "review_decision",
        data_type=bool,
        scope="run",
        prompt=(
            f"Analysis found {analysis['anomalies_detected']} anomalies "
            f"with {analysis['confidence']:.0%} confidence. "
            "Approve for downstream processing? (true/false)"
        ),
    )
    approved: bool = await event.wait.aio()
    return approved
# {{/docs-fragment hitl-event}}


# {{docs-fragment main}}
@env.task(report=True)
async def main(dataset: str = "s3://my-bucket/data.parquet") -> str:
    analysis = await analyze_data(dataset=dataset)

    approved = await request_human_review(analysis=analysis)

    if approved:
        return "Processing approved — continuing pipeline."
    else:
        return "Processing rejected by reviewer — pipeline halted."


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
    print(r.outputs())
# {{/docs-fragment main}}
