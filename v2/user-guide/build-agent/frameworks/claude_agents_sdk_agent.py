# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte>=2.0.0",
#     "flyteplugins-codegen[agent]",
# ]
# main = "log_analysis_agent_workflow"
# ///

"""Claude Agent SDK on Flyte via the flyteplugins-codegen AutoCoderAgent.

``AutoCoderAgent(backend="claude")`` runs an autonomous Claude Agent SDK agent
that generates a solution + tests and iterates until the tests pass, executing
each test run in an isolated Flyte sandbox.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

# {{docs-fragment all}}
import flyte
from flyte.io import File
from flyte.sandbox import sandbox_environment

from flyteplugins.codegen import AutoCoderAgent

agent = AutoCoderAgent(
    name="log-parser-agent",
    model="claude-sonnet-4-5-20250929",
    backend="claude",  # use the Claude Agent SDK (requires ANTHROPIC_API_KEY)
)

env = flyte.TaskEnvironment(
    name="claude-agents-sdk-example",
    secrets=[flyte.Secret(key="ANTHROPIC_API_KEY")],
    resources=flyte.Resources(cpu=2, memory="5Gi"),
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-codegen[agent]"),
    depends_on=[sandbox_environment],  # required: lets the agent spin up sandboxes
)


@env.task
async def analyze_logs(prompt: str, log_file: File) -> dict[str, str | int | bool]:
    """Let the Claude agent generate + test a log parser, then run it on the sample."""
    result = await agent.generate.aio(
        prompt=prompt,
        schema=(
            "Output JSON schema for report_json: "
            '{"total_lines": int, "by_level": {"INFO": int, "WARN": int, "ERROR": int}, '
            '"error_messages": ["list of error message strings"]}'
        ),
        constraints=[
            "Must handle all log levels: INFO, WARN, ERROR",
            "Must not crash on malformed lines - skip them",
            "filter_level controls minimum severity: INFO shows all, WARN shows WARN+ERROR, ERROR shows only ERROR",
        ],
        samples={"log_file": log_file},
        inputs={"filter_level": str},
        outputs={"report_json": str, "total_errors": int, "has_critical_errors": bool},
    )

    if not result.success:
        return {"error": result.error or "generation failed", "attempts": result.attempts}

    # Run the generated, tested solution on the sample data.
    report_json, total_errors, has_critical_errors = await result.run.aio(filter_level="WARN")
    return {
        "report_json": report_json,
        "total_errors": total_errors,
        "has_critical_errors": has_critical_errors,
    }
# {{/docs-fragment all}}


@env.task(cache="auto")
def create_log_file() -> File:
    """Create a sample log file for processing."""
    log_content = (
        "2024-01-15 08:23:01 INFO  [auth] User login successful user_id=1001\n"
        "2024-01-15 08:23:15 ERROR [db] Connection timeout after 30s host=db-primary\n"
        "2024-01-15 08:24:05 ERROR [api] POST /api/v1/orders 500 error='inventory check failed'\n"
        "2024-01-15 08:25:00 WARN  [auth] Rate limit approaching ip=192.168.1.50\n"
        "2024-01-15 08:26:00 INFO  [auth] User login successful user_id=1002\n"
    )
    log_path = Path(tempfile.gettempdir()) / "app.log"
    log_path.write_text(log_content)
    return File.from_local_sync(str(log_path))


@env.task
async def log_analysis_agent_workflow(
    prompt: str = (
        "Parse application log files (format: TIMESTAMP LEVEL [component] message key=value...) "
        "and produce a structured health report. Only include lines at or above filter_level "
        "severity. Set has_critical_errors to true if any ERROR lines exist. Output report_json "
        "as a JSON string."
    ),
) -> dict[str, str | int | bool]:
    log_file = create_log_file()
    return await analyze_logs(prompt=prompt, log_file=log_file)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(log_analysis_agent_workflow)
    print(f"Run URL: {run.url}")
