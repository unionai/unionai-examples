# /// script
# requires-python = "==3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "openai>=1.0.0",
#    "pydantic>=2.0.0",
# ]
# main = "report_batch_pipeline"
# ///

"""
Resilient Agentic Report Generator

A report generation pipeline that demonstrates Flyte 2.0's advanced features:
- Reusable environments with ReusePolicy for cost efficiency
- Traced LLM calls with @flyte.trace for checkpointing and recovery
- Retry strategies for API resilience
- Agentic refinement loops with flyte.group for observability
- Parallel output formatting with asyncio.gather
"""

# {{docs-fragment imports}}
import asyncio
import json
import os
import tempfile
from datetime import timedelta

import flyte
from flyte.io import Dir

from prompts import (
    CRITIC_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    REVISER_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    Critique,
)
# {{/docs-fragment imports}}


# {{docs-fragment mock-mode}}
# Set to True to use mock responses instead of real LLM calls
MOCK_MODE = True
# {{/docs-fragment mock-mode}}

# {{docs-fragment reusable-env}}
# Reusable environment for tasks that make many LLM calls in a loop.
# The ReusePolicy keeps containers warm, reducing cold start latency for iterative work.
llm_env = flyte.TaskEnvironment(
    name="llm-worker",
    secrets=[] if MOCK_MODE else [flyte.Secret(key="openai-api-key", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "unionai-reuse>=0.1.10",
        "openai>=1.0.0",
        "pydantic>=2.0.0",
    ),
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    reusable=flyte.ReusePolicy(
        replicas=2,              # Keep 2 container instances ready
        concurrency=4,           # Allow 4 concurrent tasks per container
        scaledown_ttl=timedelta(minutes=5),   # Wait 5 min before scaling down
        idle_ttl=timedelta(minutes=30),       # Shut down after 30 min idle
    ),
    cache="auto",
)
# {{/docs-fragment reusable-env}}

# {{docs-fragment driver-env}}
# Standard environment for orchestration tasks that don't need container reuse.
# depends_on declares that this environment's tasks call tasks in llm_env.
driver_env = flyte.TaskEnvironment(
    name="driver",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "pydantic>=2.0.0",
    ),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    depends_on=[llm_env],
)
# {{/docs-fragment driver-env}}


# {{docs-fragment mock-responses}}
MOCK_REPORT = """# The Impact of Large Language Models on Software Development

## Introduction

Large Language Models (LLMs) are transforming how software is built. From code
completion to automated documentation, AI coding assistants are becoming
essential tools in the modern developer's toolkit.

## Key Changes in Developer Workflows

### Code Generation and Completion
AI assistants can now generate boilerplate code, suggest implementations, and
complete complex functions based on natural language descriptions.

### Documentation and Explanation
LLMs excel at explaining code, generating documentation, and helping developers
understand unfamiliar codebases quickly.

### Debugging and Code Review
AI tools can identify potential bugs, suggest fixes, and provide code review
feedback, augmenting human reviewers.

## Impact on Productivity

Studies suggest AI coding assistants can improve developer productivity by
30-50% for certain tasks, particularly repetitive coding and documentation.

## Skills for the Future

Developers increasingly need skills in:
- Prompt engineering
- AI tool integration
- Critical evaluation of AI-generated code

## Conclusion

LLMs are not replacing developers but augmenting their capabilities. The most
effective developers will be those who learn to collaborate with AI tools."""

MOCK_CRITIQUE_GOOD = """{
    "score": 8,
    "strengths": [
        "Clear structure with logical sections",
        "Covers key aspects of the topic",
        "Professional tone throughout"
    ],
    "improvements": [
        "Could include more specific statistics",
        "Add real-world examples of AI tools"
    ],
    "summary": "A solid overview that meets publication standards."
}"""

MOCK_CRITIQUE_NEEDS_WORK = """{
    "score": 6,
    "strengths": [
        "Good topic coverage",
        "Clear writing style"
    ],
    "improvements": [
        "Add more specific examples and data",
        "Expand the productivity section",
        "Include potential challenges and limitations"
    ],
    "summary": "Good foundation but needs more depth and concrete examples."
}"""

MOCK_SUMMARY = """This report examines how Large Language Models are reshaping software
development practices. Key findings include significant productivity gains of 30-50%
for certain tasks, fundamental changes to developer workflows around code generation
and documentation, and the emergence of new skills like prompt engineering.

The conclusion emphasizes that AI tools augment rather than replace developers,
with the most successful practitioners being those who effectively collaborate
with these new capabilities."""
# {{/docs-fragment mock-responses}}


# {{docs-fragment traced-llm-call}}
@flyte.trace
async def call_llm(prompt: str, system: str, json_mode: bool = False) -> str:
    """
    Make an LLM call with automatic checkpointing.

    The @flyte.trace decorator provides:
    - Automatic caching of results for identical inputs
    - Recovery from failures without re-running successful calls
    - Full observability in the Flyte UI

    Args:
        prompt: The user prompt to send
        system: The system prompt defining the LLM's role
        json_mode: Whether to request JSON output

    Returns:
        The LLM's response text
    """
    # Use mock responses for testing without API keys
    if MOCK_MODE:
        import asyncio
        await asyncio.sleep(0.5)  # Simulate API latency

        if "critique" in prompt.lower() or "critic" in system.lower():
            # Return good score if draft has been revised (contains revision marker)
            if "[REVISED]" in prompt:
                return MOCK_CRITIQUE_GOOD
            return MOCK_CRITIQUE_NEEDS_WORK
        elif "summary" in system.lower():
            return MOCK_SUMMARY
        elif "revis" in system.lower():
            # Return revised version with marker
            return MOCK_REPORT.replace("## Introduction", "[REVISED]\n\n## Introduction")
        else:
            return MOCK_REPORT

    from openai import AsyncOpenAI

    client = AsyncOpenAI()

    kwargs = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2000,
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content
# {{/docs-fragment traced-llm-call}}


# {{docs-fragment generate-draft}}
@flyte.trace
async def generate_initial_draft(topic: str) -> str:
    """
    Generate the initial report draft.

    The @flyte.trace decorator provides checkpointing - if the task fails
    after this completes, it won't re-run on retry.

    Args:
        topic: The topic to write about

    Returns:
        The initial draft in markdown format
    """
    print(f"Generating initial draft for topic: {topic}")

    prompt = f"Write a comprehensive report on the following topic:\n\n{topic}"
    draft = await call_llm(prompt, GENERATOR_SYSTEM_PROMPT)

    print(f"Generated initial draft ({len(draft)} characters)")
    return draft
# {{/docs-fragment generate-draft}}


# {{docs-fragment critique-content}}
@flyte.trace
async def critique_content(draft: str) -> Critique:
    """
    Critique the current draft and return structured feedback.

    Uses Pydantic models to parse the LLM's JSON response into
    a typed object for reliable downstream processing.

    Args:
        draft: The current draft to critique

    Returns:
        Structured critique with score, strengths, and improvements
    """
    print("Critiquing current draft...")

    response = await call_llm(
        f"Please critique the following report:\n\n{draft}",
        CRITIC_SYSTEM_PROMPT,
        json_mode=True,
    )

    # Parse the JSON response into our Pydantic model
    critique_data = json.loads(response)
    critique = Critique(**critique_data)

    print(f"Critique score: {critique.score}/10")
    print(f"Strengths: {len(critique.strengths)}, Improvements: {len(critique.improvements)}")

    return critique
# {{/docs-fragment critique-content}}


# {{docs-fragment revise-content}}
@flyte.trace
async def revise_content(draft: str, improvements: list[str]) -> str:
    """
    Revise the draft based on critique feedback.

    Args:
        draft: The current draft to revise
        improvements: List of specific improvements to address

    Returns:
        The revised draft
    """
    print(f"Revising draft to address {len(improvements)} improvements...")

    improvements_text = "\n".join(f"- {imp}" for imp in improvements)
    prompt = f"""Please revise the following report to address these improvements:

IMPROVEMENTS NEEDED:
{improvements_text}

CURRENT DRAFT:
{draft}"""

    revised = await call_llm(prompt, REVISER_SYSTEM_PROMPT)

    print(f"Revision complete ({len(revised)} characters)")
    return revised
# {{/docs-fragment revise-content}}


# {{docs-fragment refinement-loop}}
@llm_env.task(retries=3)
async def refine_report(
    topic: str,
    max_iterations: int = 3,
    quality_threshold: int = 8,
) -> str:
    """
    Iteratively refine a report until it meets the quality threshold.

    This task runs in a reusable container because it makes multiple LLM calls
    in a loop. The traced helper functions provide checkpointing, so if the
    task fails mid-loop, completed LLM calls won't be re-run on retry.

    Args:
        topic: The topic to write about
        max_iterations: Maximum refinement cycles (default: 3)
        quality_threshold: Minimum score to accept (default: 8)

    Returns:
        The final refined report
    """
    # Generate initial draft
    draft = await generate_initial_draft(topic)

    # Iterative refinement loop
    for i in range(max_iterations):
        with flyte.group(f"refinement_{i + 1}"):
            # Get critique
            critique = await critique_content(draft)

            # Check if we've met the quality threshold
            if critique.score >= quality_threshold:
                print(f"Quality threshold met at iteration {i + 1}!")
                print(f"Final score: {critique.score}/10")
                break

            # Revise based on feedback
            print(f"Score {critique.score} < {quality_threshold}, revising...")
            draft = await revise_content(draft, critique.improvements)
    else:
        print(f"Reached max iterations ({max_iterations})")

    return draft
# {{/docs-fragment refinement-loop}}


# {{docs-fragment format-functions}}
@flyte.trace
async def format_as_markdown(content: str) -> str:
    """Format the report as clean markdown."""
    # Content is already markdown, but we could add TOC, metadata, etc.
    return f"""---
title: Generated Report
date: {__import__('datetime').datetime.now().isoformat()}
---

{content}
"""


@flyte.trace
async def format_as_html(content: str) -> str:
    """Convert the report to HTML."""
    # Simple markdown to HTML conversion
    import re

    html = content
    # Convert headers
    html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
    html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
    html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)
    # Convert bold/italic
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
    # Convert paragraphs
    html = re.sub(r"\n\n", r"</p><p>", html)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Generated Report</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }}
        h1, h2, h3 {{ color: #333; }}
        p {{ line-height: 1.6; }}
    </style>
</head>
<body>
<p>{html}</p>
</body>
</html>
"""


@flyte.trace
async def generate_summary(content: str) -> str:
    """Generate an executive summary of the report."""
    return await call_llm(content, SUMMARY_SYSTEM_PROMPT)
# {{/docs-fragment format-functions}}


# {{docs-fragment parallel-formatting}}
@llm_env.task
async def format_outputs(content: str) -> Dir:
    """
    Generate multiple output formats in parallel.

    Uses asyncio.gather to run all formatting operations concurrently,
    maximizing efficiency when each operation is I/O-bound.

    Args:
        content: The final report content

    Returns:
        Directory containing all formatted outputs
    """
    print("Generating output formats in parallel...")

    with flyte.group("formatting"):
        # Run all formatting operations in parallel
        markdown, html, summary = await asyncio.gather(
            format_as_markdown(content),
            format_as_html(content),
            generate_summary(content),
        )

    # Write outputs to a directory
    output_dir = tempfile.mkdtemp()

    with open(os.path.join(output_dir, "report.md"), "w") as f:
        f.write(markdown)

    with open(os.path.join(output_dir, "report.html"), "w") as f:
        f.write(html)

    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary)

    print(f"Created outputs in {output_dir}")
    return await Dir.from_local(output_dir)
# {{/docs-fragment parallel-formatting}}


# {{docs-fragment batch-pipeline}}
@driver_env.task
async def report_batch_pipeline(
    topics: list[str],
    max_iterations: int = 3,
    quality_threshold: int = 8,
) -> list[Dir]:
    """
    Generate reports for multiple topics in parallel.

    This is where ReusePolicy shines: with N topics, each going through
    up to max_iterations refinement cycles, the reusable container pool
    handles potentially N × 7 LLM calls efficiently without cold starts.

    Args:
        topics: List of topics to write about
        max_iterations: Maximum refinement cycles per topic
        quality_threshold: Minimum quality score to accept

    Returns:
        List of directories, each containing a report's formatted outputs
    """
    print(f"Starting batch pipeline for {len(topics)} topics...")

    # Fan out: refine all reports in parallel
    # Each refine_report makes 2-7 LLM calls, all hitting the reusable pool
    with flyte.group("refine_all"):
        reports = await asyncio.gather(*[
            refine_report(topic, max_iterations, quality_threshold)
            for topic in topics
        ])

    print(f"All {len(reports)} reports refined, formatting outputs...")

    # Fan out: format all reports in parallel
    with flyte.group("format_all"):
        outputs = await asyncio.gather(*[
            format_outputs(report)
            for report in reports
        ])

    print(f"Batch pipeline complete! Generated {len(outputs)} reports.")
    return outputs
# {{/docs-fragment batch-pipeline}}


# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config()

    # Multiple topics to generate reports for
    topics = [
        "The Impact of Large Language Models on Software Development",
        "Edge Computing: Bringing AI to IoT Devices",
        "Quantum Computing: Current State and Near-Term Applications",
        "The Rise of Rust in Systems Programming",
        "WebAssembly: The Future of Browser-Based Applications",
    ]

    print(f"Submitting batch run for {len(topics)} topics...")
    import sys
    sys.stdout.flush()

    # Run the batch pipeline - this will generate all reports in parallel,
    # with the reusable container pool handling 5 topics × ~7 LLM calls each
    run = flyte.run(
        report_batch_pipeline,
        topics=topics,
        max_iterations=3,
        quality_threshold=8,
    )
    print(f"Batch report generation run URL: {run.url}")
    sys.stdout.flush()
    print("Waiting for pipeline to complete (Ctrl+C to skip)...")
    try:
        run.wait()
        print(f"Pipeline complete! Outputs: {run.outputs()}")
    except KeyboardInterrupt:
        print(f"\nSkipped waiting. Check status at: {run.url}")
# {{/docs-fragment main}}
