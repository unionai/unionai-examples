# /// script
# requires-python = "==3.12"
# dependencies = [
#    "pydantic>=2.0.0",
# ]
# ///

"""
Prompt templates and Pydantic models for the report generator.

This module defines the system prompts for each stage of the report generation
pipeline and the structured output schemas for LLM responses.
"""

# {{docs-fragment imports}}
from pydantic import BaseModel, Field
# {{end-fragment}}

# {{docs-fragment critique-model}}
class Critique(BaseModel):
    """Structured critique response from the LLM."""

    score: int = Field(
        ge=1,
        le=10,
        description="Quality score from 1-10, where 10 is publication-ready",
    )
    strengths: list[str] = Field(
        description="List of strengths in the current draft",
    )
    improvements: list[str] = Field(
        description="Specific improvements needed",
    )
    summary: str = Field(
        description="Brief summary of the critique",
    )
# {{end-fragment}}


# {{docs-fragment system-prompts}}
GENERATOR_SYSTEM_PROMPT = """You are an expert report writer. Generate a well-structured,
informative report on the given topic. The report should include:

1. An engaging introduction that sets context
2. Clear sections with descriptive headings
3. Specific facts, examples, or data points where relevant
4. A conclusion that summarizes key takeaways

Write in a professional but accessible tone. Use markdown formatting for structure.
Aim for approximately 500-800 words."""

CRITIC_SYSTEM_PROMPT = """You are a demanding but fair editor reviewing a report draft.
Evaluate the report on these criteria:

- Clarity: Is the writing clear and easy to follow?
- Structure: Is it well-organized with logical flow?
- Depth: Does it provide sufficient detail and insight?
- Accuracy: Are claims supported and reasonable?
- Engagement: Is it interesting to read?

Provide your response as JSON matching this schema:
{
    "score": <1-10 integer>,
    "strengths": ["strength 1", "strength 2", ...],
    "improvements": ["improvement 1", "improvement 2", ...],
    "summary": "brief overall assessment"
}

Be specific in your feedback. A score of 8+ means the report is ready for publication."""

REVISER_SYSTEM_PROMPT = """You are an expert editor revising a report based on feedback.
Your task is to improve the report by addressing the specific improvements requested
while preserving its strengths.

Guidelines:
- Address each improvement point specifically
- Maintain the original voice and style
- Keep the same overall structure unless restructuring is requested
- Preserve any content that was praised as a strength
- Ensure the revised version is cohesive and flows well

Return only the revised report in markdown format, no preamble or explanation."""

SUMMARY_SYSTEM_PROMPT = """Create a concise executive summary (2-3 paragraphs) of the
following report. Capture the key points and main takeaways. Write in a professional
tone suitable for busy executives who need the essential information quickly."""
# {{end-fragment}}
