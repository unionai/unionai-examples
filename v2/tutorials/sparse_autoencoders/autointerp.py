"""Automated interpretability: what Claude sees when it explains a feature, and when it tests one.

The method follows Bills et al. (2023) and Paulo et al. (2024). One call shows Claude a
feature's top-activating contexts and a random sample of its other firings, with the
activating tokens marked, and asks for a short explanation. A second call, which never sees those contexts, gets the explanation and a
shuffled mix of held-out contexts: some where the feature fires, some where it does not.
Claude says which ones fire. The score is the balanced accuracy, so 0.5 is chance.
"""

import json
import random

# Bump when the prompts, the schemas, the excerpt format, or the scoring change. label_features
# salts its cache with this, so a change here re-labels instead of returning cached labels.
VERSION = "top12-sampled8/2"

# {{docs-fragment prompts}}
EXPLAIN_SYSTEM = """\
You are an interpretability researcher. You study one feature of a sparse autoencoder trained
on a small language model's residual stream. A feature is a direction that is active on some
tokens and silent on the rest. You will see the text excerpts where the feature is most active,
then a random sample of other excerpts where it fires, with every token where it fires at a tenth of its peak or more marked as <<token>>, and the
peak activation of each excerpt on a 0-10 scale. Find what the marked tokens have in common, in the token itself and in its
context. Be specific: name the words, the syntax, or the topic. If it fires on a token only
in some contexts, say which. Your explanation will be used to predict, for unseen excerpts,
whether the feature fires anywhere in them."""

EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "notes": {"type": "string", "description": "One or two sentences of reasoning about the pattern."},
        "explanation": {"type": "string", "description": "The explanation, at most 20 words."},
    },
    "required": ["notes", "explanation"],
    "additionalProperties": False,
}

DETECT_SYSTEM = """\
You test an explanation of one feature of a sparse autoencoder trained on a small language
model. You will get the explanation and numbered text excerpts. Some excerpts contain a token
where the feature fires and the rest do not. Using only the explanation, list the numbers of
the excerpts in which the feature fires."""

DETECT_SCHEMA = {
    "type": "object",
    "properties": {"active": {"type": "array", "items": {"type": "integer"}}},
    "required": ["active"],
    "additionalProperties": False,
}
# {{/docs-fragment prompts}}


def _clean(tok: str) -> str:
    return tok.replace("\n", "\\n")


def marked(example: dict, peak: float) -> str:
    """One excerpt with every token that fires at 1/10 of the peak or more marked, then its peak on a 0-10 scale."""
    pairs = zip(example["tokens"], example["acts"])
    text = "".join(f"<<{_clean(t)}>>" if a >= 0.1 * peak else _clean(t) for t, a in pairs)
    return f"{text}    [peak {round(10 * max(example['acts']) / peak)}]"


def explain_prompt(feature: dict) -> str:
    peak = feature["max_act"]
    top = "\n".join(f"{i + 1}. {marked(ex, peak)}" for i, ex in enumerate(feature["top"]))
    n = len(feature["top"])
    rest = "\n".join(f"{n + i + 1}. {marked(ex, peak)}" for i, ex in enumerate(feature["sampled"]))
    if feature.get("sampled_clear", True):
        which = "A random sample of the other excerpts where it fires clearly (at least a quarter of its peak)"
    else:  # too few clear firings: feature_examples fell back to the next-strongest excerpts
        which = "A random sample of its next-strongest excerpts, some of them weak firings"
    return (f"The excerpts where one feature fires most strongly:\n\n{top}\n\n"
            f"{which}:\n\n{rest}\n\n"
            "Explain the feature. Cover what it does at moderate strength too, not only at its peak.")


def detect_set(feature: dict, seed: int) -> list[dict]:
    """The feature's held-out excerpts in a fixed random order."""
    test = list(feature["test"])
    random.Random(seed).shuffle(test)
    return test


def detect_prompt(explanation: str, test: list[dict]) -> str:
    lines = "\n".join(f"{i + 1}. {''.join(_clean(t) for t in ex['tokens'])}" for i, ex in enumerate(test))
    return f"Explanation: {explanation}\n\nExcerpts:\n\n{lines}\n\nWhich excerpts contain a token where the feature fires?"


# {{docs-fragment score}}
def detection_score(answer_json: str, test: list[dict]) -> float:
    """Balanced accuracy of Claude's predictions against where the feature really fired."""
    predicted = {int(i) - 1 for i in json.loads(answer_json)["active"]}
    on = [i for i, ex in enumerate(test) if ex["active"]]
    off = [i for i, ex in enumerate(test) if not ex["active"]]
    hit = sum(i in predicted for i in on) / max(len(on), 1)
    reject = sum(i not in predicted for i in off) / max(len(off), 1)
    return (hit + reject) / 2
# {{/docs-fragment score}}
