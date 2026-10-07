# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.6.0",
# ]
# main = "decision_demo"
# params = "num_decisions=1000000"
# ///
"""One million decisions in five minutes with an open-weights decision model.

Three stages, two reusable environments:

1. ``build_template_bank`` — one cached call per (language, weakness class) to
   an open-weights code LLM behind the Union LLM Gateway. It writes a handful of
   short snippet templates with placeholder identifiers.
2. ``synthesize_snippets`` — CPU replicas instantiate the templates into as many
   unique snippets as the run asks for. Each snippet is clean or carries exactly
   one planted weakness, so the dataset comes with ground truth.
3. ``assess_snippets`` — an open-weights *decision model*: a zero-shot NLI encoder
   that never generates text. Every snippet gets one seven-way choice question
   ("no weakness" or one of six classes); each hypothesis is one forward pass,
   so seven decisions per snippet. Each GPU replica has eight T4s and one shared
   ``DynamicBatcher`` per GPU, fed by every concurrent task on the replica.

``decision_demo`` is the driver. It fans out per chunk, chaining synthesis into
assessment so the GPUs start as soon as the first chunk exists, then merges the
chunk statistics into a report. ``num_decisions`` sets the scale.
"""

from __future__ import annotations

import asyncio
import html
import math
import os
import random
import re
import socket
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache, partial

import flyte
import flyte.report
from flyte.extras import DynamicBatcher

# ─────────────────────────────────────────────────────────────────────────────
# Models and environments
# ─────────────────────────────────────────────────────────────────────────────

# {{docs-fragment models}}
# The template bank is written by Claude Haiku behind the Union LLM Gateway (an
# OpenAI-compatible endpoint; model ids are "<provider>/<model>"). It is called
# 21 times per run, not once per snippet. Any model on the gateway works
# here, for example "qwen38-27b-vllm/qwen38-27b" for an open-weights one.
LLM_GATEWAY_URL = "https://llm-gateway.apps.demo.hosted.unionai.cloud/v1"
SYNTH_MODEL = "anthropic/claude-haiku-4-5-20251001"

# The decision model is a zero-shot NLI encoder: it never generates text. One
# forward pass over (code, hypothesis) says how strongly the code entails the
# hypothesis. The base variant (~150M parameters) is three times cheaper than
# the large one and runs comfortably in fp16 on a T4.
DECISION_MODEL = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
# {{/docs-fragment models}}

# {{docs-fragment image}}
# unionai-reuse installs the actor bridge that reusable containers require.
synth_image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "httpx", "unionai-reuse>=0.1.11"
)

gpu_image = (
    flyte.Image.from_debian_base(python_version=(3, 12))
    .with_pip_packages("torch", "transformers", "unionai-reuse>=0.1.11")
    .with_env_vars({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# The driver only needs flyte and the standard library.
driver_image = flyte.Image.from_debian_base(python_version=(3, 12))
# {{/docs-fragment image}}

# {{docs-fragment envs}}
# ``concurrency`` is how many task invocations may run on a replica at once.
#
# The synthesizer is cheap: template instantiation is pure Python and the one
# gateway-bound task is cached. One small CPU replica is enough.
synth_env = flyte.TaskEnvironment(
    name="code-synthesizer",
    image=synth_image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    secrets=[flyte.Secret(key="DEMO_GATEWAY_ANTHROPIC_API_KEY", as_env_var="LLM_GATEWAY_API_KEY")],
    reusable=flyte.ReusePolicy(
        replicas=1,
        concurrency=16,
        idle_ttl=600,
        scaledown_ttl=120,
    ),
)

# The decision model is GPU-bound. Each replica has eight T4s and keeps one
# model copy and one batcher per GPU; every task on the replica round-robins
# its records across the eight batchers, so all GPUs see one continuous stream
# of work. Two fixed replicas give sixteen GPUs; a fixed count (rather than an
# autoscaling range) keeps both warm between runs, because a scaled-down
# replica costs minutes of node provisioning on the next run.
#
# idle_ttl keeps replicas warm between runs of the same code, so the first run
# pays for node provisioning and later ones do not. It is deliberately not
# longer: a code change deploys a new environment, and the old one keeps its
# GPU nodes until its idle_ttl expires, which can starve the new one on a
# cluster with few T4 nodes.
decision_env = flyte.TaskEnvironment(
    name="vulnerability-decision-model",
    image=gpu_image,
    resources=flyte.Resources(cpu=8, memory="32Gi", gpu="T4:8", disk="20Gi"),
    reusable=flyte.ReusePolicy(
        replicas=2,
        concurrency=16,
        idle_ttl=600,
        scaledown_ttl=120,
    ),
)

# The driver fans out into both environments, so it declares them as
# dependencies: their images are built and registered alongside it.
driver_env = flyte.TaskEnvironment(
    name="decision-demo-driver",
    image=driver_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[synth_env, decision_env],
)
# {{/docs-fragment envs}}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset spec
# ─────────────────────────────────────────────────────────────────────────────

# {{docs-fragment spec}}
WEAKNESSES: dict[str, str] = {
    "sql_injection": "SQL injection: untrusted input is concatenated or formatted into a SQL query",
    "command_injection": "OS command injection: untrusted input reaches a shell or subprocess command string",
    "path_traversal": "path traversal: an untrusted path segment is joined into a filesystem path without a containment check",
    "hardcoded_secret": "a hard-coded credential, API token, or private key in the source",
    "insecure_deserialization": "insecure deserialization: untrusted bytes are handed to pickle, yaml.load, or an equivalent unsafe loader",
    "ssrf": "server-side request forgery: the server fetches an untrusted URL or host without validation",
}

LANGUAGES = ["python", "javascript", "go"]

# Placeholders the template bank uses; instantiation swaps them for identifiers.
PLACEHOLDERS = ("__A__", "__B__", "__C__")
STRING_PLACEHOLDER = "__S__"
IDENTIFIERS = [
    "user", "order", "invoice", "session", "record", "payload", "report", "account",
    "ticket", "profile", "item", "entry", "config", "job", "event", "token", "asset",
    "batch", "customer", "document", "message", "shipment", "vendor", "widget",
]
WORDS = ["alpha", "north", "blue", "summit", "delta", "harbor", "ember", "quartz", "lumen", "cedar"]


@dataclass
class Template:
    language: str
    planted: str  # a key of WEAKNESSES, or "none" for a clean template
    code: str  # short snippet with __A__/__B__/__C__ identifier and __S__ string placeholders


@dataclass
class TemplateBank:
    templates: list[Template]


@dataclass
class Snippet:
    snippet_id: str
    language: str
    planted: str
    code: str


def instantiate(template: Template, rng: random.Random) -> str:
    """Turn a template into a concrete snippet by filling its placeholders."""
    code = template.code
    for placeholder, name in zip(PLACEHOLDERS, rng.sample(IDENTIFIERS, len(PLACEHOLDERS))):
        code = code.replace(placeholder, name)
    return code.replace(STRING_PLACEHOLDER, f"{rng.choice(WORDS)}_{rng.randrange(100, 999)}")
# {{/docs-fragment spec}}


# ─────────────────────────────────────────────────────────────────────────────
# Report helpers shared by every stage
# ─────────────────────────────────────────────────────────────────────────────

REPORT_CSS = """
  body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 24px; color: #1d1d1f; }
  h1 { margin-bottom: 4px; } h2 { margin-top: 32px; } h3 { margin: 18px 0 8px; color: #3a3a3c; }
  p.lead { color: #6e6e73; margin-top: 0; }
  .cards { display: flex; flex-wrap: wrap; gap: 12px; }
  .card { background: #f5f5f7; border-radius: 10px; padding: 14px 18px; min-width: 150px; }
  .card .v { font-size: 26px; font-weight: 600; } .card .k { color: #6e6e73; font-size: 13px; }
  table { border-collapse: collapse; font-size: 14px; }
  th, td { padding: 6px 10px; border-bottom: 1px solid #e5e5ea; text-align: left; vertical-align: top; }
  small { color: #6e6e73; }
  .bar { height: 14px; border-radius: 3px; min-width: 2px; background: #ffb100; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 12px; }
  .tile { border: 1px solid #e5e5ea; border-radius: 10px; padding: 10px 12px; background: #fff; }
  .tile .meta { display: flex; gap: 8px; align-items: center; margin-bottom: 6px; font-size: 13px; color: #6e6e73; }
  .pill { border-radius: 10px; padding: 1px 8px; font-size: 12px; background: #e5e5ea; color: #1d1d1f; }
  .pill.lang { background: #dbe9ff; } .pill.clean { background: #d9f2d0; } .pill.planted { background: #fde2e2; }
  .pill.vulnerable { background: #fde2e2; } .pill.review { background: #fff8e1; }
  td.hit { background: #d9f2d0; font-weight: 600; } td.miss { background: #fde2e2; }
  tr.review td { background: #fff8e1; }
  tr.code td { border-bottom: 2px solid #e5e5ea; padding: 0 10px 8px; }
  pre { background: #f5f5f7; padding: 10px; border-radius: 6px; overflow-x: auto; font-size: 12px; margin: 0; max-height: 320px; }
"""


def _page(title: str, lead: str, body: str) -> str:
    return (
        f"<!doctype html><html><head><meta charset=\"utf-8\"><title>{html.escape(title)}</title>"
        f"<style>{REPORT_CSS}</style></head><body><h1>{html.escape(title)}</h1>"
        f'<p class="lead">{lead}</p>{body}</body></html>'
    )


def _cards(items: list[tuple[str, str]]) -> str:
    return '<div class="cards">' + "".join(
        f'<div class="card"><div class="v">{html.escape(v)}</div><div class="k">{html.escape(k)}</div></div>'
        for k, v in items
    ) + "</div>"


def _pill(text: str, kind: str = "") -> str:
    return f'<span class="pill {kind}">{html.escape(text.replace("_", " "))}</span>'


def _code_tile(meta: str, code: str) -> str:
    return f'<div class="tile"><div class="meta">{meta}</div><pre>{html.escape(code)}</pre></div>'


def _bar_table(header: tuple[str, str], counts: dict[str, int]) -> str:
    peak = max(counts.values(), default=1) or 1
    rows = "".join(
        f"<tr><td>{html.escape(k.replace('_', ' '))}</td><td>{n:,}</td>"
        f'<td><div class="bar" style="width:{int(100 * n / peak)}%"></div></td></tr>'
        for k, n in counts.items()
    )
    return f'<table><tr><th>{header[0]}</th><th>{header[1]}</th><th style="width:260px"></th></tr>{rows}</table>'


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: template bank from an open-weights LLM behind the gateway
# ─────────────────────────────────────────────────────────────────────────────

# {{docs-fragment bank_prompt}}
TEMPLATES_PER_CALL = 4


def _bank_messages(language: str, planted: str) -> list[dict[str, str]]:
    if planted == "none":
        requirement = (
            "Each snippet must be free of security weaknesses: parameterised queries, no shell "
            "strings built from input, no secrets in source, safe deserialization only."
        )
    else:
        requirement = (
            "Each snippet must contain exactly one security weakness of this kind, written the "
            f"way it appears in real code: {WEAKNESSES[planted]}."
        )
    return [
        {
            "role": "system",
            "content": (
                "You write short, realistic code snippet templates for a security training dataset. "
                "Reply with fenced code blocks only, one per snippet, and no prose."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Write {TEMPLATES_PER_CALL} different {language} snippets of 5 to 10 lines each, "
                f"each a small function that handles a request or a record. {requirement} "
                "Use the placeholders __A__, __B__ and __C__ wherever a function, variable, table "
                "or field name appears, and __S__ for any string literal, so the snippets can be "
                "instantiated with different names. Do not mention security, weaknesses, "
                "vulnerabilities, or these instructions anywhere."
            ),
        },
    ]


_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)  # some open models emit a reasoning preamble


def _templates_in(reply: str) -> list[str]:
    """Fenced code blocks in a reply, ignoring any reasoning preamble."""
    return [code.strip() for code in _FENCE.findall(_THINK.sub("", reply)) if code.strip()]
# {{/docs-fragment bank_prompt}}


# {{docs-fragment bank_task}}
# Process-level singletons: a reusable container keeps the Python process alive
# between task invocations, so the HTTP client (and its connection pool) is
# created once per replica and every task on the replica shares it.
_http_client = None
_in_flight: asyncio.Semaphore | None = None
MAX_IN_FLIGHT_PER_REPLICA = 8  # the gateway queues requests; more in flight only means longer waits


def _gateway():
    global _http_client, _in_flight
    if _http_client is None:
        import httpx

        _http_client = httpx.AsyncClient(
            base_url=LLM_GATEWAY_URL,
            headers={"Authorization": f"Bearer {os.environ['LLM_GATEWAY_API_KEY']}"},
            # Generous: a large self-hosted model behind a shared gateway can take minutes per reply.
            timeout=httpx.Timeout(900.0, connect=10.0),
        )
        _in_flight = asyncio.Semaphore(MAX_IN_FLIGHT_PER_REPLICA)
    return _http_client, _in_flight


@flyte.trace
async def _chat(messages: list[dict[str, str]], seed: int) -> str:
    """One chat completion through the gateway, with backoff on rate limits.
    ``flyte.trace`` records the call in the run's lineage and checkpoints its
    result, so a retried task replays completed calls instead of repeating them."""
    client, in_flight = _gateway()
    body = {"model": SYNTH_MODEL, "messages": messages, "temperature": 0.8, "max_tokens": 2000, "seed": seed}
    delay = 2.0
    for attempt in range(6):
        async with in_flight:
            response = await client.post("/chat/completions", json=body)
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 5:
            await asyncio.sleep(delay + random.random())
            delay *= 2
            continue
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    raise RuntimeError("unreachable")


def _render_bank_report(templates: list[Template]) -> str:
    by_language: dict[str, list[Template]] = defaultdict(list)
    for t in templates:
        by_language[t.language].append(t)
    per_class = Counter(t.planted for t in templates)
    body = _cards(
        [
            ("Templates", str(len(templates))),
            ("Languages", str(len(by_language))),
            ("Weakness classes", str(len([c for c in per_class if c != "none"]))),
            ("Clean templates", str(per_class.get("none", 0))),
            ("Model", SYNTH_MODEL.split("/")[-1]),
        ]
    )
    body += "<h2>Templates per class</h2>" + _bar_table(("Class", "Templates"), dict(per_class))
    for language in LANGUAGES:
        body += f"<h2>{html.escape(language)}</h2>"
        for planted in ["none", *WEAKNESSES]:
            tiles = [
                _code_tile(_pill(planted, "clean" if planted == "none" else "planted"), t.code)
                for t in by_language[language]
                if t.planted == planted
            ]
            if tiles:
                body += f"<h3>{html.escape(planted.replace('_', ' '))}</h3><div class=\"grid\">{''.join(tiles)}</div>"
    return _page(
        "Template bank",
        f"Snippet templates written by <code>{html.escape(SYNTH_MODEL)}</code> via the Union LLM Gateway. "
        "Placeholders <code>__A__</code>, <code>__B__</code>, <code>__C__</code> and <code>__S__</code> are "
        "filled in at synthesis time, so each template yields many distinct snippets.",
        body,
    )


@synth_env.task(retries=2, report=True)
async def build_template_bank() -> TemplateBank:
    """Ask the LLM for a few templates per (language, class). Not cached on
    purpose: it takes a few seconds with Haiku, sits outside the timed window,
    and rendering its report on every run is worth more than the saved calls."""
    jobs = [(language, planted) for language in LANGUAGES for planted in ["none", *WEAKNESSES]]
    templates: list[Template] = []
    # Two passes: pairs whose reply had no usable code block get one more try
    # with a different seed before the task gives up.
    for attempt in range(2):
        pending = [job for job in jobs if job not in {(t.language, t.planted) for t in templates}]
        if not pending:
            break
        replies = await asyncio.gather(
            *(
                _chat(_bank_messages(language, planted), seed=100 * attempt + jobs.index((language, planted)))
                for language, planted in pending
            )
        )
        for (language, planted), reply in zip(pending, replies):
            found = _templates_in(reply)
            if not found:
                print(f"[bank] no code block for {(language, planted)}; reply started: {reply[:160]!r}")
            templates += [Template(language=language, planted=planted, code=code) for code in found]
    covered = {(t.language, t.planted) for t in templates}
    missing = sorted(set(jobs) - covered)
    if missing:
        raise RuntimeError(f"the model returned no usable templates for {missing}")
    print(f"[bank] {len(templates)} templates across {len(covered)} (language, class) pairs")
    await flyte.report.replace.aio(_render_bank_report(templates), do_flush=True)
    return TemplateBank(templates=templates)
# {{/docs-fragment bank_task}}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: instantiate snippets
# ─────────────────────────────────────────────────────────────────────────────

# {{docs-fragment synth_task}}
PREVIEW_SNIPPETS = 12


def _render_snippets_report(snippets: list[Snippet], start: int) -> str:
    planted = Counter(s.planted for s in snippets)
    languages = Counter(s.language for s in snippets)
    body = _cards(
        [
            ("Snippets", f"{len(snippets):,}"),
            ("Id range", f"{snippets[0].snippet_id} to {snippets[-1].snippet_id}" if snippets else "none"),
            ("With a planted weakness", f"{len(snippets) - planted.get('none', 0):,}"),
            ("Clean", f"{planted.get('none', 0):,}"),
            ("Distinct code", f"{len({s.code for s in snippets}):,}"),
        ]
    )
    body += "<h2>Planted classes</h2>" + _bar_table(("Class", "Snippets"), dict(planted))
    body += "<h2>Languages</h2>" + _bar_table(("Language", "Snippets"), dict(languages))
    body += f"<h2>First {min(PREVIEW_SNIPPETS, len(snippets))} snippets</h2><div class=\"grid\">"
    for s in snippets[:PREVIEW_SNIPPETS]:
        meta = (
            f"<b>{html.escape(s.snippet_id)}</b> {_pill(s.language, 'lang')} "
            f"{_pill(s.planted, 'clean' if s.planted == 'none' else 'planted')}"
        )
        body += _code_tile(meta, s.code)
    body += "</div>"
    return _page(
        f"Synthesized snippets from #{start:,}",
        "Each snippet is a template from the bank with fresh identifiers and string literals. "
        "The planted class is the ground truth the decision model is scored against.",
        body,
    )


@synth_env.task(cache="auto", report=True)
async def synthesize_snippets(
    bank: TemplateBank, start: int, count: int, clean_fraction: float, seed: int
) -> list[Snippet]:
    """Instantiate ``count`` snippets with ids starting at ``start``. Everything
    is derived from ``seed`` and the index, so chunks are reproducible and
    independent of how the run was chunked."""
    by_key: dict[tuple[str, str], list[Template]] = defaultdict(list)
    for template in bank.templates:
        by_key[(template.language, template.planted)].append(template)

    snippets = []
    for i in range(start, start + count):
        rng = random.Random(f"{seed}:{i}")
        language = LANGUAGES[i % len(LANGUAGES)]
        planted = "none" if rng.random() < clean_fraction else rng.choice(list(WEAKNESSES))
        template = rng.choice(by_key[(language, planted)])
        snippets.append(
            Snippet(snippet_id=f"s{i:07d}", language=language, planted=planted, code=instantiate(template, rng))
        )
    await flyte.report.replace.aio(_render_snippets_report(snippets, start), do_flush=True)
    return snippets
# {{/docs-fragment synth_task}}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: decide with an open-weights decision model
# ─────────────────────────────────────────────────────────────────────────────

# {{docs-fragment questions}}
@dataclass(frozen=True)
class Question:
    """A typed question, in the style of a System One decision model.

    ``kind`` mirrors the question types such models answer: ``noul`` is a yes/no
    probability from a single hypothesis, ``choice`` picks one option from a list
    of hypotheses. Every hypothesis is one forward pass: one decision.
    """

    key: str
    kind: str  # "noul" | "choice"
    options: tuple[tuple[str, str], ...]  # (value, hypothesis)


# One seven-way choice: "no weakness" competes with the six classes, so every
# decision is relative to the alternatives. (A lone-hypothesis noul question is
# supported by ``_decide`` but its raw entailment probability is not calibrated
# against anything, so it is a poor fit for an NLI encoder.)
QUESTIONS: tuple[Question, ...] = (
    Question(
        key="weakness",
        kind="choice",
        options=(("none", "This code has no security weakness."),)
        + tuple((key, f"This code contains {description}.") for key, description in WEAKNESSES.items()),
    ),
)
DECISIONS_PER_SNIPPET = sum(len(q.options) for q in QUESTIONS)

# Decisions whose top option is below this go to a human-review bucket in the report.
REVIEW_THRESHOLD = 0.5

# Templates are 5 to 10 lines; these bounds keep every pair short, which is
# what makes a T4 forward pass cheap. A batch is padded to its longest pair,
# so the cap also bounds padding waste.
MAX_CODE_CHARS = 800
MAX_PAIR_TOKENS = 256
# Each batch is split into this many length-sorted sub-batches before the
# forward pass, so short pairs are not padded out to the longest one.
LENGTH_BUCKETS = 2
# {{/docs-fragment questions}}


# {{docs-fragment decision_records}}
@dataclass
class Hypothesis:
    """One (code, hypothesis) pair: the unit of work the batcher groups.

    A question with k options becomes k records. They are independent, so the
    batcher is free to pack them with records from any other question, snippet,
    or concurrent task on the replica.
    """

    snippet_id: str
    question: str  # a key of QUESTIONS
    option: str  # the option's value
    code: str
    text: str  # the hypothesis

    def estimate_cost(self) -> int:
        # Rough token count. The batcher calls this to fill a token budget per batch.
        return len(self.code) // 4 + 24


@dataclass
class Decision:
    """A typed answer with the full distribution behind it."""

    value: str
    confidence: float
    distribution: dict[str, float]
# {{/docs-fragment decision_records}}


# {{docs-fragment decision_singletons}}
@lru_cache(maxsize=None)
def _decision_model(device: str):
    """Load the NLI encoder onto one GPU, once per container lifetime.

    The model has one output head with two labels, ``entailment`` and
    ``not_entailment``. It never generates text: given a premise (the code) and
    a hypothesis (one option), a single forward pass says how strongly the
    premise entails the hypothesis.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DECISION_MODEL)
    # Weights stay fp32: ModernBERT keeps some buffers in fp32 and a pure fp16
    # load fails in its matmuls without FlashAttention. The forward pass runs
    # under fp16 autocast instead (fp16 rather than bf16, as the T4 is a Turing
    # card with no native bfloat16), so the heavy matmuls are still half precision.
    model = AutoModelForSequenceClassification.from_pretrained(DECISION_MODEL)
    model = model.to(device).eval()
    entail = model.config.label2id["entailment"]
    return tokenizer, model, entail


def _entailment_sync(device: str, batch: list[Hypothesis]) -> list[float]:
    """One forward pass for the whole batch on one GPU. Returns the entailment
    log-odds of each pair; the task turns a question's log-odds into a decision."""
    import torch

    tokenizer, model, entail = _decision_model(device)
    order = sorted(range(len(batch)), key=lambda i: len(batch[i].code))
    size = math.ceil(len(order) / LENGTH_BUCKETS)
    buckets = [order[i : i + size] for i in range(0, len(order), size)]
    log_odds = [0.0] * len(batch)
    for bucket in buckets:
        enc = tokenizer(
            [batch[i].code[:MAX_CODE_CHARS] for i in bucket],
            [batch[i].text for i in bucket],
            padding=True,
            truncation="only_first",  # trim the code, never the hypothesis
            max_length=MAX_PAIR_TOKENS,
            return_tensors="pt",
        ).to(device)
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")),
        ):
            logits = model(**enc).logits.float()
        for i, value in zip(bucket, (logits[:, entail] - logits[:, 1 - entail]).tolist()):
            log_odds[i] = value
    return log_odds


async def _entailment(device: str, batch: list[Hypothesis]) -> list[float]:
    # The forward pass is synchronous. Running it in a thread keeps the event
    # loop free for the batchers' aggregation loops and for the task runtime,
    # and lets the eight GPUs' batches run at the same time.
    return await asyncio.to_thread(_entailment_sync, device, batch)


_batchers: list[DynamicBatcher[Hypothesis, float]] | None = None
_batchers_lock = asyncio.Lock()


async def get_decision_batchers() -> list[DynamicBatcher[Hypothesis, float]]:
    """One batcher per GPU, created once per replica and shared by every
    concurrent assess task on it."""
    global _batchers
    async with _batchers_lock:
        if _batchers is None:
            import torch

            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
            batchers = []
            for device in devices:
                batcher = DynamicBatcher[Hypothesis, float](
                    process_fn=partial(_entailment, device),
                    # Hypothesis.estimate_cost() supplies the per-record cost (pair tokens).
                    target_batch_cost=24_000,
                    max_batch_size=256,
                    # A forward pass is fast, so dispatch quickly rather than wait for more.
                    batch_timeout_s=0.02,
                    max_queue_size=20_000,
                )
                await batcher.start()
                batchers.append(batcher)
            _batchers = batchers
            print(f"[decision] {len(batchers)} batchers on {socket.gethostname()}: {devices}")
    return _batchers
# {{/docs-fragment decision_singletons}}


# {{docs-fragment assessment}}
@dataclass
class Assessment:
    snippet_id: str
    language: str
    planted: str
    vulnerable: bool
    p_vulnerable: float
    weakness: str
    p_weakness: float
    needs_review: bool


@dataclass
class BatcherSnapshot:
    """What one GPU's batcher had done by the time this task returned."""

    host: str
    gpu: int
    total_completed: int
    total_batches: int
    avg_batch_size: float
    utilization: float


CALIBRATION_BINS = 10
MAX_SAMPLES = 24  # sample findings kept for the report across all chunks


@dataclass
class ChunkStats:
    """Everything the driver needs from a chunk, without the per-snippet rows.
    A million decisions must not become a million-row task output."""

    snippets: int = 0
    decisions: int = 0
    flagged: int = 0
    needs_review: int = 0
    detection_hits: int = 0  # "vulnerable?" agreed with whether a weakness was planted
    planted_total: int = 0
    weakness_hits: int = 0  # decided class matched the planted class
    confusion: dict[str, int] = field(default_factory=dict)  # "planted>decided" -> count
    bin_counts: list[int] = field(default_factory=lambda: [0] * CALIBRATION_BINS)
    bin_planted: list[int] = field(default_factory=lambda: [0] * CALIBRATION_BINS)
    samples: list[Assessment] = field(default_factory=list)
    sample_code: list[str] = field(default_factory=list)
    batchers: list[BatcherSnapshot] = field(default_factory=list)
    seconds: float = 0.0

    def add(self, a: Assessment) -> None:
        is_planted = a.planted != "none"
        self.snippets += 1
        self.decisions += DECISIONS_PER_SNIPPET
        self.flagged += a.vulnerable
        self.needs_review += a.needs_review
        self.detection_hits += a.vulnerable == is_planted
        self.planted_total += is_planted
        self.weakness_hits += is_planted and a.weakness == a.planted
        key = f"{a.planted}>{a.weakness}"
        self.confusion[key] = self.confusion.get(key, 0) + 1
        b = min(int(a.p_vulnerable * CALIBRATION_BINS), CALIBRATION_BINS - 1)
        self.bin_counts[b] += 1
        self.bin_planted[b] += is_planted

    def merge(self, other: ChunkStats) -> None:
        for name in ("snippets", "decisions", "flagged", "needs_review", "detection_hits", "planted_total", "weakness_hits"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for key, n in other.confusion.items():
            self.confusion[key] = self.confusion.get(key, 0) + n
        self.bin_counts = [x + y for x, y in zip(self.bin_counts, other.bin_counts)]
        self.bin_planted = [x + y for x, y in zip(self.bin_planted, other.bin_planted)]
        keep = max(0, MAX_SAMPLES - len(self.samples))
        self.samples += other.samples[:keep]
        self.sample_code += other.sample_code[:keep]
        # Snapshots are cumulative per GPU, so only the latest one per GPU matters.
        latest = {(b.host, b.gpu): b for b in self.batchers}
        for b in other.batchers:
            if (b.host, b.gpu) not in latest or b.total_completed > latest[(b.host, b.gpu)].total_completed:
                latest[(b.host, b.gpu)] = b
        self.batchers = sorted(latest.values(), key=lambda b: (b.host, b.gpu))
        self.seconds += other.seconds


def _decide(question: Question, log_odds: list[float]) -> Decision:
    """A noul question is the entailment probability of its single hypothesis.
    A choice question softmaxes its options' log-odds into one distribution,
    the same normalisation a zero-shot classification pipeline applies."""
    if question.kind == "noul":
        p = 1.0 / (1.0 + math.exp(-log_odds[0]))
        return Decision(value="yes" if p >= 0.5 else "no", confidence=max(p, 1.0 - p), distribution={"yes": p, "no": 1.0 - p})
    peak = max(log_odds)
    weights = [math.exp(x - peak) for x in log_odds]
    total = sum(weights)
    probs = [w / total for w in weights]
    best = max(range(len(probs)), key=probs.__getitem__)
    return Decision(
        value=question.options[best][0],
        confidence=probs[best],
        distribution={value: p for (value, _), p in zip(question.options, probs)},
    )


def _to_assessment(snippet: Snippet, decisions: dict[str, Decision]) -> Assessment:
    weakness = decisions["weakness"]
    p_vulnerable = 1.0 - weakness.distribution["none"]
    return Assessment(
        snippet_id=snippet.snippet_id,
        language=snippet.language,
        planted=snippet.planted,
        vulnerable=weakness.value != "none",
        p_vulnerable=p_vulnerable,
        weakness=weakness.value,
        p_weakness=weakness.confidence,
        # Abstain rather than guess: a low-confidence pick sends the snippet to a human.
        needs_review=weakness.confidence < REVIEW_THRESHOLD,
    )
# {{/docs-fragment assessment}}


# {{docs-fragment assess_task}}
SAMPLES_PER_CHUNK = 2


@decision_env.task(retries=2)
async def assess_snippets(snippets: list[Snippet]) -> ChunkStats:
    """Ask every question about every snippet. Each (code, hypothesis) pair is
    one record, round-robined across the replica's per-GPU batchers, which pack
    records from every concurrent task into token-budgeted forward passes."""
    batchers = await get_decision_batchers()
    started = time.perf_counter()

    futures: dict[tuple[str, str, str], asyncio.Future[float]] = {}
    for snippet in snippets:
        for question in QUESTIONS:
            for value, text in question.options:
                record = Hypothesis(
                    snippet_id=snippet.snippet_id, question=question.key, option=value, code=snippet.code, text=text
                )
                batcher = batchers[len(futures) % len(batchers)]
                futures[(snippet.snippet_id, question.key, value)] = await batcher.submit(record)

    log_odds = dict(zip(futures.keys(), await asyncio.gather(*futures.values())))

    stats = ChunkStats()
    for snippet in snippets:
        decisions = {
            q.key: _decide(q, [log_odds[(snippet.snippet_id, q.key, value)] for value, _ in q.options])
            for q in QUESTIONS
        }
        assessment = _to_assessment(snippet, decisions)
        stats.add(assessment)
        if len(stats.samples) < SAMPLES_PER_CHUNK:
            stats.samples.append(assessment)
            stats.sample_code.append(snippet.code)

    stats.seconds = time.perf_counter() - started
    host = socket.gethostname()
    stats.batchers = [
        BatcherSnapshot(
            host=host,
            gpu=i,
            total_completed=b.stats.total_completed,
            total_batches=b.stats.total_batches,
            avg_batch_size=b.stats.avg_batch_size,
            utilization=b.stats.utilization,
        )
        for i, b in enumerate(batchers)
    ]
    print(
        f"[assess] {stats.snippets} snippets, {stats.decisions} decisions in {stats.seconds:.1f}s "
        f"({stats.decisions / stats.seconds:.0f}/s) on {host}"
    )
    return stats
# {{/docs-fragment assess_task}}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: aggregate into a report
# ─────────────────────────────────────────────────────────────────────────────

# {{docs-fragment summary}}
@dataclass
class DemoSummary:
    num_decisions: int
    num_snippets: int
    wall_seconds: float
    decisions_per_second: float
    gpus: int
    replicas: int
    mean_gpu_utilization: float  # fraction of time each GPU's batcher spent in a forward pass
    mean_batch_size: float  # records per forward pass, averaged over GPUs
    num_flagged: int
    num_needs_review: int
    detection_accuracy: float  # "vulnerable?" vs. whether a weakness was planted
    weakness_accuracy: float  # decided class vs. the planted class, on planted samples


def summarise(stats: ChunkStats, wall_seconds: float) -> DemoSummary:
    gpus = {(b.host, b.gpu) for b in stats.batchers}
    n = len(stats.batchers) or 1
    return DemoSummary(
        num_decisions=stats.decisions,
        num_snippets=stats.snippets,
        wall_seconds=wall_seconds,
        decisions_per_second=stats.decisions / wall_seconds if wall_seconds else 0.0,
        gpus=len(gpus),
        replicas=len({host for host, _ in gpus}),
        mean_gpu_utilization=sum(b.utilization for b in stats.batchers) / n,
        mean_batch_size=sum(b.avg_batch_size for b in stats.batchers) / n,
        num_flagged=stats.flagged,
        num_needs_review=stats.needs_review,
        detection_accuracy=stats.detection_hits / stats.snippets if stats.snippets else 0.0,
        weakness_accuracy=stats.weakness_hits / stats.planted_total if stats.planted_total else 0.0,
    )
# {{/docs-fragment summary}}


def _pct(x: float) -> str:
    return f"{x:.0%}"


def _render_report(summary: DemoSummary, stats: ChunkStats) -> str:
    cards = [
        ("Decisions", f"{summary.num_decisions:,}"),
        ("Wall time", f"{summary.wall_seconds:.0f} s"),
        ("Decisions / second", f"{summary.decisions_per_second:,.0f}"),
        ("GPUs (replicas)", f"{summary.gpus} ({summary.replicas})"),
        ("GPU utilization", _pct(summary.mean_gpu_utilization)),
        ("Records / forward pass", f"{summary.mean_batch_size:.0f}"),
        ("Snippets", f"{summary.num_snippets:,}"),
        ("Flagged vulnerable", f"{summary.num_flagged:,}"),
        ("Needs human review", f"{summary.num_needs_review:,}"),
        ("Detection accuracy", _pct(summary.detection_accuracy)),
        ("Weakness-class accuracy", _pct(summary.weakness_accuracy)),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="v">{html.escape(v)}</div><div class="k">{html.escape(k)}</div></div>'
        for k, v in cards
    )

    classes = ["none", *WEAKNESSES]
    confusion_head = "".join(f"<th>{c.replace('_', ' ')}</th>" for c in classes)
    confusion_rows = ""
    for planted in classes:
        cells = ""
        for decided in classes:
            n = stats.confusion.get(f"{planted}>{decided}", 0)
            cls = "hit" if planted == decided and n else ("miss" if n else "")
            cells += f'<td class="{cls}">{n or ""}</td>'
        confusion_rows += f"<tr><th>{planted.replace('_', ' ')}</th>{cells}</tr>"

    calibration_rows = ""
    for b, (n, planted) in enumerate(zip(stats.bin_counts, stats.bin_planted)):
        lo, hi = b / CALIBRATION_BINS, (b + 1) / CALIBRATION_BINS
        rate = planted / n if n else 0.0
        calibration_rows += (
            f"<tr><td>{lo:.1f} to {hi:.1f}</td><td>{n:,}</td><td>{_pct(rate) if n else ''}</td>"
            f'<td><div class="bar" style="width:{int(100 * rate)}%"></div></td></tr>'
        )

    batcher_rows = "".join(
        f"<tr><td>{html.escape(s.host)}</td><td>{s.gpu}</td><td>{s.total_completed:,}</td>"
        f"<td>{s.total_batches:,}</td><td>{s.avg_batch_size:.0f}</td><td>{_pct(s.utilization)}</td></tr>"
        for s in stats.batchers
    )

    sample_rows = ""
    for a, code in zip(stats.samples, stats.sample_code):
        flag = "review" if a.needs_review else ("vulnerable" if a.vulnerable else "clean")
        sample_rows += (
            f'<tr class="{flag}"><td>{a.snippet_id}</td><td>{a.language}</td>'
            f"<td>{a.planted.replace('_', ' ')}</td><td>{_pct(a.p_vulnerable)}</td>"
            f"<td>{a.weakness.replace('_', ' ')} <small>{_pct(a.p_weakness)}</small></td><td>{flag}</td></tr>"
            f'<tr class="code"><td colspan="6"><details><summary>code</summary>'
            f"<pre>{html.escape(code)}</pre></details></td></tr>"
        )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Decision demo</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 24px; color: #1d1d1f; }}
  h1 {{ margin-bottom: 4px; }} h2 {{ margin-top: 32px; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 12px; }}
  .card {{ background: #f5f5f7; border-radius: 10px; padding: 14px 18px; min-width: 150px; }}
  .card .v {{ font-size: 26px; font-weight: 600; }} .card .k {{ color: #6e6e73; font-size: 13px; }}
  table {{ border-collapse: collapse; font-size: 14px; }}
  th, td {{ padding: 6px 10px; border-bottom: 1px solid #e5e5ea; text-align: left; vertical-align: top; }}
  small {{ color: #6e6e73; }}
  .bar {{ height: 14px; border-radius: 3px; min-width: 2px; background: #ffb100; }}
  td.hit {{ background: #d9f2d0; font-weight: 600; }} td.miss {{ background: #fde2e2; }}
  tr.review td {{ background: #fff8e1; }}
  tr.code td {{ border-bottom: 2px solid #e5e5ea; padding: 0 10px 8px; }}
  pre {{ background: #f5f5f7; padding: 10px; border-radius: 6px; overflow-x: auto; font-size: 12px; max-height: 320px; }}
</style></head><body>
<h1>{summary.num_decisions:,} decisions in {summary.wall_seconds:.0f} seconds</h1>
<p>{summary.num_snippets:,} synthetic snippets instantiated from templates written by <code>{SYNTH_MODEL}</code>
via the Union LLM Gateway, each judged by <code>{DECISION_MODEL}</code>, a decision model: one forward pass per
hypothesis, no generation, {DECISIONS_PER_SNIPPET} decisions per snippet. Accuracy is measured against the
weakness each template was asked to plant. Wall time covers synthesis and assessment; the cached template
bank is excluded.</p>
<div class="cards">{cards_html}</div>

<h2>Planted weakness vs. decided weakness</h2>
<table><tr><th>planted \\ decided</th>{confusion_head}</tr>{confusion_rows}</table>

<h2>Calibration of P(vulnerable)</h2>
<p>For a calibrated decision model the observed rate of planted weaknesses in each bin tracks the bin's probability.</p>
<table><tr><th>P(vulnerable)</th><th>Snippets</th><th>Actually planted</th><th style="width:300px"></th></tr>{calibration_rows}</table>

<h2>Per-GPU batchers</h2>
<p>Cumulative stats of each GPU's shared batcher, as last reported by an assess task on that replica.</p>
<table><tr><th>Replica</th><th>GPU</th><th>Records</th><th>Batches</th><th>Avg batch</th><th>Utilization</th></tr>{batcher_rows}</table>

<h2>Samples</h2>
<table><tr><th>Snippet</th><th>Language</th><th>Planted</th><th>P(vulnerable)</th><th>Weakness</th><th>Triage</th></tr>{sample_rows}</table>
</body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

# {{docs-fragment driver}}
async def _synthesize_then_assess(
    bank: TemplateBank, start: int, count: int, clean_fraction: float, seed: int, batch_size: int
) -> ChunkStats:
    """One chunk, end to end. Chaining the stages per chunk means the GPUs start
    receiving work as soon as the first chunk is instantiated. The chunk is
    assessed in smaller batches so each task's input stays small enough to
    inspect in the UI."""
    with flyte.group("synthesize"):
        snippets = await synthesize_snippets(bank, start, count, clean_fraction, seed)
    with flyte.group("assess"):
        batches = await asyncio.gather(
            *(assess_snippets(snippets[i : i + batch_size]) for i in range(0, len(snippets), batch_size))
        )
    stats = ChunkStats()
    for batch in batches:
        stats.merge(batch)
    return stats


@driver_env.task(report=True)
async def decision_demo(
    num_decisions: int = 1_000_000,
    chunk_size: int = 6_000,
    assess_batch_size: int = 1_000,
    clean_fraction: float = 0.3,
    seed: int = 0,
) -> DemoSummary:
    """Make ``num_decisions`` decisions: instantiate enough snippets, judge each
    one with the decision model, and merge the chunk statistics into a report.

    ``chunk_size`` is snippets per synthesize task and ``assess_batch_size`` is
    snippets per assess task, so a chunk fans out into several assess tasks.
    Tasks beyond a replica's ``concurrency`` queue up and start as earlier ones
    finish.
    """
    num_snippets = math.ceil(num_decisions / DECISIONS_PER_SNIPPET)
    bank = await build_template_bank()

    started = time.perf_counter()
    chunks = await asyncio.gather(
        *(
            _synthesize_then_assess(
                bank, start, min(chunk_size, num_snippets - start), clean_fraction, seed, assess_batch_size
            )
            for start in range(0, num_snippets, chunk_size)
        )
    )
    wall = time.perf_counter() - started

    stats = ChunkStats()
    for chunk in chunks:
        stats.merge(chunk)
    summary = summarise(stats, wall)

    await flyte.report.replace.aio(_render_report(summary, stats), do_flush=True)
    print(
        f"{summary.num_decisions:,} decisions in {wall:.0f}s ({summary.decisions_per_second:,.0f}/s) on "
        f"{summary.gpus} GPUs at {summary.mean_gpu_utilization:.0%} utilization, "
        f"{summary.mean_batch_size:.0f} records per pass | detection {summary.detection_accuracy:.0%} | "
        f"weakness class {summary.weakness_accuracy:.0%} | {summary.num_needs_review:,} need review"
    )
    return summary
# {{/docs-fragment driver}}


# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(decision_demo, num_decisions=1_000_000)
    print(run.url)
    run.wait()
# {{/docs-fragment main}}
