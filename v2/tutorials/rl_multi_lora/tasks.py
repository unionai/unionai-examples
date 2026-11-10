"""
The two demo skills: data generators, prompts, and verifiable rewards.

Pure Python (stdlib only) so the CPU reward task, the GPU tasks, and the Gradio app can all import it
without pulling in torch or vLLM.

- ``countdown``: combine the given numbers with + - * / to hit a target. Small models guess, reuse
  numbers, and botch arithmetic; RL teaches them to search and check their work.
- ``json``: turn a messy customer email into one strict JSON record. Small models wrap output in
  markdown, grab the wrong email or amount, and miss normalization rules; RL makes them reliable.

Every problem is a ``(user_message, answer)`` pair. ``answer`` is a JSON string with whatever the
reward needs to verify a completion.
"""

from __future__ import annotations

import ast
import itertools
import json
import random
import re
from datetime import date, timedelta
from fractions import Fraction
from typing import Any

TASKS = ("countdown", "json")

# ----------------------------------------------------------------------------------------------------
# Countdown
# ----------------------------------------------------------------------------------------------------
COUNTDOWN_SYSTEM = (
    "You solve number puzzles. Combine the given numbers with +, -, *, / and parentheses so the "
    "expression equals the target. Use every number exactly once and no other numbers. "
    "Think step by step inside <think> </think>, trying combinations and checking the arithmetic. "
    "Then write only the final expression inside <answer> </answer>, for example "
    "<answer>(6 - 2) * 5</answer>."
)

COUNTDOWN_SIZES = (3, 4)  # how many numbers per puzzle
COUNTDOWN_MAX_NUMBER = 25
COUNTDOWN_TARGET_RANGE = (10, 100)

_OPS = ("+", "-", "*", "/")


def _apply(op: str, a: Fraction, b: Fraction) -> Fraction | None:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    return a / b if b != 0 else None


def _make_countdown(rng: random.Random) -> tuple[list[int], int]:
    """Build a solvable puzzle by combining random numbers with random ops (solvable by construction)."""
    lo, hi = COUNTDOWN_TARGET_RANGE
    while True:
        k = rng.choice(COUNTDOWN_SIZES)
        numbers = [rng.randint(1, COUNTDOWN_MAX_NUMBER) for _ in range(k)]
        pool = [Fraction(n) for n in numbers]
        while len(pool) > 1:
            a, b = (pool.pop(rng.randrange(len(pool))) for _ in range(2))
            op = rng.choice(_OPS)
            # Keep every intermediate a positive integer so the intended solution reads naturally.
            if op == "-" and a < b:
                a, b = b, a
            if op == "/" and (b == 0 or a % b != 0):
                op = "*"
            value = _apply(op, a, b)
            assert value is not None
            pool.append(value)
        target = pool[0]
        if target.denominator == 1 and lo <= target <= hi and int(target) not in numbers:
            return numbers, int(target)


def countdown_message(numbers: list[int], target: int) -> str:
    return f"Numbers: {', '.join(map(str, numbers))}\nTarget: {target}"


def parse_countdown_message(text: str) -> tuple[list[int], int] | None:
    """Recover (numbers, target) from a user message, so free-typed puzzles can be graded too."""
    m_nums = re.search(r"numbers?\s*[:=]?\s*\[?([\d,\s]+)\]?", text, re.IGNORECASE)
    m_target = re.search(r"(?:target|make|equals?|get)\s*[:=]?\s*(\d+)", text, re.IGNORECASE)
    if not m_nums or not m_target:
        return None
    numbers = [int(n) for n in re.findall(r"\d+", m_nums.group(1))]
    return (numbers, int(m_target.group(1))) if len(numbers) >= 2 else None


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>(.*?)</think>\s*<answer>", re.DOTALL | re.IGNORECASE)
MIN_THINK_CHARS = 40  # an empty or token <think> block doesn't count as reasoning


def _has_reasoning(completion: str) -> bool:
    """True if a non-trivial <think> block comes right before the <answer>."""
    return any(len(t.strip()) >= MIN_THINK_CHARS for t in _THINK_RE.findall(completion))


def _extract_expression(completion: str) -> str | None:
    matches = _ANSWER_RE.findall(completion)
    if not matches:
        return None
    expr = matches[-1].strip()
    expr = expr.replace("×", "*").replace("x", "*").replace("÷", "/").replace("−", "-")
    return expr.split("=")[0].strip()  # tolerate "(6 - 2) * 5 = 20"


def _evaluate(expr: str) -> tuple[Fraction, list[int]] | None:
    """Safely evaluate an arithmetic expression; returns (value, integer literals used)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    used: list[int] = []

    def walk(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            used.append(node.value)
            return Fraction(node.value)
        if isinstance(node, ast.BinOp):
            ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
            op = ops.get(type(node.op))
            if op is None:
                raise ValueError("operator not allowed")
            value = _apply(op, walk(node.left), walk(node.right))
            if value is None:
                raise ZeroDivisionError
            return value
        raise ValueError(f"not allowed: {type(node).__name__}")

    try:
        return walk(tree), used
    except (ValueError, ZeroDivisionError):
        return None


# {{docs-fragment check_countdown}}
def check_countdown(completion: str, numbers: list[int], target: int) -> tuple[bool, bool]:
    """Returns (formatted, correct)."""
    expr = _extract_expression(completion)
    if expr is None:
        return False, False
    result = _evaluate(expr)
    if result is None:
        return False, False
    value, used = result
    return True, value == target and sorted(used) == sorted(numbers)


# {{/docs-fragment check_countdown}}


def solve_countdown(numbers: list[int], target: int) -> str | None:
    """Brute-force a solution (tiny search for <= 5 numbers); used by the UI to show a reference answer."""

    def search(items: list[tuple[Fraction, str]]) -> str | None:
        if len(items) == 1:
            return items[0][1] if items[0][0] == target else None
        for i, j in itertools.permutations(range(len(items)), 2):
            (a, ea), (b, eb) = items[i], items[j]
            rest = [items[k] for k in range(len(items)) if k not in (i, j)]
            for op in _OPS:
                if op in "+*" and i > j:
                    continue  # commutative: try each pair once
                value = _apply(op, a, b)
                if value is not None:
                    found = search(rest + [(value, f"({ea} {op} {eb})")])
                    if found:
                        return found
        return None

    found = search([(Fraction(n), str(n)) for n in numbers])
    return found[1:-1] if found and found.startswith("(") and found.endswith(")") else found


# ----------------------------------------------------------------------------------------------------
# JSON extraction
# ----------------------------------------------------------------------------------------------------
JSON_FIELDS = (
    "customer_name",
    "email",
    "order_id",
    "amount_cents",
    "currency",
    "due_date",
    "priority",
    "refund_requested",
)

JSON_SYSTEM = (
    "You extract support tickets into JSON. Reply with exactly one JSON object and nothing else: "
    "no markdown, no code fences, no explanation. Use exactly these keys:\n"
    '- "customer_name": full name of the customer who wrote the message (not anyone they cc or mention)\n'
    '- "email": the customer\'s own email address, or null if they did not give one\n'
    '- "order_id": the order number formatted as "ORD-" followed by its digits, e.g. "ORD-48213"\n'
    '- "amount_cents": the amount charged for this order as an integer number of cents, e.g. $12.50 -> 1250\n'
    '- "currency": one of "USD", "EUR", "GBP"\n'
    '- "due_date": the date the customer needs this resolved by, as "YYYY-MM-DD", or null if none is given\n'
    '- "priority": "high" if urgent or blocking, "medium" if they want it soon, "low" if there is no rush\n'
    '- "refund_requested": true only if the customer asks for their money back, otherwise false'
)

_FIRST = (
    "Priya",
    "Marcus",
    "Elena",
    "Tomás",
    "Aisha",
    "Kenji",
    "Olivia",
    "Dmitri",
    "Fatima",
    "Liam",
    "Sofia",
    "Kwame",
    "Hannah",
    "Mateo",
    "Ingrid",
    "Ravi",
    "Chloe",
    "Yusuf",
    "Mei",
    "Gabriel",
)
_LAST = (
    "Natarajan",
    "Okafor",
    "Kowalski",
    "Lindqvist",
    "Haddad",
    "Tanaka",
    "Brennan",
    "Volkov",
    "Moreno",
    "Achterberg",
    "Silva",
    "Mensah",
    "Fischer",
    "Dubois",
    "Kaur",
    "Whitfield",
)
_DOMAINS = ("gmail.com", "outlook.com", "fastmail.com", "acme.io", "proton.me", "northwind.co")
_CURRENCY_SYMBOL = {"USD": "$", "EUR": "€", "GBP": "£"}
_CURRENCY_WORD = {"USD": "dollars", "EUR": "euros", "GBP": "pounds"}
_PRIORITY_PHRASES = {
    "low": ("No rush on this.", "Whenever you get a chance is fine.", "Not urgent at all."),
    "medium": (
        "Would appreciate a reply this week.",
        "Please look into it soon.",
        "Sometime in the next few days would be good.",
    ),
    "high": ("URGENT!!", "This is blocking our launch.", "Please treat this as top priority.", "Need this ASAP."),
}
_REFUND_PHRASES = {
    True: ("I'd like a full refund.", "Please refund me.", "I want my money back.", "Refund it, please."),
    False: (
        "I don't want a refund, just send the right item.",
        "No refund needed - please just reship it.",
        "Please replace it rather than refunding.",
        "Not asking for a refund, I just want it fixed.",
    ),
}


def _fmt_amount(cents: int, currency: str, rng: random.Random) -> str:
    whole, frac = divmod(cents, 100)
    sym, word = _CURRENCY_SYMBOL[currency], _CURRENCY_WORD[currency]
    styles = [
        f"{sym}{whole:,}.{frac:02d}",
        f"{whole}.{frac:02d} {currency}",
        f"{whole:,}.{frac:02d} {word}",
    ]
    if frac == 0:
        styles.append(f"{sym}{whole:,}")
        if whole >= 1000 and whole % 100 == 0:
            styles.append(f"{sym}{whole / 1000:g}k")
    return rng.choice(styles)


def _ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _fmt_date(d: date, rng: random.Random) -> str:
    return rng.choice(
        [
            d.strftime("%B ") + f"{d.day}, {d.year}",
            f"{d.day} " + d.strftime("%B %Y"),
            d.isoformat(),
            f"the {_ordinal(d.day)} of " + d.strftime("%B, %Y"),
            d.strftime("%b ") + f"{_ordinal(d.day)} {d.year}",
        ]
    )


def _fmt_order(digits: str, rng: random.Random) -> str:
    return rng.choice(
        [f"order #{digits}", f"order number {digits}", f"ORD{digits}", f"ord-{digits}", f"my order ({digits})"]
    )


def _make_ticket(rng: random.Random) -> tuple[str, dict[str, Any]]:
    """Sample a ground-truth record, then render it as a messy email with distractors."""
    first, last = rng.choice(_FIRST), rng.choice(_LAST)
    name = f"{first} {last}"
    has_email = rng.random() < 0.75
    email = f"{first.lower()}.{last.lower()}@{rng.choice(_DOMAINS)}".replace("á", "a") if has_email else None
    digits = str(rng.randint(10000, 99999))
    currency = rng.choice(("USD", "USD", "EUR", "GBP"))
    cents = rng.choice([rng.randint(5, 2500) * 100, rng.randint(500, 250000)])
    order_day = date(2025, 1, 1) + timedelta(days=rng.randint(0, 600))
    due = order_day + timedelta(days=rng.randint(5, 40)) if rng.random() < 0.8 else None
    priority = rng.choice(("low", "medium", "high"))
    refund = rng.random() < 0.5

    truth = {
        "customer_name": name,
        "email": email,
        "order_id": f"ORD-{digits}",
        "amount_cents": cents,
        "currency": currency,
        "due_date": due.isoformat() if due else None,
        "priority": priority,
        "refund_requested": refund,
    }

    # Distractors: another person (with an email), an older amount, and the order date.
    cc_first, cc_last = rng.choice(_FIRST), rng.choice(_LAST)
    cc = f"{cc_first} {cc_last}"
    cc_email = f"{cc_first.lower()}@{rng.choice(_DOMAINS)}"
    old_cents = max(100, cents + rng.choice([-1, 1]) * rng.randint(1, 50) * 100)

    body = [
        (
            f"I placed {_fmt_order(digits, rng)} on {_fmt_date(order_day, rng)} and was charged "
            f"{_fmt_amount(cents, currency, rng)}, but the package arrived damaged."
        ),
        rng.choice(_REFUND_PHRASES[refund]),
        rng.choice(_PRIORITY_PHRASES[priority]),
    ]
    if due:
        body.append(
            rng.choice(["I need this sorted by", "Please resolve it before", "Deadline on my end is"])
            + f" {_fmt_date(due, rng)}."
        )
    distractors = [
        f"(For reference, my previous order was {_fmt_amount(old_cents, currency, rng)} and that one was fine.)",
        f"I'm cc'ing {cc} ({cc_email}) who handles our purchasing.",
    ]
    rng.shuffle(body)
    rng.shuffle(distractors)
    body.insert(rng.randint(1, len(body)), distractors[0])
    body.append(distractors[1])

    greeting = rng.choice(["Hi team,", "Hello,", "hey support -", "To whom it may concern,"])
    signoff = rng.choice(["Thanks,", "Best,", "Cheers,", "Regards,", "-"])
    contact = f"\n{email}" if email else ""
    text = f"{greeting}\n\n{' '.join(body)}\n\n{signoff}\n{name}{contact}"
    return text, truth


def _parse_json_strict(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_json_lenient(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _field_ok(key: str, got: Any, want: Any) -> bool:
    if want is None or isinstance(want, bool):
        return got is want
    if key == "amount_cents":
        return isinstance(got, int) and not isinstance(got, bool) and got == want
    if not isinstance(got, str):
        return False
    if key in ("customer_name", "email"):
        return " ".join(got.split()).casefold() == want.casefold()
    return got.strip() == want


def grade_json(completion: str, truth: dict[str, Any]) -> dict[str, Any]:
    """Grade a completion field by field. ``wrong`` lists the keys the UI should highlight."""
    strict = _parse_json_strict(completion)
    obj = strict if strict is not None else _parse_json_lenient(completion)
    if obj is None:
        return {"strict": False, "keys_ok": False, "wrong": list(JSON_FIELDS), "frac": 0.0}
    wrong = [k for k in JSON_FIELDS if k not in obj or not _field_ok(k, obj[k], truth[k])]
    return {
        "strict": strict is not None,
        "keys_ok": set(obj) == set(JSON_FIELDS),
        "wrong": wrong,
        "frac": 1 - len(wrong) / len(JSON_FIELDS),
    }


def check_json_schema(completion: str) -> tuple[bool, bool]:
    """For free-typed emails with no ground truth: (bare JSON object, exactly the right keys)."""
    obj = _parse_json_strict(completion)
    return obj is not None, obj is not None and set(obj) == set(JSON_FIELDS)


# ----------------------------------------------------------------------------------------------------
# Common interface used by training, eval, and the chat UI
# ----------------------------------------------------------------------------------------------------
SYSTEM_PROMPTS = {"countdown": COUNTDOWN_SYSTEM, "json": JSON_SYSTEM}


def messages(task: str, user: str) -> list[dict[str, str]]:
    """Chat messages for one problem. Training, eval, and serving all go through this."""
    return [{"role": "system", "content": SYSTEM_PROMPTS[task]}, {"role": "user", "content": user}]


# {{docs-fragment problems}}
def make_problem(task: str, rng: random.Random) -> tuple[str, str]:
    if task == "countdown":
        numbers, target = _make_countdown(rng)
        return countdown_message(numbers, target), json.dumps({"numbers": numbers, "target": target})
    if task == "json":
        text, truth = _make_ticket(rng)
        return text, json.dumps(truth)
    raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")


EVAL_SEED = 1_000_003


def eval_set(task: str, n: int = 100) -> list[tuple[str, str]]:
    """Fixed held-out problems; training skips anything in here."""
    rng = random.Random(EVAL_SEED)
    return [make_problem(task, rng) for _ in range(n)]


def train_batch(task: str, iteration: int, n: int) -> list[tuple[str, str]]:
    """Fresh problems for one GRPO iteration (deterministic per iteration, so resumes are reproducible)."""
    held_out = {q for q, _ in eval_set(task)}
    rng = random.Random(iteration * 7919 + 17)
    batch: list[tuple[str, str]] = []
    while len(batch) < n:
        q, a = make_problem(task, rng)
        if q not in held_out:
            batch.append((q, a))
    return batch


# {{/docs-fragment problems}}


# {{docs-fragment score}}
def score(task: str, completion: str, answer: str) -> tuple[float, bool, bool]:
    """Returns (reward, formatted, correct). Reward is in [0, 1.1] for countdown and [0, 1] for json."""
    truth = json.loads(answer)
    if task == "countdown":
        formatted, correct = check_countdown(completion, truth["numbers"], truth["target"])
        # No reasoning, no reward. Without this gate the first run learned to drop <think> and guess:
        # format went to 100%, but the search that solves the harder puzzles disappeared.
        formatted = formatted and _has_reasoning(completion)
        return 0.1 * formatted + 1.0 * (correct and formatted), formatted, correct
    g = grade_json(completion, truth)
    reward = 0.2 * g["strict"] + 0.2 * g["keys_ok"] + 0.6 * g["frac"]
    return reward, g["strict"] and g["keys_ok"], g["strict"] and not g["wrong"]


# {{/docs-fragment score}}


REWARD_RULES = {
    "countdown": (
        "Zero unless the reply reasons inside `<think>` before its `<answer>`. Then +0.1 for a valid "
        "+ - * / expression, and +1.0 more if it uses exactly the given numbers and equals the target. "
        "No worked solutions are used: the reward is the checker."
    ),
    "json": (
        "+0.2 for a bare JSON object (no prose or code fences), +0.2 for exactly the 8 schema keys, "
        "+0.6 × the fraction of fields that match the ground truth."
    ),
}


def verdict(task: str, user: str, completion: str, answer: str | None = None) -> tuple[bool | None, str]:
    """Human-readable grade for the chat UI: (passed, reason). ``passed`` is None when ungradable.

    Countdown puzzles are graded from the message itself; JSON needs the ground truth (``answer``) of a
    generated or preset email, and otherwise falls back to a schema check.
    """
    if task == "countdown":
        parsed = parse_countdown_message(user)
        if parsed is None:
            return None, "Can't grade: write the puzzle as 'Numbers: 1, 2, 3' and 'Target: 6'."
        numbers, target = parsed
        expr = _extract_expression(completion)
        if expr is None:
            return False, "No <answer>…</answer> expression."
        result = _evaluate(expr)
        if result is None:
            return False, f"`{expr}` is not a valid + - * / expression."
        value, used = result
        shown = str(value) if value.denominator == 1 else f"{float(value):.3g}"
        if sorted(used) != sorted(numbers):
            return False, f"`{expr}` uses {sorted(used)}, but must use exactly {sorted(numbers)}."
        if value != target:
            return False, f"`{expr}` = {shown}, not {target}."
        return True, f"`{expr}` = {target}"

    if answer is None:
        bare, keys_ok = check_json_schema(completion)
        if not bare:
            return False, "Not a bare JSON object (extra text or markdown around it)."
        if not keys_ok:
            return False, "JSON, but the keys don't match the schema."
        return None, "Valid JSON with the right keys (no ground truth for custom text)."
    truth = json.loads(answer)
    g = grade_json(completion, truth)
    if not g["strict"]:
        return False, "Not a bare JSON object (extra text or markdown around it)."
    if g["wrong"]:
        obj = _parse_json_strict(completion)
        diffs = [
            f"`{k}`: missing, expected `{json.dumps(truth[k])}`"
            if k not in obj
            else f"`{k}`: got `{json.dumps(obj[k])}`, expected `{json.dumps(truth[k])}`"
            for k in g["wrong"]
        ]
        return False, f"{len(g['wrong'])} of 8 fields wrong: " + "; ".join(diffs)
    return True, "All 8 fields correct."


# Curated stage prompts for the live demo: each is checked by tests to be solvable / well-formed.
PRESETS: dict[str, list[tuple[str, str]]] = {
    "countdown": [
        (countdown_message(n, t), json.dumps({"numbers": n, "target": t}))
        for n, t in [([7, 3, 9, 2], 24), ([25, 4, 3], 97), ([8, 6, 5, 2], 58), ([19, 3, 11], 46), ([12, 9, 7, 4], 87)]
    ],
    "json": [make_problem("json", random.Random(seed)) for seed in (11, 23, 42, 77, 314)],
}
