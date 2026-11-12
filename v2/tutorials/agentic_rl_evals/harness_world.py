"""Worlds, tasks, tools and the agent loop for ``harness_eval.py``.

A *world* is a small company's shared drive: a board-report PDF, an operating-model spreadsheet with
formulas, a strategy deck, and an inbox. *Tasks* are questions an analyst would answer from those files,
each with a deterministic answer. The *harness* is everything between the model and the world: the tool
implementations, timeouts, retries, which packages the code sandbox has, and the budget nudge.

``HARNESS_V1`` reproduces the classes of harness bug Mercor describes in its 397B RL guide: a PDF reader
that flattens a table's layout, a spreadsheet reader that shows values but hides formulas, a slides tool
that returns ``None`` on every call, packages missing from the code sandbox, and a tool with no timeout
in front of a flaky server. ``HARNESS_V2`` fixes each one. Nothing about the model changes between them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

REGIONS = ["North", "South", "East", "West", "Central"]
COMPANIES = ["Halvorsen Logistics", "Pell & Marsh Foods", "Corvane Health", "Tidewater Materials",
             "Brightline Freight", "Okafor Retail", "Lindqvist Energy", "Arden Biotech"]
TASK_TYPES = ["pdf_table", "sheet_formula", "slides", "email_lookup"]


# --- the harness --------------------------------------------------------------------------------------


# {{docs-fragment harness}}
class Harness(BaseModel):
    """Everything between the model and the world. Versioned like model weights, because it moves reward
    like model weights. Bump ``version`` whenever tool code changes: the cache key is the inputs."""

    version: str
    pdf_mode: str  # "text" (pypdf, content-stream order) | "layout" (pdfplumber tables + layout text)
    sheet_mode: str  # "values" | "values_and_formulas"
    slides_returns_none: bool  # the "PowerPoint tool returned None on every call" bug
    blocked_packages: list[str]  # packages missing from the code sandbox
    tool_timeout_s: float | None  # None = no per-tool timeout
    tool_retries: int  # automatic retries of a timed-out tool call
    nudge_at: float | None  # fraction of the turn budget after which the agent is told to wrap up
    max_turns: int = 16
    agent_timeout_s: float = 150.0


HARNESS_V1 = Harness(
    version="v1", pdf_mode="text", sheet_mode="values", slides_returns_none=True,
    blocked_packages=["pdfplumber", "pptx"], tool_timeout_s=None, tool_retries=0, nudge_at=None,
)
HARNESS_V2 = Harness(
    version="v2", pdf_mode="layout", sheet_mode="values_and_formulas", slides_returns_none=False,
    blocked_packages=[], tool_timeout_s=10.0, tool_retries=1, nudge_at=0.75,
)
# {{/docs-fragment harness}}


class Policy(BaseModel):
    model: str = "claude-haiku-5-5"
    effort: str = "medium"


class TaskSpec(BaseModel):
    task_id: str
    world_id: str
    task_type: str
    prompt: str
    answer: str  # canonical answer; numeric answers are compared with a tolerance
    numeric: bool = True


# --- worlds -------------------------------------------------------------------------------------------


def _rng(*parts: str) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:12], 16))


class WorldFacts(BaseModel):
    company: str
    revenue_m: dict[str, float]
    fte: dict[str, int]
    units: int
    price: float
    cogs_pct: float
    cost_per_fte_k: float
    gross_margin_target: int
    exit_region: str
    cfo: str


def world_facts(world_id: str) -> WorldFacts:
    r = _rng("world", world_id)
    return WorldFacts(
        company=r.choice(COMPANIES),
        revenue_m={g: round(r.uniform(12, 90), 1) for g in REGIONS},
        fte={g: r.randrange(40, 420) for g in REGIONS},
        units=r.randrange(2_500_000, 4_000_000),
        price=round(r.uniform(60, 140), 2),
        cogs_pct=round(r.uniform(0.25, 0.45), 2),
        cost_per_fte_k=round(r.uniform(80, 160), 1),
        gross_margin_target=r.randrange(31, 58),
        exit_region=r.choice(REGIONS),
        cfo=r.choice(["Dana Whitfield", "Dana Okoro", "Dana Lindgren"]),
    )


def build_world_files(world_id: str, root: Path) -> WorldFacts:
    """Write the world's files. Requires reportlab, openpyxl and python-pptx (world-build image only)."""
    f = world_facts(world_id)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    (root / "decks").mkdir(exist_ok=True)
    (root / "mail").mkdir(exist_ok=True)
    _write_pdf(f, world_id, root / "reports" / "q3_board_report.pdf")
    _write_xlsx(f, root / "models" / "fy27_operating_model.xlsx")
    _write_pptx(f, root / "decks" / "fy27_strategy.pptx")
    _write_inbox(f, world_id, root / "mail" / "inbox.json")
    return f


def _write_pdf(f: WorldFacts, world_id: str, path: Path) -> None:
    """Two tables. The headcount table's values are *drawn* bottom-up, so a reader that follows the content
    stream (pypdf) pairs labels with the wrong numbers, while a geometry-aware reader (pdfplumber) doesn't."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 740, f"{f.company} - Q3 Board Report")
    c.setFont("Helvetica", 10)
    c.drawString(72, 720, "Prepared for the board. Figures are unaudited.")

    def grid(x0: float, y_top: float, rows: int, widths: list[float], h: float = 18) -> None:
        total = sum(widths)
        for i in range(rows + 1):
            c.line(x0, y_top - i * h, x0 + total, y_top - i * h)
        x = x0
        for w in [0.0, *widths]:
            x += w
            c.line(x, y_top, x, y_top - rows * h)

    # table 1: Q3 revenue by region, drawn in reading order
    x0, top, h = 72, 690, 18
    grid(x0, top, len(REGIONS) + 1, [90, 110])
    c.drawString(x0 + 4, top - 13, "Region")
    c.drawString(x0 + 94, top - 13, "Q3 revenue ($M)")
    for i, g in enumerate(REGIONS, start=1):
        c.drawString(x0 + 4, top - i * h - 13, g)
        c.drawString(x0 + 94, top - i * h - 13, f"{f.revenue_m[g]:.1f}")

    # table 2: headcount by region, labels top-down, values bottom-up (the layout trap)
    x1 = 330
    order = REGIONS[:]
    _rng("pdf-order", world_id).shuffle(order)
    grid(x1, top, len(order) + 1, [90, 110])
    c.drawString(x1 + 4, top - 13, "Region")
    c.drawString(x1 + 94, top - 13, "Headcount (FTE)")
    for i, g in enumerate(order, start=1):
        c.drawString(x1 + 4, top - i * h - 13, g)
    for i in range(len(order), 0, -1):
        c.drawString(x1 + 94, top - i * h - 13, str(f.fte[order[i - 1]]))
    c.showPage()
    c.save()


def _write_xlsx(f: WorldFacts, path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    a = wb.active
    a.title = "Assumptions"
    a.append(["Assumption", "Value"])
    a.append(["Units sold (FY27)", f.units])
    a.append(["Price per unit ($)", f.price])
    a.append(["COGS (% of revenue)", f.cogs_pct])
    a.append(["Loaded cost per FTE ($K)", f.cost_per_fte_k])
    a.append(["Total FTE (from Q3 board report)", sum(f.fte.values())])
    m = wb.create_sheet("Model")
    m.append(["Line ($K)", "FY27"])
    m.append(["Revenue", "=Assumptions!B2*Assumptions!B3/1000"])
    m.append(["COGS", "=B2*Assumptions!B4"])
    m.append(["Gross profit", "=B2-B3"])
    m.append(["Operating expense", "=Assumptions!B5*Assumptions!B6"])
    m.append(["EBITDA", "=B4-B5"])
    wb.save(path)


def _write_pptx(f: WorldFacts, path: Path) -> None:
    from pptx import Presentation

    p = Presentation()
    slides = [
        ("FY27 strategy", f"{f.company} leadership offsite"),
        ("Where we stand", "Q3 closed ahead of plan in three of five regions."),
        ("FY27 targets", f"Gross margin target: {f.gross_margin_target}%\nNet retention target: 112%"),
        ("Next steps", "Region reviews in January; budget lock in February."),
    ]
    for title, body in slides:
        s = p.slides.add_slide(p.slide_layouts[1])
        s.shapes.title.text = title
        s.placeholders[1].text = body
    p.save(path)


def _write_inbox(f: WorldFacts, world_id: str, path: Path) -> None:
    others = [g for g in REGIONS if g != f.exit_region]
    mails = [
        {"from": "ops@corp", "subject": "Facilities update", "body": "The Q4 office move is on schedule."},
        {"from": f.cfo, "subject": "Portfolio decision",
         "body": f"After the Q3 review we're exiting the {f.exit_region} region in FY27. Keep this internal."},
        {"from": "hr@corp", "subject": "Benefits enrollment", "body": "Open enrollment closes Nov 30."},
        {"from": "sales@corp", "subject": f"{others[0]} pipeline", "body": f"{others[0]} pipeline is up 8%."},
    ]
    _rng("inbox", world_id).shuffle(mails)
    path.write_text(json.dumps(mails, indent=1))


# {{docs-fragment tasks}}
def make_tasks(world_id: str) -> list[TaskSpec]:
    f = world_facts(world_id)
    region = _rng("task-region", world_id).choice(REGIONS)
    rev_per_fte = f.revenue_m[region] * 1000 / f.fte[region]
    return [
        TaskSpec(task_id=f"{world_id}-pdf", world_id=world_id, task_type="pdf_table",
                 prompt=f"Using the Q3 board report, what was the {region} region's Q3 revenue per FTE, "
                        "in $K per FTE, rounded to one decimal place?",
                 answer=f"{rev_per_fte:.1f}"),
        TaskSpec(task_id=f"{world_id}-sheet", world_id=world_id, task_type="sheet_formula",
                 prompt="In the FY27 operating model, by how much would FY27 EBITDA change (in $K) if the loaded "
                        "cost per FTE rose by $5K? Give a signed number.",
                 answer=f"{-5 * sum(f.fte.values()):.0f}"),
        TaskSpec(task_id=f"{world_id}-slides", world_id=world_id, task_type="slides",
                 prompt="What FY27 gross margin target (in percent) does the strategy deck set?",
                 answer=str(f.gross_margin_target)),
        TaskSpec(task_id=f"{world_id}-email", world_id=world_id, task_type="email_lookup",
                 prompt=f"{f.cfo.split()[0]} from finance emailed which region we're exiting in FY27. What was that "
                        "region's Q3 revenue, in $M, per the board report?",
                 answer=f"{f.revenue_m[f.exit_region]:.1f}"),
    ]
# {{/docs-fragment tasks}}


def verify(task: TaskSpec, answer: str | None) -> bool:
    """The deterministic verifier. Numeric answers within 1% (or 0.15 absolute) count."""
    if not answer:
        return False
    if not task.numeric:
        return task.answer.lower() in answer.lower()
    nums = re.findall(r"-?\d[\d,]*\.?\d*", answer.replace("−", "-"))
    if not nums:
        return False
    got = float(nums[0].replace(",", ""))
    want = float(task.answer)
    return abs(got - want) <= max(0.15, 0.01 * abs(want))


# --- tools ------------------------------------------------------------------------------------------


TOOL_SPECS: list[dict[str, Any]] = [
    {"name": "list_files", "description": "List every file in the company drive.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "read_pdf", "description": "Extract the contents of a PDF in the drive.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
                      "additionalProperties": False}},
    {"name": "read_spreadsheet", "description": "Read every sheet of an .xlsx workbook.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
                      "additionalProperties": False}},
    {"name": "read_slides", "description": "Read the text of every slide in a .pptx deck.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
                      "additionalProperties": False}},
    {"name": "search_email", "description": "Search the inbox. Returns matching emails.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
                      "additionalProperties": False}},
    {"name": "run_python", "description": "Run a Python script with the drive as the working directory. "
                                          "Returns stdout and stderr. Common data packages are installed.",
     "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"],
                      "additionalProperties": False}},
    {"name": "submit_answer", "description": "Submit the final answer and end the task.",
     "input_schema": {"type": "object", "properties": {"answer": {"type": "string"},
                                                       "justification": {"type": "string"}},
                      "required": ["answer", "justification"], "additionalProperties": False}},
]


class ToolError(Exception):
    pass


class World:
    """Tool implementations over one trial's private copy of the world, configured by the harness."""

    def __init__(self, root: Path, harness: Harness, trial_seed: str):
        self.root, self.h, self.seed, self.calls = root, harness, trial_seed, 0

    def _path(self, p: str) -> Path:
        q = (self.root / p.lstrip("/")).resolve()
        if not str(q).startswith(str(self.root.resolve())) or not q.exists():
            raise ToolError(f"no such file: {p}")
        return q

    async def list_files(self) -> str:
        return "\n".join(str(p.relative_to(self.root)) for p in sorted(self.root.rglob("*")) if p.is_file())

    async def read_pdf(self, path: str) -> str:
        p = self._path(path)
        if self.h.pdf_mode == "text":
            from pypdf import PdfReader

            return "\n".join(page.extract_text() or "" for page in PdfReader(str(p)).pages)
        import pdfplumber

        out = []
        with pdfplumber.open(str(p)) as pdf:
            for page in pdf.pages:
                for t in page.extract_tables():
                    out.append("\n".join(" | ".join(c or "" for c in row) for row in t))
                out.append(page.extract_text(layout=False) or "")
        return "\n\n".join(out)

    async def read_spreadsheet(self, path: str) -> str:
        from openpyxl import load_workbook

        wb = load_workbook(self._path(path), data_only=False)
        values = _evaluate_workbook(wb)
        lines = []
        for ws in wb.worksheets:
            lines.append(f"## sheet {ws.title}")
            for row in ws.iter_rows():
                cells = []
                for c in row:
                    if c.value is None:
                        continue
                    v = values.get((ws.title, c.coordinate), c.value)
                    is_formula = isinstance(c.value, str) and c.value.startswith("=")
                    shown = f"{_fmt(v)} [formula {c.value}]" if is_formula and self.h.sheet_mode != "values" else _fmt(v)
                    cells.append(f"{c.coordinate}: {shown}")
                lines.append("  ".join(cells))
        return "\n".join(lines)

    async def read_slides(self, path: str) -> str | None:
        p = self._path(path)
        from pptx import Presentation

        text = []
        for i, s in enumerate(Presentation(str(p)).slides, start=1):
            parts = [sh.text_frame.text for sh in s.shapes if sh.has_text_frame]
            text.append(f"slide {i}: " + " / ".join(parts))
        if self.h.slides_returns_none:
            return None  # the bug: the tool did its work and then forgot to return it
        return "\n".join(text)

    async def search_email(self, query: str) -> str:
        self.calls += 1
        # a flaky mail server: some calls hang. Only the harness decides whether that costs a turn or a trial.
        if _rng("mail", self.seed, str(self.calls)).random() < 0.35:
            await asyncio.sleep(3600)
        mails = json.loads((self.root / "mail" / "inbox.json").read_text())
        words = [w.lower() for w in re.findall(r"\w+", query)]
        hits = [m for m in mails if any(w in json.dumps(m).lower() for w in words)] or mails
        return json.dumps(hits, indent=1)

    async def run_python(self, code: str) -> str:
        blocked = json.dumps(self.h.blocked_packages)
        prelude = (
            "import sys\n"
            "class _Missing:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            f"        if name.split('.')[0] in {blocked}:\n"
            "            raise ModuleNotFoundError(f\"No module named '{name}'\")\n"
            "sys.meta_path.insert(0, _Missing())\n"
        )
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", prelude + code, cwd=str(self.root),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=60)
        except TimeoutError:
            proc.kill()
            raise ToolError("run_python exceeded 60 s")
        return (out.decode()[-3000:] + ("\nSTDERR:\n" + err.decode()[-2000:] if err else "")).strip()


def _fmt(v: Any) -> str:
    return f"{v:,.2f}" if isinstance(v, float) else str(v)


def _evaluate_workbook(wb) -> dict[tuple[str, str], Any]:
    """A tiny formula evaluator (cell refs, + - * /) so 'values' mode can show computed numbers."""
    raw = {(ws.title, c.coordinate): c.value for ws in wb.worksheets for row in ws.iter_rows() for c in row}
    memo: dict[tuple[str, str], Any] = {}

    def ev(sheet: str, coord: str) -> Any:
        key = (sheet, coord)
        if key in memo:
            return memo[key]
        v = raw.get(key)
        if isinstance(v, str) and v.startswith("="):
            expr = v[1:]

            def ref(m: re.Match) -> str:
                sh, cc = (m.group(1), m.group(2)) if m.group(1) else (sheet, m.group(2))
                return repr(float(ev(sh, cc)))

            # only evaluates the workbook formulas this module wrote, never agent or user input
            v = eval(re.sub(r"(?:(\w+)!)?([A-Z]+\d+)", ref, expr), {"__builtins__": {}})
        memo[key] = v
        return v

    return {k: ev(*k) for k in raw}


# --- the agent loop ---------------------------------------------------------------------------------


class Step(BaseModel):
    tool: str
    input: str
    output: str
    error: str | None = None
    seconds: float


class Trajectory(BaseModel):
    trial_id: str
    task_id: str
    world_id: str
    task_type: str
    harness_version: str
    model: str
    pass_idx: int
    stop_reason: str  # submitted | max_turns | agent_timeout | refusal | error
    answer: str | None = None
    justification: str | None = None
    steps: list[Step] = []
    turns: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    elapsed_s: float = 0.0
    produced_by_run: str = ""  # which run actually executed this cell; differs from the current run on a cache hit


SYSTEM = (
    "You are an analyst agent with tool access to a company's shared drive (PDFs, spreadsheets, slide decks, "
    "email). Work out the answer from the files; never guess a number you could read or compute. When you are "
    "done, call submit_answer with just the final value in `answer` (a number or a name, no units) and a short "
    "justification that names the files you used."
)


# {{docs-fragment agent_loop}}
async def run_agent(task: TaskSpec, policy: Policy, harness: Harness, world: World, traj: Trajectory) -> None:
    """One trial. Tool failures go back to the model as errors; only the harness's limits end the trial."""
    import anthropic

    client = anthropic.AsyncAnthropic(max_retries=3)
    messages: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
    for turn in range(harness.max_turns):
        traj.turns = turn + 1
        resp = await client.messages.create(
            model=policy.model, max_tokens=8000, system=SYSTEM, tools=TOOL_SPECS, messages=messages,
            output_config={"effort": policy.effort}, cache_control={"type": "ephemeral"},
        )
        traj.input_tokens += resp.usage.input_tokens + (resp.usage.cache_read_input_tokens or 0)
        traj.cached_input_tokens += resp.usage.cache_read_input_tokens or 0
        traj.output_tokens += resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            traj.stop_reason = "refusal"
            return
        messages.append({"role": "assistant", "content": resp.content})
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            messages.append({"role": "user", "content": "Call submit_answer when you have the answer."})
            continue
        results: list[dict[str, Any]] = []
        for u in uses:
            if u.name == "submit_answer":
                traj.answer, traj.justification = u.input.get("answer"), u.input.get("justification")
                traj.stop_reason = "submitted"
                return
            results.append(await _call_tool(world, harness, u, traj))
        if harness.nudge_at is not None and turn + 1 >= harness.nudge_at * harness.max_turns:
            left = harness.max_turns - turn - 1
            results.append({"type": "text", "text": f"Budget note: {left} turns left. Submit your best answer soon."})
        messages.append({"role": "user", "content": results})
    traj.stop_reason = "max_turns"


async def _call_tool(world: World, harness: Harness, u: Any, traj: Trajectory) -> dict[str, Any]:
    fn = getattr(world, u.name, None)
    t0 = time.monotonic()
    # Record the call *before* awaiting it: a call that hangs until the trial deadline still shows up.
    step = Step(tool=u.name, input=json.dumps(u.input)[:500], output="", error="did not return", seconds=0.0)
    traj.steps.append(step)
    out, err = "", None
    for attempt in range(harness.tool_retries + 1):
        try:
            coro = fn(**u.input) if fn else None
            if coro is None:
                raise ToolError(f"unknown tool {u.name}")
            res = await (asyncio.wait_for(coro, harness.tool_timeout_s) if harness.tool_timeout_s else coro)
            out, err = ("None" if res is None else str(res)), None
            break
        except TimeoutError:
            err = f"{u.name} timed out after {harness.tool_timeout_s:.0f} s"
        except (ToolError, TypeError) as e:
            err = str(e)
            break
        except Exception as e:  # a broken tool is a tool error for the model, not a dead trial
            err = f"{type(e).__name__}: {e}"
            break
    step.output, step.error, step.seconds = out[:1500], err, round(time.monotonic() - t0, 2)
    content = out[:12000] if err is None else f"ERROR: {err}"
    return {"type": "tool_result", "tool_use_id": u.id, "content": content, "is_error": err is not None}
# {{/docs-fragment agent_loop}}
