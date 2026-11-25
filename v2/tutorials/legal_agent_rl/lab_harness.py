"""Harvey's Legal Agent Benchmark (LAB), run inside Flyte tasks.

LAB (github.com/harveyai/harvey-labs, MIT) ships tasks, rubrics, an agent harness and a grader as the ``lab_core``
package. This module runs that harness unmodified where it can, and replaces two things a Kubernetes pod can't do:

- **The Podman sandbox.** LAB runs each agent inside its own ``--network=none`` Podman container. A pod can't start
  containers, so ``PodSandbox`` runs the agent's shell in the task's own container instead, which is built FROM
  LAB's sandbox image (same pandoc, LibreOffice and parsers). Each rollout gets a private workspace directory, and
  the canonical ``/workspace`` paths the agent sees are mapped onto it. This is a fidelity difference: the pod has a
  network, and several rollouts share one container.
- **The hosted judges.** LAB's standard judges are ``claude-sonnet-4-6`` and ``gpt-5.5``. ``OpenAICompatJudge`` sends
  LAB's own ``rubric_criterion`` prompt to any OpenAI-compatible server (here, an open-weight model on vLLM), so a
  rubric can be graded on the same node as the policy. ``lab_core``'s own ``Judge`` still works for the standard pair.

Nothing here imports Flyte, so it runs the same in a task, a test, or a notebook.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

LAB_REPO = "https://github.com/harveyai/harvey-labs.git"
POD_WS_ROOT = Path(os.environ.get("LAB_POD_WS_ROOT", "/tmp/lab-ws"))


# --- tasks -----------------------------------------------------------------------------------------------------


def fetch_lab(task_ids: list[str], commit: str, dest: Path) -> Path:
    """Sparse-checkout only the listed task directories of LAB at a pinned commit. Returns the LAB root."""
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        subprocess.run(["git", "init", "-q", str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", LAB_REPO], check=True)
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "--no-cone", *[f"tasks/{t}/" for t in task_ids]],
                   check=True)
    subprocess.run(["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit],
                   check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", commit], check=True)
    return dest


class LabTask(BaseModel):
    """What the rollout and the grader need from one LAB task. ``criteria`` is LAB's inline rubric, unchanged."""

    id: str
    title: str
    instructions: str
    work_type: str | None = None
    deliverables: dict[str, str]
    criteria: list[dict[str, Any]]


def load_task(lab_root: Path, task_id: str) -> tuple[LabTask, Path]:
    os.environ["LAB_ROOT"] = str(lab_root)
    from lab_core.harness.run import load_task as lab_load_task

    t = lab_load_task(task_id)
    cfg = t["config"]
    task = LabTask(id=task_id, title=cfg["title"], instructions=t["instructions"], work_type=cfg.get("work_type"),
                   deliverables=cfg.get("deliverables", {}), criteria=cfg["criteria"])
    return task, Path(t["docs_dir"])


# --- the sandbox -----------------------------------------------------------------------------------------------


def _make_pod_sandbox_class():
    """Built lazily so importing this module doesn't need lab_core."""
    from lab_core.sandbox.sandbox import WORKSPACE_PATH, ExecResult, Sandbox

    # {{docs-fragment pod_sandbox}}
    class PodSandbox(Sandbox):
        """LAB's Sandbox interface, backed by a private directory in this pod instead of a Podman container.

        The agent's tools only ever see the canonical paths (/workspace, /workspace/documents, /workspace/output).
        File tools go through Sandbox's own path mapping; shell commands get those paths rewritten to this
        rollout's directory on the way in, and the directory rewritten back to /workspace on the way out.
        """

        def start(self) -> None:
            self.container_name = f"pod-{self.workspace_dir.name}"
            self._started = True

        def stop(self) -> None:
            self.container_name = None

        def exec(self, command: str, *, cwd: str = WORKSPACE_PATH, timeout: int | None = None,
                 env: dict[str, str] | None = None) -> ExecResult:
            self.assert_sandbox_path(cwd)
            timeout = timeout if timeout is not None else self.default_timeout
            host_ws = str(self.workspace_dir)
            cmd = command.replace(WORKSPACE_PATH, host_ws)
            full_env = {**os.environ, "DOCUMENTS_DIR": str(self.documents_dir), "OUTPUT_DIR": str(self.output_dir),
                        "WORKSPACE_DIR": host_ws, "HOME": host_ws, **self.extra_env, **(env or {})}
            try:
                r = subprocess.run(["timeout", "--kill-after=2", str(timeout), "bash", "-lc", cmd],
                                   cwd=str(self._to_host(cwd)), env=full_env, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=timeout + 5)
            except subprocess.TimeoutExpired as e:
                return ExecResult(stdout=str(e.stdout or ""), stderr=str(e.stderr or ""), returncode=None,
                                  timed_out=True)
            out, err = (s.replace(host_ws, WORKSPACE_PATH) for s in (r.stdout, r.stderr))
            if r.returncode in self._TIMEOUT_EXITS:
                return ExecResult(stdout=out, stderr=err, returncode=None, timed_out=True)
            return ExecResult(stdout=out, stderr=err, returncode=r.returncode, timed_out=False)
    # {{/docs-fragment pod_sandbox}}

    return PodSandbox


# --- the policy ------------------------------------------------------------------------------------------------


class Policy(BaseModel):
    """A model behind an OpenAI-compatible endpoint. ``name`` is what the server calls it (a LoRA adapter name works)."""

    name: str
    hf_id: str
    temperature: float = 1.0
    max_tokens: int = 8192
    reasoning_effort: str | None = None  # passed through to models that take it (gpt-oss: low | medium | high)


class Harness(BaseModel):
    """Everything between the model and the matter. Season 1's lesson: version it like weights."""

    version: str = "lab-1.4.0"
    max_turns: int = 40
    shell_timeout: int = 60
    max_result_chars: int | None = 40_000  # LAB's default is None (whole results); small contexts need a cap
    enable_finish: bool = True
    skills: list[str] = ["docx", "pptx", "xlsx"]  # LAB's default: every skill manual in the system prompt
    max_minutes: float | None = None  # wall-clock limit per episode (Harvey: one hour); hitting it counts as cut off


class TurnTokens(BaseModel):
    """One assistant turn as the server saw it: what's needed to train on it token-in/token-out."""

    prompt_ids: list[int]
    completion_ids: list[int]
    logprobs: list[float]  # empty when the server's logprobs didn't line up with completion_ids
    logprob_ids_matched: bool = True


class Rollout(BaseModel):
    rollout_id: str
    task_id: str
    policy: str
    harness: str
    sample: int
    finish_reason: str
    turns: int
    input_tokens: int
    output_tokens: int
    wall_s: float
    tool_metrics: dict[str, Any] = Field(default_factory=dict)
    outputs_b64: str = ""  # tar.gz of the output directory: the deliverables exactly as written
    deliverable_chars: int = 0
    error: str | None = None
    last_text: str = ""  # the final assistant message, clipped: what a model that stops calling tools said instead
    tokens: list[TurnTokens] = Field(default_factory=list)


class EpisodeTimeLimit(RuntimeError):
    pass


def _token_capturing_adapter(policy: Policy, base_url: str, capture: bool, deadline: float | None = None):
    """LAB's OpenAI-compatible adapter, asking vLLM for the token ids and logprobs it sampled."""
    from lab_core.harness.adapters.base import ModelResponse, ToolCall
    from lab_core.harness.adapters.baseten import BasetenAdapter

    class CapturingAdapter(BasetenAdapter):
        def __init__(self):
            super().__init__(model=policy.name, temperature=policy.temperature, max_tokens=policy.max_tokens,
                             reasoning_effort=None, base_url=base_url, api_key="local")
            self.turns: list[TurnTokens] = []

        def chat(self, messages, tools):
            if deadline is not None and time.time() > deadline:
                raise EpisodeTimeLimit("episode time limit")
            extra: dict[str, Any] = {}
            if policy.reasoning_effort:
                extra["reasoning_effort"] = policy.reasoning_effort
            if capture:
                extra.update(return_token_ids=True, return_tokens_as_token_ids=True)
            kwargs = dict(model=self.model, messages=messages, tools=[self._translate_tool(t) for t in tools],
                          temperature=self.temperature, max_tokens=self.max_tokens, extra_body=extra)
            if capture:
                kwargs.update(logprobs=True, top_logprobs=0)
            for attempt in range(5):
                try:
                    response = self.client.chat.completions.create(**kwargs)
                    break
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    if "maximum context length" in msg or "context_length_exceeded" in msg or "too long" in msg:
                        raise RuntimeError(f"context_length_exceeded: {msg[:300]}") from e
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            choice = response.choices[0]
            msg = choice.message
            message = msg.model_dump(exclude_none=True)
            # vLLM returns reasoning as `reasoning_content`; keep it out of the history LAB replays, as LAB does
            # for hosted reasoning models, but the token ids below still contain it.
            message.pop("reasoning_content", None)
            message.pop("reasoning", None)
            if capture:
                extra_c = choice.model_extra or {}
                comp_ids = extra_c.get("token_ids") or []
                prompt_ids = (response.model_extra or {}).get("prompt_token_ids") or []
                content = (choice.logprobs.content or []) if choice.logprobs else []
                lp_ids = [int(c.token.split(":", 1)[1]) if c.token.startswith("token_id:") else -1 for c in content]
                # Keep the sampler's logprobs only if they cover exactly the tokens it returned. With a reasoning
                # parser they can cover only part of the completion; then the trainer uses its own (ratio = 1).
                lps = [c.logprob for c in content] if lp_ids == comp_ids else []
                self.turns.append(TurnTokens(prompt_ids=prompt_ids, completion_ids=comp_ids, logprobs=lps,
                                             logprob_ids_matched=lp_ids == comp_ids))
            calls = [ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
                     for tc in (msg.tool_calls or [])]
            u = response.usage
            return ModelResponse(message=message, tool_calls=calls, text=msg.content or "",
                                 input_tokens=u.prompt_tokens if u else 0, output_tokens=u.completion_tokens if u else 0,
                                 finish_reason=choice.finish_reason)

    return CapturingAdapter()


def _last_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            c = m.get("content") or ""
            calls = [tc.get("function", {}).get("name") for tc in m.get("tool_calls") or []]
            return (c if isinstance(c, str) else json.dumps(c))[:1500] + (f" [tool calls: {calls}]" if calls else "")
    return ""


def _tar_b64(d: Path) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted(d.rglob("*")):
            if p.is_file():
                tf.add(p, arcname=str(p.relative_to(d)))
    return base64.b64encode(buf.getvalue()).decode()


def untar_b64(blob: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    if blob:
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(blob)), mode="r:gz") as tf:
            tf.extractall(dest, filter="data")
    return dest


# {{docs-fragment rollout}}
def run_rollout(lab_root: Path, task_id: str, policy: Policy, harness: Harness, base_url: str, sample: int,
                capture_tokens: bool = False) -> Rollout:
    """One agent episode on one LAB task: LAB's system prompt, skills, tools and agent loop, in a private workspace."""
    from lab_core.harness.agent_loop import run_agent
    from lab_core.harness.run import build_system_preamble, load_skills, setup_skill_scripts
    from lab_core.harness.tools import ToolExecutor, get_all_tool_definitions

    task, docs_dir = load_task(lab_root, task_id)
    rid = f"{task_id.replace('/', '__')}__{policy.name}__{harness.version}__s{sample}__{uuid.uuid4().hex[:6]}"
    ws = POD_WS_ROOT / rid
    (ws / "output").mkdir(parents=True, exist_ok=True)
    shutil.copytree(docs_dir, ws / "documents")
    subprocess.run(["chmod", "-R", "a-w", str(ws / "documents")], check=False)
    setup_skill_scripts(harness.skills, ws)

    sandbox = _make_pod_sandbox_class()(documents_dir=ws / "documents", output_dir=ws / "output", workspace_dir=ws,
                                        default_timeout=harness.shell_timeout)
    sandbox.start()
    executor = ToolExecutor(sandbox=sandbox, shell_timeout=harness.shell_timeout, enable_finish=harness.enable_finish,
                            max_result_chars=harness.max_result_chars)
    deadline = time.time() + 60 * harness.max_minutes if harness.max_minutes else None
    adapter = _token_capturing_adapter(policy, base_url, capture_tokens, deadline)
    system_prompt = build_system_preamble(harness.enable_finish) + (load_skills(harness.skills) if harness.skills else "")
    t0 = time.time()
    try:
        res = run_agent(adapter, system_prompt, task.instructions, executor,
                        tools=get_all_tool_definitions(enable_finish=harness.enable_finish), max_turns=harness.max_turns)
        finish, err = res["finish_reason"], None
    except RuntimeError as e:  # the context window filled up (LAB: context_overflow), or the clock ran out
        if isinstance(e, EpisodeTimeLimit):
            finish = "time_limit"
        elif "context_length_exceeded" in str(e):
            finish = "context_overflow"
        else:
            raise
        res, err = {"turn_count": len(adapter.turns), "input_tokens": 0, "output_tokens": 0,
                    "tool_metrics": executor.get_metrics()}, str(e)[:300]
    out = ws / "output"
    rollout = Rollout(rollout_id=rid, task_id=task_id, policy=policy.name, harness=harness.version, sample=sample,
                      finish_reason=finish, turns=res["turn_count"], input_tokens=res["input_tokens"],
                      output_tokens=res["output_tokens"], wall_s=round(time.time() - t0, 1),
                      tool_metrics=res.get("tool_metrics", {}), outputs_b64=_tar_b64(out),
                      deliverable_chars=sum(p.stat().st_size for p in out.rglob("*") if p.is_file()), error=err,
                      last_text=_last_text(res.get("messages", [])), tokens=adapter.turns)
    subprocess.run(["chmod", "-R", "u+w", str(ws)], check=False)
    shutil.rmtree(ws, ignore_errors=True)
    return rollout
# {{/docs-fragment rollout}}


# --- the judge -------------------------------------------------------------------------------------------------

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "verdict": {"type": "string", "enum": ["pass", "fail"]}},
    "required": ["reasoning", "verdict"],
    "additionalProperties": False,
}


class JudgeSpec(BaseModel):
    """Who grades. ``kind="openai_compat"`` is an open model on a server we run; ``"lab"`` is lab_core's own Judge."""

    name: str
    kind: str = "openai_compat"  # openai_compat | lab
    hf_id: str = ""
    temperature: float = 0.0
    reasoning_effort: str | None = None
    thinking: bool | None = None  # for models with a thinking switch in their chat template (Nemotron 3); None = default
    max_tokens: int = 4096


class OpenAICompatJudge:
    """LAB's Judge interface (``evaluate_from_file``) over chat completions with a JSON-schema response format."""

    def __init__(self, spec: JudgeSpec, base_url: str):
        import openai

        self.spec, self.model = spec, spec.name
        self.client = openai.OpenAI(base_url=base_url, api_key="local", timeout=900, max_retries=4)

    def evaluate_from_file(self, prompt_name: str, variables: dict) -> dict:
        from lab_core.evaluation.judge import PROMPTS_DIR, Judge

        template = (PROMPTS_DIR / f"{prompt_name}.txt").read_text(encoding="utf-8")
        prompt = template.format(**variables)
        extra: dict[str, Any] = {"structured_outputs": {"json": _VERDICT_SCHEMA}}
        if self.spec.reasoning_effort:
            extra["reasoning_effort"] = self.spec.reasoning_effort
        if self.spec.thinking is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": self.spec.thinking}
        last: Exception | None = None
        for _ in range(3):
            try:
                r = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}],
                                                        temperature=self.spec.temperature,
                                                        max_tokens=self.spec.max_tokens, extra_body=extra)
                return Judge._parse_json(r.choices[0].message.content or "")
            except Exception as e:  # noqa: BLE001
                last = e
        raise ValueError(f"judge failed after 3 attempts: {last}")


class Verdicts(BaseModel):
    rollout_id: str
    task_id: str
    judge: str
    judge_sample: int = 0
    verdicts: dict[str, str]  # criterion id -> pass | fail | error
    reasoning: dict[str, str] = Field(default_factory=dict)


def grade(rollout: Rollout, task: LabTask, judge, judge_name: str, parallel: int = 16,
          judge_sample: int = 0) -> Verdicts:
    """LAB's grader (``score_rubric``) on one rollout's deliverables."""
    from lab_core.evaluation.scoring import score_rubric

    run_dir = POD_WS_ROOT / f"grade-{rollout.rollout_id}-{judge_name}-{judge_sample}-{uuid.uuid4().hex[:4]}"
    untar_b64(rollout.outputs_b64, run_dir / "output")
    try:
        res = score_rubric(task.criteria, run_dir, judge, task.title, parallel)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    return Verdicts(rollout_id=rollout.rollout_id, task_id=task.id, judge=judge_name, judge_sample=judge_sample,
                    verdicts={c["id"]: c["verdict"] for c in res.criteria_results},
                    reasoning={c["id"]: (c.get("reasoning") or "")[:400] for c in res.criteria_results})


# --- the reward ------------------------------------------------------------------------------------------------


class RewardWeights(BaseModel):
    """Harvey's published reward: 0.35 criteria + 0.35 issue groups + 0.30 all-pass, zero if the episode is cut off."""

    criteria: float = 0.35
    groups: float = 0.35
    all_pass: float = 0.30
    zero_on_truncation: bool = True


TRUNCATED = {"max_turns_exceeded", "context_overflow", "time_limit"}


# {{docs-fragment reward}}
def reward_terms(verdicts: dict[str, str], issue_groups: dict[str, list[str]]) -> dict[str, float]:
    """The three terms before weighting. ``issue_groups`` maps an issue to the criterion ids that make it up."""
    ids = list(verdicts)
    passed = {i for i in ids if verdicts[i] == "pass"}
    c = len(passed) / len(ids) if ids else 0.0
    groups = [g for g in issue_groups.values() if g]
    h = sum(all(i in passed for i in g) for g in groups) / len(groups) if groups else c
    a = float(bool(ids) and len(passed) == len(ids))
    return {"c": c, "h": h, "a": a}


def reward(terms: dict[str, float], finish_reason: str, w: RewardWeights = RewardWeights()) -> float:
    if w.zero_on_truncation and finish_reason in TRUNCATED:
        return 0.0
    return w.criteria * terms["c"] + w.groups * terms["h"] + w.all_pass * terms["a"]
# {{/docs-fragment reward}}


# --- issue groups ----------------------------------------------------------------------------------------------

_GROUP_PROMPT = """You are organizing a legal grading rubric. Below are the rubric criteria for one task, each with an id
and a title. Group the criteria by the underlying legal issue they check (for example, every criterion about one
contract's change-of-control consent is one issue). Every criterion id must appear in exactly one group. Prefer
groups of 2-6 criteria; a criterion about formatting or overall structure goes in a group named "form".

Task: {title}

Criteria:
{criteria}

Respond with JSON: {{"groups": {{"<short issue name>": ["C-001", ...], ...}}}}"""


def group_issues(task: LabTask, client, model: str) -> dict[str, list[str]]:
    """LAB's public rubrics have no issue-group field, so an LLM proposes groups from the criterion titles.

    This is our stand-in for Harvey's "issue groups", not theirs. Ungrouped or duplicated ids are repaired so every
    criterion lands in exactly one group.
    """
    lines = "\n".join(f"{c['id']}: {c['title']}" for c in task.criteria)
    schema = {"type": "object", "properties": {"groups": {"type": "object", "additionalProperties": {
        "type": "array", "items": {"type": "string"}}}}, "required": ["groups"]}
    r = client.chat.completions.create(model=model, temperature=0.0, max_tokens=8192,
                                       messages=[{"role": "user", "content": _GROUP_PROMPT.format(title=task.title,
                                                                                                  criteria=lines)}],
                                       extra_body={"structured_outputs": {"json": schema}})
    from lab_core.evaluation.judge import Judge

    raw = Judge._parse_json(r.choices[0].message.content or "{}").get("groups", {})
    valid, seen, out = {c["id"] for c in task.criteria}, set(), {}
    for name, ids in raw.items():
        keep = [i for i in ids if i in valid and i not in seen]
        seen.update(keep)
        if keep:
            out[re.sub(r"\s+", " ", name)[:80]] = keep
    for i in sorted(valid - seen):
        out[f"ungrouped {i}"] = [i]
    return out
