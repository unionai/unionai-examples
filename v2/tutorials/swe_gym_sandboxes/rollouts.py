"""Exact-token multi-turn SWE-Gym trajectories, graded in fresh sandboxes.

A port of Modal's `helpers/rollouts.py` (generate_rollout and grade_rollout). The rollout
logic is the same: Qwen's chat template, one bash tool, the XML tool-call format, 120 s a
command, the last 20,000 characters of its output, token masks and behavior logprobs, and the
four ways an episode stops (turn_limit, context_limit, generation_limit, final). Two changes:

- It is async: `await environments.command(...)` instead of a blocking call in a thread.
- The sampler is pluggable (`Sampler` below): Tinker as in Modal's example, any
  OpenAI-compatible completions endpoint (vLLM, SGLang), or `OracleSampler`, which needs no
  model and replays the gold patch, to check the sandboxes and grading end to end.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import math
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Protocol

from union_environments import InfrastructureError

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command in /testbed to inspect, edit, and test the repository.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]
SYSTEM = """You are a software engineer fixing a repository issue. Use the bash tool
to inspect and edit /testbed, and run relevant tests. The testbed conda environment
is activated for every command. All shell commands are isolated and have no network
access. Make a minimal correct source fix; do not weaken or delete tests. When the
fix is complete, respond with a short final explanation without a tool call."""

MODEL = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

CONFIG = {"context_tokens": 131_072, "turn_tokens": 8192, "max_turns": 20, "temperature": 1.0}
SMOKE_CONFIG = {"context_tokens": 131_072, "turn_tokens": 8192, "max_turns": 4, "temperature": 1.0}


def parse_commands(text):
    text = text.split("</think>")[-1]
    calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    result = []
    for call in calls:
        match = re.fullmatch(
            r"<function=bash>\s*<parameter=command>\s*(.*?)\s*</parameter>\s*</function>",
            call,
            re.DOTALL,
        )
        if not match:
            raise ValueError("Expected bash(command) in the documented XML tool format")
        result.append(match.group(1))
    if "<tool_call>" in text and not calls:
        raise ValueError("Incomplete tool call")
    return result


def generation_budget(cfg, prompt_tokens):
    return min(cfg["turn_tokens"], cfg["context_tokens"] - prompt_tokens - 16)


# ---------------------------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------------------------


@dataclass
class Sample:
    tokens: List[int]
    logprobs: List[float]
    stop_reason: str


class Sampler(Protocol):
    def bind(self, inst: Dict[str, Any], identity: Dict[str, Any]) -> "Sampler":
        """The sampler for one rollout (most return themselves)."""

    async def sample(self, tokens: List[int], max_tokens: int, temperature: float) -> Sample: ...


class TinkerSampler:
    """Tinker (or Modal's Spindle, which speaks its API), as in Modal's example.
    Reads TINKER_API_KEY and TINKER_BASE_URL from the driver's environment."""

    def __init__(self, base_model: str = MODEL, model_path: Optional[str] = None) -> None:
        import tinker  # noqa: PLC0415

        self._types = tinker.types
        self.client = tinker.ServiceClient().create_sampling_client(base_model=base_model, model_path=model_path)

    def bind(self, inst, identity):
        return self

    async def sample(self, tokens, max_tokens, temperature):
        t = self._types
        future = self.client.sample(
            t.ModelInput.from_ints(tokens),
            1,
            t.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=1.0),
        )
        seq = (await asyncio.to_thread(future.result, 3600)).sequences[0]
        return Sample(list(seq.tokens), list(seq.logprobs or []), str(seq.stop_reason))


class OpenAICompatSampler:
    """Any OpenAI-compatible `/v1/completions` endpoint that takes token ids as the prompt and
    returns token ids with their logprobs: vLLM (`return_tokens_as_token_ids`) and SGLang do.
    Reads OPENAI_API_KEY from the driver's environment, if the endpoint needs one."""

    def __init__(self, base_url: str, model: str = MODEL, api_key: Optional[str] = None) -> None:
        import os  # noqa: PLC0415

        import httpx  # noqa: PLC0415

        key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.model = model
        self.http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers={"Authorization": f"Bearer {key}"}, timeout=3600
        )

    def bind(self, inst, identity):
        return self

    async def sample(self, tokens, max_tokens, temperature):
        r = await self.http.post(
            "/completions",
            json={
                "model": self.model,
                "prompt": tokens,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 1.0,
                "logprobs": 0,
                "return_tokens_as_token_ids": True,
                "skip_special_tokens": False,
            },
        )
        r.raise_for_status()
        choice = r.json()["choices"][0]
        lp = choice.get("logprobs") or {}
        ids = [int(t.split(":", 1)[1]) for t in lp.get("tokens", []) if t.startswith("token_id:")]
        if len(ids) != len(lp.get("tokens", [])) or not ids:
            raise InfrastructureError("endpoint did not return token ids (vLLM: --return-tokens-as-token-ids)")
        return Sample(ids, [float(x) for x in lp["token_logprobs"]], choice.get("finish_reason") or "")


class OracleSampler:
    """No model: a scripted agent, to test the sandboxes and grading without a GPU or a key.

    Attempts with an even number look around and then apply the instance's gold patch, so they
    should score 1; odd ones look around and give up, so they should score 0. A run whose
    rewards come out 1, 0, 1, 0 shows the testbed, the forks, the patch capture and the hidden
    tests all work. Logprobs are 0.0: these tokens are not from a policy.
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.inst: Dict[str, Any] = {}
        self.solve = False
        self.turn = 0

    def bind(self, inst, identity):
        bound = OracleSampler(self.tokenizer)
        bound.inst, bound.solve = inst, identity["attempt"] % 2 == 0
        return bound

    @staticmethod
    def _call(cmd: str) -> str:
        return f"<tool_call>\n<function=bash>\n<parameter=command>\n{cmd}\n</parameter>\n</function>\n</tool_call>"

    async def sample(self, tokens, max_tokens, temperature):
        await asyncio.sleep(0.05)  # stands in for the model
        turn, self.turn = self.turn, self.turn + 1
        if turn == 0:
            text = "Let me look at the repository.</think>\n" + self._call(
                "git status --short && git log --oneline -1 && ls"
            )
        elif turn == 1 and self.solve:
            patch = self.inst["patch"]
            text = "I found the bug and will fix it.</think>\n" + self._call(
                f"cat > /tmp/fix.patch <<'EOF_FIX'\n{patch}\nEOF_FIX\ngit apply /tmp/fix.patch && git diff --stat"
            )
        elif turn == 1:
            text = "</think>\nI could not find the cause of this issue."
        else:
            text = "</think>\nThe fix is applied: the change makes the failing case behave as the issue asks."
        ids = [
            *self.tokenizer.encode(text, add_special_tokens=False),
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        ]
        return Sample(ids[:max_tokens], [0.0] * min(len(ids), max_tokens), "stop")


class ChatMLByteTokenizer:
    """A stand-in for Qwen's tokenizer when there is no model: bytes as tokens, plus ChatML's
    special tokens, with the same chat template shape. Enough for OracleSampler; a real run
    loads Qwen's tokenizer (`load_tokenizer`)."""

    SPECIAL: ClassVar[Dict[str, int]] = {"<|im_start|>": 256, "<|im_end|>": 257}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.SPECIAL[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        out: List[int] = []
        for part in re.split(r"(<\|im_start\|>|<\|im_end\|>)", text):
            out.extend([self.SPECIAL[part]] if part in self.SPECIAL else part.encode())
        return out

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        names = {v: k for k, v in self.SPECIAL.items()}
        out, buf = [], bytearray()
        for i in ids:
            if i in names:
                out.append(buf.decode(errors="replace"))
                buf = bytearray()
                if not skip_special_tokens:
                    out.append(names[i])
            else:
                buf.append(i)
        out.append(buf.decode(errors="replace"))
        return "".join(out)

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=True, return_dict=False):
        text = ""
        for m in messages:
            content = m["content"]
            if m["role"] == "system" and tools:
                content += "\n\n# Tools\n<tools>\n" + "\n".join(json.dumps(t) for t in tools) + "\n</tools>"
            text += f"<|im_start|>{m['role']}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n<think>\n"
        return self.encode(text) if tokenize else text


def load_tokenizer(name: str = MODEL, revision: str = MODEL_REVISION):
    from transformers import AutoTokenizer  # noqa: PLC0415

    return AutoTokenizer.from_pretrained(name, revision=revision)


def make_sampler(kind: str, *, base_url: Optional[str] = None, model: str = MODEL):
    """(sampler, tokenizer) for `--sampler tinker|openai|oracle`."""
    if kind == "oracle":
        tok = ChatMLByteTokenizer()
        return OracleSampler(tok), tok
    if kind == "tinker":
        return TinkerSampler(model), load_tokenizer(model)
    if kind == "openai":
        if not base_url:
            raise SystemExit("--sampler openai needs --base-url (e.g. http://my-vllm:8000/v1)")
        return OpenAICompatSampler(base_url, model), load_tokenizer(model)
    raise SystemExit(f"unknown sampler {kind!r}")


# ---------------------------------------------------------------------------------------------
# The rollout
# ---------------------------------------------------------------------------------------------


class EventLog:
    """Modal's `log`: JSON lines under `root`, one event per line."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._f = open(self.root / "events.jsonl", "a")

    def event(self, kind: str, **fields) -> None:
        self._f.write(json.dumps({"t": time.time(), "event": kind, **fields}, default=str) + "\n")
        self._f.flush()


async def generate_rollout(inst, sampler, tokenizer, environments, cfg, log, identity):
    start = time.time()
    sampler = sampler.bind(inst, identity)
    log.event("rollout_start", **identity, instance_id=inst["instance_id"])
    path = (
        log.root
        / "episodes"
        / identity["phase"]
        / str(identity["job"])
        / f"g{identity['group']:06d}-a{identity['attempt']}"
    )
    path.mkdir(parents=True, exist_ok=True)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": inst["problem_statement"]},
    ]
    tokens = list(
        tokenizer.apply_chat_template(
            messages, tools=TOOLS, tokenize=True, add_generation_prompt=True, return_dict=False
        )
    )
    behavior = [0.0] * len(tokens)
    mask = [0] * len(tokens)
    turns = []
    termination = "turn_limit"
    async with environments.sandbox(inst) as sb:
        await environments.prepare_agent(sb)
        for turn in range(cfg["max_turns"]):
            budget = generation_budget(cfg, len(tokens))
            if budget < 1:
                termination = "context_limit"
                break
            sample_start = time.time()
            sample = await sampler.sample(tokens, budget, cfg["temperature"])
            generated, lp = list(sample.tokens), list(sample.logprobs or [])
            if not generated or len(lp) != len(generated) or not all(math.isfinite(x) for x in lp):
                raise InfrastructureError("Missing/nonfinite sampling log probabilities")
            sample_s = time.time() - sample_start
            text = tokenizer.decode(generated, skip_special_tokens=False)
            row = {
                "turn": turn,
                "prompt_tokens": len(tokens),
                "tokens": len(generated),
                "sample_s": sample_s,
                "text": text,
                "stop_reason": str(sample.stop_reason),
            }
            turns.append(row)
            log.event("sample", **identity, **{k: v for k, v in row.items() if k != "text"})
            tokens.extend(generated)
            behavior.extend(lp)
            mask.extend([1] * len(generated))
            if len(generated) >= budget:
                termination = "generation_limit"
                break
            try:
                cmds = parse_commands(text)
                feedback = []
                for cmd in cmds:
                    ts = time.time()
                    rc, output = await environments.command(
                        sb,
                        "source /opt/miniconda3/bin/activate testbed && cd /testbed && bash -c " + shlex.quote(cmd),
                        120,
                    )
                    feedback.append(f"exit_code={rc}\n{output[-20000:]}")
                    log.event("tool", **identity, turn=turn, elapsed_s=time.time() - ts, exit_code=rc)
                if not cmds:
                    termination = "final"
                    break
            except ValueError as exc:
                feedback = [str(exc)]
            suffix = "" if tokens[-1] == tokenizer.convert_tokens_to_ids("<|im_end|>") else "<|im_end|>"
            suffix += "\n<|im_start|>user"
            for output in feedback:
                suffix += "\n<tool_response>\n" + output.strip() + "\n</tool_response>"
            suffix += "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            obs = tokenizer.encode(suffix, add_special_tokens=False)
            tokens.extend(obs)
            behavior.extend([0.0] * len(obs))
            mask.extend([0] * len(obs))
        patch = await environments.patch(sb)
    while mask and not mask[-1]:
        tokens.pop()
        behavior.pop()
        mask.pop()
    episode = {
        **identity,
        "instance_id": inst["instance_id"],
        "tokens": tokens,
        "behavior_logprobs": behavior,
        "mask": mask,
        "turns": turns,
        "termination": termination,
        "generated_tokens": sum(mask),
    }
    (path / "prediction.patch").write_text(patch)
    with gzip.open(path / "ungraded.json.gz", "wt") as f:
        json.dump(episode, f)
    ready = time.time()
    log.event("rollout_ready", **identity, instance_id=inst["instance_id"], generation_s=ready - start)
    return {
        "path": str(path),
        "identity": identity,
        "started_at": start,
        "ready_at": ready,
        "generation_s": ready - start,
    }


async def grade_rollout(prepared, inst, environments, log):
    path, identity = Path(prepared["path"]), prepared["identity"]
    grade_start = time.time()
    queue_s = grade_start - prepared["ready_at"]
    log.event("grading_start", **identity, instance_id=inst["instance_id"], grading_queue_s=queue_s)
    verdict = await environments.grade(inst, (path / "prediction.patch").read_text(), path)
    episode = load_episode(path / "ungraded.json.gz")
    episode.update(
        reward=verdict["reward"],
        elapsed_s=time.time() - prepared["started_at"],
        generation_s=prepared["generation_s"],
        grading_queue_s=queue_s,
        grading_s=time.time() - grade_start,
    )
    trace = path / "trajectory.json.gz"
    with gzip.open(trace, "wt") as f:
        json.dump(episode, f)
    log.event(
        "episode",
        **identity,
        instance_id=inst["instance_id"],
        reward=episode["reward"],
        termination=episode["termination"],
        elapsed_s=episode["elapsed_s"],
        trace=str(trace),
    )
    return summarize(episode, str(trace))


def summarize(episode: Dict[str, Any], trace: str = "") -> Dict[str, Any]:
    """One attempt's row: what the drivers print."""
    return {
        "instance_id": episode["instance_id"],
        "attempt": episode["attempt"],
        "reward": episode["reward"],
        "turns": len(episode["turns"]),
        "termination": episode["termination"],
        "generated_tokens": episode["generated_tokens"],
        "trace": trace,
    }


def load_episode(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def print_table(rows: List[Dict[str, Any]]) -> None:
    """Each attempt and its reward, then the mean."""
    print(f"\n{'instance':32} {'attempt':>7} {'reward':>6} {'turns':>5}  stop")
    for r in sorted(rows, key=lambda r: (r["instance_id"], r["attempt"])):
        print(f"{r['instance_id']:32} {r['attempt']:>7} {r['reward']:6.1f} {r['turns']:5d}  {r['termination']}")
    if rows:
        print(f"\nmean reward {sum(r['reward'] for r in rows) / len(rows):.2f} over {len(rows)} attempts")
