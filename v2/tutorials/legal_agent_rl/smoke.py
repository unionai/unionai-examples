# /// script
# requires-python = ">=3.12"
# dependencies = ["flyte>=2.13.1", "pydantic", "kubernetes"]
# ///
"""Smoke test on one GPU node: every server in the sweep layout, one LAB rollout per policy, LAB's grader with each
open-weight judge, and one LoRA gradient step on captured tokens.

    CLUSTER=amd GPU_QUEUE=<queue> GPU_IMAGE_URI=<uri> python smoke.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import flyte

import lab_harness as lh
import servers
from config import CLUSTER, PROPAGATE, gpu_env_kwargs
from images import LAB_COMMIT

env = flyte.TaskEnvironment(name="legal-rl-smoke", env_vars=PROPAGATE, **gpu_env_kwargs())

TASKS = ["trusts-estates-private-client/compare-trust-documents-against-client-instructions",
         "immigration/compare-draft-eb"]
S = servers.ServerSpec
SERVERS = [
    S(served_name="gpt-oss-120b", hf_id="openai/gpt-oss-120b", gpus=[0, 1], max_model_len=131072),
    S(served_name="gpt-oss-20b", hf_id="openai/gpt-oss-20b", gpus=[2], max_model_len=131072),
    S(served_name="nemotron-3-nano-30b", hf_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", gpus=[3],
      max_model_len=131072),
]
EFFORT = {"gpt-oss-120b": "medium", "gpt-oss-20b": "medium"}


@env.task
async def smoke(task_ids: list[str] = TASKS, max_turns: int = 40, train_gpu: int = 7) -> str:
    out: dict = {"cluster": CLUSTER, "node": os.getenv("NODE_NAME"), "gpus": servers.gpu_report()}
    t0 = time.time()
    lab = lh.fetch_lab(task_ids, LAB_COMMIT, Path("/tmp/lab"))
    out["fetch_s"] = round(time.time() - t0, 1)
    out["pandoc"] = subprocess.run(["pandoc", "--version"], capture_output=True, text=True).stdout.split("\n")[0]

    async def start(s):
        try:
            return s.served_name, {"ok": True, "load_s": round(await servers.ensure(s, timeout_s=2400), 1)}
        except Exception as e:  # noqa: BLE001
            return s.served_name, {"ok": False, "error": f"{type(e).__name__}: {e}"[:500]}

    out["servers"] = dict(await asyncio.gather(*[start(s) for s in SERVERS]))
    up = [s for s in SERVERS if out["servers"][s.served_name]["ok"]]
    harness = lh.Harness(max_turns=max_turns)

    async def roll(s, t):
        p = lh.Policy(name=s.served_name, hf_id=s.hf_id, reasoning_effort=EFFORT.get(s.served_name))
        try:
            return await asyncio.to_thread(lh.run_rollout, lab, t, p, harness, s.url, 0, s.served_name == "gpt-oss-20b")
        except Exception:  # noqa: BLE001
            out.setdefault("rollout_errors", []).append({"policy": s.served_name, "task": t,
                                                         "tb": traceback.format_exc()[-1500:]})
            return None

    rollouts = [r for r in await asyncio.gather(*[roll(s, t) for s in up for t in task_ids]) if r]
    judges = [s for s in up if s.served_name in ("gpt-oss-120b", "nemotron-3-nano-30b")]
    rows = []
    for r in rollouts:
        task, _ = lh.load_task(lab, r.task_id)
        row = {"policy": r.policy, "task": r.task_id, "finish": r.finish_reason, "turns": r.turns,
               "out_tokens": r.output_tokens, "wall_s": r.wall_s, "deliverable_bytes": r.deliverable_chars,
               "tools": r.tool_metrics, "error": r.error, "captured_turns": len(r.tokens), "last_text": r.last_text[:700]}
        for j in judges:
            spec = lh.JudgeSpec(name=j.served_name, hf_id=j.hf_id, reasoning_effort=EFFORT.get(j.served_name) and "low")
            try:
                v = await asyncio.to_thread(lh.grade, r, task, lh.OpenAICompatJudge(spec, j.url), j.served_name, 16)
                row[f"judge:{j.served_name}"] = {"pass": sum(x == "pass" for x in v.verdicts.values()),
                                                 "error": sum(x == "error" for x in v.verdicts.values()),
                                                 "n": len(v.verdicts),
                                                 "sample_reason": next(iter(v.reasoning.values()), "")[:200]}
            except Exception as e:  # noqa: BLE001
                row[f"judge:{j.served_name}"] = f"{type(e).__name__}: {e}"[:400]
        rows.append(row)
    out["rollouts"] = rows

    # Issue grouping, once, with the big judge.
    if judges:
        import openai

        task, _ = lh.load_task(lab, task_ids[0])
        try:
            g = lh.group_issues(task, openai.OpenAI(base_url=judges[0].url, api_key="x"), judges[0].served_name)
            out["groups_example"] = {"n_criteria": len(task.criteria), "n_groups": len(g), "groups": dict(list(g.items())[:5])}
        except Exception as e:  # noqa: BLE001
            out["groups_example"] = f"{type(e).__name__}: {e}"[:400]

    # One LoRA step on gpt-oss-20b's captured tokens, on a GPU no server uses.
    try:
        os.environ[servers.VISIBLE] = str(train_gpu) if CLUSTER == "amd" else "0"
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        t1 = time.time()
        hf = "openai/gpt-oss-20b"
        from transformers import Mxfp4Config
        model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, attn_implementation="eager",
                                                     quantization_config=Mxfp4Config(dequantize=True)).to("cuda")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=64, lora_alpha=64, target_modules="all-linear",
                                                 target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]))
        turns = [t for r in rollouts if r.policy == "gpt-oss-20b" for t in r.tokens if t.completion_ids]
        turn = max(turns, key=lambda t: len(t.prompt_ids)) if turns else None
        if turn is not None:
            ids = torch.tensor([turn.prompt_ids + turn.completion_ids], device="cuda")
            logits = model(ids).logits[0, len(turn.prompt_ids) - 1:-1].float()
            lp = torch.log_softmax(logits, -1).gather(-1, ids[0, len(turn.prompt_ids):, None])[:, 0]
            old = torch.tensor(turn.logprobs[:lp.numel()], device="cuda")
            (-(lp[:old.numel()]).sum() / 55_000).backward()
            out["train"] = {"seq_len": ids.shape[1], "completion_len": len(turn.completion_ids),
                            "mean_abs_logprob_gap": float((lp[:old.numel()].detach() - old).abs().mean()),
                            "step_s": round(time.time() - t1, 1), "device": torch.cuda.get_device_name(0),
                            "max_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    except Exception:  # noqa: BLE001
        out["train_error"] = traceback.format_exc()[-2000:]
    import glob
    out["server_errors"] = {s.served_name: servers.log_errors(s.served_name) for s in SERVERS}
    out["total_s"] = round(time.time() - t0, 1)
    print(json.dumps(out, indent=1), flush=True)
    return json.dumps(out)


if __name__ == "__main__":
    flyte.init_from_config(os.getenv("FLYTE_CONFIG"))
    r = flyte.with_runcontext(queue=os.getenv("GPU_QUEUE") or None).run(smoke)
    print(r.name, r.url)
