"""Diagnostics for the training path on one node: does the trainer's log-probability match the sampler's?

A CISPO importance ratio is only meaningful if the trainer and the sampler compute the same distribution. This serves
a bf16 copy of gpt-oss-20b, captures real turns from two LAB rollouts, and compares vLLM's log-probs of the sampled
tokens with the trainer's under each attention implementation, then times a long backward pass.

    CLUSTER=amd GPU_QUEUE=<queue> python diag.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import flyte

import lab_harness as lh
import servers
from config import PROPAGATE, gpu_env_kwargs
from images import LAB_COMMIT
from train import bf16_copy

env = flyte.TaskEnvironment(name="legal-rl-diag", env_vars=PROPAGATE, **gpu_env_kwargs())
TASKS = ["immigration/compare-draft-eb", "trusts-estates-private-client/compare-trust-documents-against-client-instructions"]


def gap_probe(gpu: int, model_dir: str, turns_path: str, attn: str, long_len: int, experts: str = "eager") -> dict:
    os.environ[servers.VISIBLE] = str(gpu)
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    if attn == "sink_flex":
        import gptoss_attention

        gptoss_attention.register()
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation=attn,
                                                 experts_implementation=experts).to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=64, lora_alpha=64, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    turns = [lh.TurnTokens(**t) for t in json.load(open(turns_path))]
    gaps, n_tok = [], 0
    with torch.no_grad():
        for t in turns[:12]:
            if not t.logprobs or (attn == "eager" and len(t.prompt_ids) > 7000):
                continue
            ids = torch.tensor([t.prompt_ids + t.completion_ids], device="cuda")
            pos = torch.arange(len(t.prompt_ids) - 1, ids.shape[1] - 1, device="cuda")
            h = model.base_model.model.model(input_ids=ids).last_hidden_state[0, pos]
            lp = torch.log_softmax(model.base_model.model.lm_head(h).float(), -1).gather(-1, ids[0, pos + 1, None])[:, 0]
            old = torch.tensor(t.logprobs, device="cuda")
            gaps.append(float((lp - old).abs().mean()))
            n_tok += len(t.logprobs)
    out = {"attn": attn, "experts": experts, "turns": len(gaps), "tokens": n_tok, "mean_abs_gap": sum(gaps) / max(len(gaps), 1),
           "per_turn": [round(g, 4) for g in gaps]}
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, 100_000, (1, long_len), device="cuda")
    pos = torch.arange(long_len - 1024, long_len - 1, device="cuda")
    t0 = time.time()
    h = model.base_model.model.model(input_ids=ids).last_hidden_state[0, pos]
    torch.log_softmax(model.base_model.model.lm_head(h).float(), -1).gather(-1, ids[0, pos + 1, None]).sum().backward()
    torch.cuda.synchronize()
    out.update({f"backward_s@{long_len}": round(time.time() - t0, 1),
                f"peak_gib@{long_len}": round(torch.cuda.max_memory_allocated() / 2**30, 1)})
    return out


def sanity_probe(gpu: int, model_dir: str, turns_path: str, which: str, vllm_scores_path: str) -> dict:
    """Is the HF model sane, and does it agree with vLLM's *scoring* (prompt_logprobs) of the same tokens?"""
    os.environ[servers.VISIBLE] = str(gpu)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    kw = {"quantization_config": Mxfp4Config(dequantize=True)} if which.startswith("original") else {}
    src = "openai/gpt-oss-20b" if which.startswith("original") else model_dir
    attn = "flex_attention" if which.endswith("flex") else "eager"
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, attn_implementation=attn, **kw
                                                 ).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    msgs = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
    ids = torch.tensor([tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                return_dict=False)], device="cuda")
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=40, do_sample=False)
    out = {"which": which, "greedy": tok.decode(gen[0, ids.shape[1]:]),
           "config_quant": str(json.load(open(f"{model_dir}/config.json")).get("quantization_config"))[:200]}
    vs = json.load(open(vllm_scores_path))
    with torch.no_grad():
        x = torch.tensor([vs["ids"]], device="cuda")
        lp = torch.log_softmax(model(x).logits[0, :-1].float(), -1).gather(-1, x[0, 1:, None])[:, 0]
    v = torch.tensor(vs["logprobs"][1:], device="cuda")
    out["gap_vs_vllm_scoring"] = float((lp - v).abs().mean())
    out["hf_mean_lp"], out["vllm_mean_lp"] = float(lp.mean()), float(v.mean())
    # and against the sampler's own logprobs for the first turn's completion
    t = lh.TurnTokens(**json.load(open(turns_path))[0])
    n = len(t.prompt_ids)
    out["sampler_mean_lp"] = sum(t.logprobs) / len(t.logprobs)
    out["hf_on_sampler_tokens"] = float(lp[n - 1:n - 1 + len(t.completion_ids)].mean()) if len(vs["ids"]) > n else None
    return out


def _probe_subprocess(*args) -> dict:
    fn = "sanity_probe" if args[3] in ("original", "copy", "original-flex") else "gap_probe"
    code = ("import faulthandler, json, diag; faulthandler.enable(); "
            f"print('RESULT', json.dumps(diag.{fn}(*{args!r})))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.path.dirname(__file__))
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
    return json.loads(line[7:]) if line else {"args": list(args), "returncode": r.returncode,
                                              "stderr": r.stderr[-3000:], "stdout": r.stdout[-400:]}


@env.task
async def diag() -> str:
    model_dir = await asyncio.to_thread(bf16_copy, "openai/gpt-oss-20b")
    srv = servers.ServerSpec(served_name="policy", hf_id=model_dir, gpus=[0], tool_parser="openai",
                             reasoning_parser="openai_gptoss", max_model_len=65536, enable_lora=True)
    await servers.ensure(srv)
    lab = lh.fetch_lab(TASKS, LAB_COMMIT, Path("/tmp/lab"))
    pol = lh.Policy(name="policy", hf_id=model_dir, reasoning_effort="medium")
    rs = await asyncio.gather(*[asyncio.to_thread(lh.run_rollout, lab, t, pol, lh.Harness(max_turns=8), srv.url, 0, True)
                                for t in TASKS])
    turns = [t.model_dump() for r in rs for t in r.tokens]
    json.dump(turns, open("/tmp/turns.json", "w"))
    out = {"turns_captured": len(turns), "matched": sum(t["logprob_ids_matched"] for t in turns),
           "prompt_lens": [len(t["prompt_ids"]) for t in turns][:12]}
    # vLLM scoring the first turn's full sequence (prompt + sampled completion) with prompt_logprobs.
    import httpx

    t0_ = turns[0]
    seq = t0_["prompt_ids"] + t0_["completion_ids"]
    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.post(srv.url + "/completions", json={"model": "policy", "prompt": seq, "max_tokens": 1,
                                                          "prompt_logprobs": 0, "temperature": 0})
    pl = r.json()["choices"][0].get("prompt_logprobs") or []
    vlp = [0.0] + [float(next(iter(d.values()))["logprob"]) if d else 0.0 for d in pl[1:]]
    json.dump({"ids": seq, "logprobs": vlp}, open("/tmp/vllm_scores.json", "w"))
    n = len(t0_["prompt_ids"])
    out["vllm_scoring_vs_sampling_gap"] = sum(abs(a - b) for a, b in zip(vlp[n:], t0_["logprobs"])) / len(t0_["logprobs"])
    res = await asyncio.gather(*[asyncio.to_thread(_probe_subprocess, g, model_dir, "/tmp/turns.json", w,
                                                   "/tmp/vllm_scores.json") for g, w in ((4, "original-flex"), (5, "original"))])
    out["probes"] = res
    print(json.dumps(out, indent=1), flush=True)
    return json.dumps(out)


if __name__ == "__main__":
    flyte.init_from_config(os.getenv("FLYTE_CONFIG"))
    r = flyte.with_runcontext(queue=os.getenv("GPU_QUEUE") or None).run(diag)
    print(r.name, r.url)
