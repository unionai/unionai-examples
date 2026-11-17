"""Sample completions with vLLM in a *subprocess*, so every bit of GPU memory it took is released when it exits.

Called by ``lora_screen.py`` once per step. Token-in/token-out: it returns the prompt and completion token ids
vLLM actually used, and the trainer computes log-probs over exactly those ids (no re-tokenization).

    python vllm_generate.py --model <dir> [--lora <adapter dir>] --requests in.json --out out.json \
        --n 8 --max-tokens 512 --temperature 1.0 --seed 0 --gpu-mem 0.45
"""

import argparse
import json
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--requests", required=True)  # json: list[str] prompts
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.45)
    ap.add_argument("--stop", default="[]")  # json list of stop strings
    a = ap.parse_args()

    t0 = time.monotonic()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=a.model,
        dtype="bfloat16",  # full-weight policies are saved as fp32 master weights
        enable_lora=a.lora is not None,
        max_lora_rank=a.lora_rank,
        gpu_memory_utilization=a.gpu_mem,
        max_model_len=2048,
        enforce_eager=True,
        seed=a.seed,
    )
    t_init = time.monotonic() - t0
    with open(a.requests) as f:
        prompts = json.load(f)
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=1.0, max_tokens=a.max_tokens, seed=a.seed,
                        stop=json.loads(a.stop))
    lora = LoRARequest("policy", 1, a.lora) if a.lora else None
    t1 = time.monotonic()
    outs = llm.generate(prompts, sp, lora_request=lora)
    t_gen = time.monotonic() - t1
    result = {
        "engine_init_s": round(t_init, 2),
        "generate_s": round(t_gen, 2),
        "groups": [
            {
                "prompt_ids": list(o.prompt_token_ids),
                "samples": [{"ids": list(c.token_ids), "text": c.text} for c in o.outputs],
            }
            for o in outs
        ],
    }
    with open(a.out, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
