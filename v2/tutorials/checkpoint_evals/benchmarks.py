"""Benchmark slices: loading, prompts, answer extraction and scoring.

Every function imports its heavy dependencies inside, so this module imports cleanly in the
driver image too. The prompt format matches the one the trainer uses for SFT, so the base
model (step 0) and every checkpoint see exactly the same prompts.
"""

import math
import random
import re
import time

SFT_PROMPT = "Question: {question}\nAnswer:"
# MMLU as a regression guard measures general knowledge, so it leaves out the math subjects the
# SFT data trains on.
MATH_SUBJECTS = ("math", "algebra", "statistics")
# ARC prompt versions. v1 is the usual cloze prompt, which is also exactly the SFT prompt, so a
# fine-tuned model expects a worked solution after it. v2 asks for the final answer instead.
ARC_PROMPTS = {"1": "Question: {question}\nAnswer:", "2": "Question: {question}\nThe answer is:"}


def _sample(n: int, limit: int, seed: int = 0) -> list[int]:
    idx = list(range(n))
    if limit and limit < n:
        idx = sorted(random.Random(seed).sample(idx, limit))
    return idx


# {{docs-fragment load}}
def load_items(benchmark: str, limit: int, version: str = "1") -> list[dict]:
    """A fixed, seeded slice of one benchmark's test split as plain dicts."""
    from datasets import load_dataset

    if benchmark == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        rows = [ds[i] for i in _sample(len(ds), limit)]
        return [{"question": r["question"], "gold": r["answer"].split("####")[-1].strip().replace(",", "")}
                for r in rows]
    if benchmark in ("arc_easy", "arc_challenge"):
        name = "ARC-Easy" if benchmark == "arc_easy" else "ARC-Challenge"
        ds = load_dataset("allenai/ai2_arc", name, split="test")
        rows = [ds[i] for i in _sample(len(ds), limit)]
        prompt = ARC_PROMPTS[version]
        return [{"context": prompt.format(question=r["question"].strip()), "choices": list(r["choices"]["text"]),
                 "gold": list(r["choices"]["label"]).index(r["answerKey"])} for r in rows]
    if benchmark == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        keep = [i for i, s in enumerate(ds["subject"]) if not any(m in s for m in MATH_SUBJECTS)]
        rows = [ds[keep[j]] for j in _sample(len(keep), limit)]
        return [{"context": f"The following is a question about {r['subject'].replace('_', ' ')}.\n\n"
                            f"Question: {r['question'].strip()}\nAnswer:",
                 "choices": list(r["choices"]), "gold": int(r["answer"])} for r in rows]
    raise ValueError(f"unknown benchmark {benchmark!r}")
# {{/docs-fragment load}}


# {{docs-fragment extract}}
_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def extract_answer(text: str) -> str | None:
    """The final number: after "The answer is" if the model says it, else the last number."""
    m = re.search(rf"answer is:?\s*\$?\s*({_NUM})", text, flags=re.IGNORECASE)
    if m is None:
        nums = re.findall(_NUM, text)
        if not nums:
            return None
        found = nums[-1]
    else:
        found = m.group(1)
    found = found.replace(",", "").rstrip(".")
    try:
        v = float(found)
    except ValueError:
        return None
    return str(int(v)) if v.is_integer() else str(v)
# {{/docs-fragment extract}}


def _summary(benchmark: str, version: str, correct: list[bool], seconds: float, examples: list[dict]) -> dict:
    n = len(correct)
    acc = sum(correct) / max(n, 1)
    return {"benchmark": benchmark, "version": version, "n": n, "correct": int(sum(correct)),
            "accuracy": round(100 * acc, 2), "stderr": round(100 * math.sqrt(acc * (1 - acc) / max(n, 1)), 2),
            "seconds": round(seconds, 1), "examples": examples}


# {{docs-fragment generate}}
def score_generate(model_dir: str, items: list[dict], version: str, n_examples: int = 24) -> dict:
    """GSM8K: greedy generation with vLLM, exact match on the final number."""
    from vllm import LLM, SamplingParams

    started = time.time()
    llm = LLM(model=model_dir, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85,
              enforce_eager=True, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=400, stop=["\nQuestion:", "\n\n\n"])
    outs = llm.generate([SFT_PROMPT.format(question=it["question"]) for it in items], params)
    texts = [o.outputs[0].text.strip() for o in outs]
    preds = [extract_answer(t) for t in texts]
    correct = [p is not None and float(p) == float(it["gold"]) for p, it in zip(preds, items)]
    examples = [{"question": it["question"], "gold": it["gold"], "pred": p, "ok": ok, "output": t[:900]}
                for it, p, ok, t in list(zip(items, preds, correct, texts))[:n_examples]]
    return _summary("gsm8k", version, correct, time.time() - started, examples)
# {{/docs-fragment generate}}


# {{docs-fragment choice}}
def score_choice(model_dir: str, items: list[dict], benchmark: str, version: str, batch_tokens: int = 24_576) -> dict:
    """Multiple choice in the cloze form: pick the answer text with the highest log-likelihood.

    Small models score near chance when they have to answer with a letter, so this scores each
    option's own text after the question, normalized by its length in characters (as OLMES does
    for small models).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    pairs = []  # (item index, token ids, number of answer tokens, answer length in characters)
    for i, it in enumerate(items):
        ctx = tok.encode(it["context"], add_special_tokens=False)
        for choice in it["choices"]:
            ans = tok.encode(" " + choice.strip(), add_special_tokens=False)
            pairs.append((i, (ctx + ans)[-1024:], len(ans), max(1, len(choice.strip()))))
    score = [0.0] * len(pairs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    order = sorted(range(len(pairs)), key=lambda k: len(pairs[k][1]))  # batch similar lengths
    with torch.inference_mode():
        b = 0
        while b < len(order):
            # Sorted by length, so the last row is the widest: fill the batch up to a token budget.
            n = max(1, batch_tokens // len(pairs[order[min(b + 255, len(order) - 1)]][1]))
            ks = order[b:b + min(n, 256)]
            b += len(ks)
            ids = [pairs[k][1] for k in ks]
            width = max(len(t) for t in ids)
            # Right padding keeps position ids exact for every row.
            x = torch.tensor([t + [pad] * (width - len(t)) for t in ids]).cuda()
            mask = torch.tensor([[1] * len(t) + [0] * (width - len(t)) for t in ids]).cuda()
            logp = torch.log_softmax(model(input_ids=x, attention_mask=mask).logits.float(), dim=-1)
            for r, k in enumerate(ks):
                _, t, n_ans, n_chars = pairs[k]
                pos = torch.arange(len(t) - n_ans, len(t), device=x.device)
                score[k] = logp[r, pos - 1, x[r, pos]].sum().item() / n_chars  # token t is predicted at t-1
    by_item: list[list[float]] = [[] for _ in items]
    for k, (i, _, _, _) in enumerate(pairs):  # pairs are in item, then option, order
        by_item[i].append(score[k])
    correct = [max(range(len(sc)), key=sc.__getitem__) == it["gold"] for sc, it in zip(by_item, items)]
    return _summary(benchmark, version, correct, time.time() - started, [])
# {{/docs-fragment choice}}
