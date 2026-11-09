# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
# ]
# ///
"""Fine-tune a small model on math and publish every checkpoint as an artifact version.

    train_and_watch (CPU driver, live leaderboard report)
      └─ finetune (L40S): SFT SmolLM2-360M on MetaMathQA
           step 0, then 8 checkpoints ──> Artifact.create(math-sft-checkpoint, step=...)
                                              │  (the evaluator's trigger runs each one)
      <── polls math-sft-checkpoint-evals, redraws curves, picks and promotes the best

The trainer never calls the evaluator. It publishes checkpoints and keeps training; the
evaluator's artifact trigger (see evals.py) launches one eval run per checkpoint. The driver
only reads the published scores to draw learning curves and promote a release candidate.

    python train.py                       # deploy the evaluator, then train (about 20 minutes)
    python train.py --smoke               # 2 checkpoints, 40 problems per benchmark
    python train.py --leaderboard <run>   # redraw the leaderboard after a backfill
"""

import asyncio
import json
import sys
import time

import flyte
import flyte.artifacts as artifacts
import flyte.report
from config import (
    BASE_MODEL,
    CHECKPOINT_ARTIFACT,
    RELEASE_ARTIFACT,
    RESULTS_ARTIFACT,
    driver_image,
    gpu_image,
)
from flyte.io import Dir, File
from reports import leaderboard_html, pick_best, release_card, training_html

# {{docs-fragment envs}}
train_env = flyte.TaskEnvironment(
    name="ckpt-evals-train",
    image=gpu_image,
    resources=flyte.Resources(cpu=6, memory="48Gi", gpu="L40s:1", disk="80Gi"),
    env_vars={"TOKENIZERS_PARALLELISM": "false"},
)

driver_env = flyte.TaskEnvironment(
    name="ckpt-evals-driver",
    image=driver_image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    depends_on=[train_env],
)
# {{/docs-fragment envs}}


def checkpoint_steps(total: int, n: int) -> list[int]:
    """Step 0, then ``n`` checkpoints spaced closer early on, where the curves move fastest."""
    fracs = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 3 / 8, 1 / 2, 3 / 4, 1][-n:] if n <= 8 else [k / n for k in range(1, n + 1)]
    return [0] + sorted({max(1, round(f * total)) for f in fracs})


def load_sft_data(tok, n: int, max_len: int, seed: int) -> list[tuple[list[int], int]]:
    """MetaMathQA's GSM8K-derived problems as (token ids, prompt length) in the eval prompt format."""
    from benchmarks import SFT_PROMPT
    from datasets import load_dataset

    ds = load_dataset("meta-math/MetaMathQA", split="train")
    ds = ds.filter(lambda t: t.startswith("GSM"), input_columns="type").shuffle(seed=seed).select(range(n))
    out = []
    for q, a in zip(ds["query"], ds["response"]):
        prompt = tok.encode(SFT_PROMPT.format(question=q), add_special_tokens=False)
        answer = tok.encode(" " + a.strip(), add_special_tokens=False) + [tok.eos_token_id]
        out.append(((prompt + answer)[:max_len], len(prompt)))
    return out


# {{docs-fragment publish}}
async def publish_checkpoint(path: str, step: int, train_run: str, profile: str, loss: float | None) -> str:
    """Upload a saved checkpoint and register it as a new version of the checkpoint artifact.

    The evaluator's trigger fires on this call; training does not wait for it.
    """
    from flyte.remote import Artifact

    ckpt = await Dir.from_local(path)
    art = await Artifact.create.aio(
        ckpt,
        name=CHECKPOINT_ARTIFACT,
        version=f"{train_run}-step{step:05d}",
        kind="model",
        description=f"{BASE_MODEL} math SFT, step {step} of {train_run}",
        partitions={"train_run": train_run, "step": str(step), "profile": profile},
        attrs={"base_model": BASE_MODEL, "train_loss": f"{loss:.4f}" if loss is not None else "n/a"},
    )
    return art.version
# {{/docs-fragment publish}}


# {{docs-fragment finetune}}
@train_env.task(report=True)
async def finetune(
    train_run: str,
    profile: str = "full",
    train_examples: int = 50_000,
    n_checkpoints: int = 8,
    learning_rate: float = 3e-5,
    micro_batch: int = 8,
    grad_accum: int = 4,
    max_len: int = 512,
    seed: int = 0,
) -> str:
    """Full fine-tune on one GPU; save and publish checkpoints without pausing for uploads."""
    import math
    import os
    import random

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32).cuda()
    data = load_sft_data(tok, train_examples, max_len, seed)
    total = len(data) // (micro_batch * grad_accum)
    save_at = checkpoint_steps(total, n_checkpoints)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    warmup = max(1, total // 30)

    def lr_at(s: int) -> float:  # linear warmup, cosine decay to 10%
        if s < warmup:
            return learning_rate * (s + 1) / warmup
        return learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))

    loop = asyncio.get_running_loop()
    uploads: list = []
    history: list[dict] = []
    published: list[dict] = []
    started = time.time()

    def save(step: int, loss: float | None) -> None:
        # Saving takes seconds and runs on the training thread. Uploading and publishing run on
        # the event loop, so the next optimizer step starts right away.
        path = f"/tmp/ckpt/step-{step:05d}"
        os.makedirs(path, exist_ok=True)
        sd = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in model.state_dict().items()
              if not (model.config.tie_word_embeddings and k == "lm_head.weight")}
        save_file(sd, f"{path}/model.safetensors", metadata={"format": "pt"})
        model.config.save_pretrained(path)
        model.generation_config.save_pretrained(path)
        tok.save_pretrained(path)
        published.append({"step": step, "loss": loss, "saved_s": round(time.time() - started, 1)})
        uploads.append(asyncio.run_coroutine_threadsafe(
            publish_checkpoint(path, step, train_run, profile, loss), loop))

    def train() -> None:
        model.train()
        order = list(range(len(data)))
        random.Random(seed).shuffle(order)
        save(0, None)
        for step in range(1, total + 1):
            for g in opt.param_groups:
                g["lr"] = lr_at(step - 1)
            step_loss = 0.0
            for m in range(grad_accum):
                lo = ((step - 1) * grad_accum + m) * micro_batch
                batch = [data[i] for i in order[lo:lo + micro_batch]]
                width = max(len(ids) for ids, _ in batch)
                x = torch.full((len(batch), width), tok.pad_token_id or tok.eos_token_id)
                y = torch.full((len(batch), width), -100)
                mask = torch.zeros((len(batch), width), dtype=torch.long)
                for r, (ids, plen) in enumerate(batch):
                    x[r, :len(ids)] = torch.tensor(ids)
                    mask[r, :len(ids)] = 1
                    y[r, plen:len(ids)] = torch.tensor(ids[plen:])  # loss on the answer only
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=x.cuda(), attention_mask=mask.cuda(), labels=y.cuda()).loss / grad_accum
                loss.backward()
                step_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % 10 == 0 or step == total:
                history.append({"step": step, "loss": round(step_loss, 4), "lr": lr_at(step - 1),
                                "elapsed_s": round(time.time() - started, 1)})
            if step in save_at:
                save(step, step_loss)
            if step % 25 == 0 or step in save_at:
                flyte.report.replace(training_html(history, published, total, train_examples, BASE_MODEL))
                flyte.report.flush()

    await asyncio.to_thread(train)
    versions = await asyncio.gather(*[asyncio.wrap_future(u) for u in uploads])
    flyte.report.replace(training_html(history, published, total, train_examples, BASE_MODEL))
    flyte.report.flush()
    return json.dumps({"steps": save_at, "versions": versions, "history": history, "published": published,
                       "total_steps": total, "train_s": round(time.time() - started, 1)})
# {{/docs-fragment finetune}}


# {{docs-fragment collect}}
async def collect_results(train_run: str) -> dict[int, dict]:
    """The newest published scores for each checkpoint of one training run, keyed by step."""
    from flyte.remote import Artifact

    rows: dict[int, dict] = {}
    async for art in Artifact.listall.aio(name=RESULTS_ARTIFACT, train_run=train_run):
        step = int(art.partitions["step"])
        if step in rows:  # listall is newest first; a backfill publishes a newer version
            continue
        f = await art.to_python.aio(File)
        async with f.open("rb") as fh:
            rows[step] = json.loads(bytes(await fh.read()))
        rows[step]["published_at"] = art.pb2.created_at.ToDatetime().timestamp()
    return rows


async def checkpoint_times(train_run: str) -> dict[int, float]:
    from flyte.remote import Artifact

    return {int(a.partitions["step"]): a.pb2.created_at.ToDatetime().timestamp()
            async for a in Artifact.listall.aio(name=CHECKPOINT_ARTIFACT, train_run=train_run)}
# {{/docs-fragment collect}}


# {{docs-fragment promote}}
async def promote(train_run: str, rows: dict[int, dict], best: int) -> str:
    """Publish the winning checkpoint again, as the release-candidate artifact with a model card."""
    from flyte.remote import Artifact

    ckpt = await Artifact.get.aio(CHECKPOINT_ARTIFACT, train_run=train_run, step=str(best))
    card = await artifacts.Card.create_from.aio(content=release_card(train_run, rows, best), format="html",
                                                card_type="model")
    scores = rows[best]["scores"]
    art = await Artifact.create.aio(
        await ckpt.to_python.aio(Dir),
        name=RELEASE_ARTIFACT,
        version=f"{train_run}-step{best:05d}",
        kind="model",
        description=f"step {best} of {train_run}: " + ", ".join(f"{k} {v['accuracy']:.1f}" for k, v in scores.items()),
        attrs={"train_run": train_run, "step": str(best), **{k: f"{v['accuracy']:.2f}" for k, v in scores.items()}},
        card=card,
        parents=[ckpt.artifact_version_id],
    )
    return art.version
# {{/docs-fragment promote}}


# {{docs-fragment driver}}
@driver_env.task(report=True)
async def train_and_watch(
    profile: str = "full",
    train_examples: int = 50_000,
    n_checkpoints: int = 8,
    learning_rate: float = 3e-5,
    wait_minutes: int = 40,
) -> str:
    """Train in the background, redraw the leaderboard as scores arrive, then promote the winner."""
    train_run = flyte.ctx().action.run_name
    started = time.time()
    trainer = asyncio.create_task(finetune(train_run=train_run, profile=profile, train_examples=train_examples,
                                           n_checkpoints=n_checkpoints, learning_rate=learning_rate))
    info, rows, deadline = None, {}, None
    while True:
        rows = await collect_results(train_run)
        times = await checkpoint_times(train_run)
        if trainer.done() and info is None:
            info = json.loads(trainer.result())  # re-raises if training failed
            deadline = time.time() + wait_minutes * 60
        expected = len(info["steps"]) if info else n_checkpoints + 1
        await flyte.report.replace.aio(
            leaderboard_html(train_run, rows, times, started, info, expected, final=False), do_flush=True)
        if info and (len(rows) >= expected or time.time() > deadline):
            break
        await asyncio.sleep(30)

    best = pick_best(rows)
    release = await promote(train_run, rows, best["step"]) if best else None
    await flyte.report.replace.aio(
        leaderboard_html(train_run, rows, times, started, info, expected, final=True, release=release), do_flush=True)
    return json.dumps({"best_step": best["step"] if best else None, "release_version": release,
                       "evaluated": sorted(rows), "expected": info["steps"]})
# {{/docs-fragment driver}}


@driver_env.task(report=True)
async def refresh_leaderboard(train_run: str) -> str:
    """Redraw the leaderboard for a finished training run, for example after a backfill."""
    rows = await collect_results(train_run)
    times = await checkpoint_times(train_run)
    await flyte.report.replace.aio(
        leaderboard_html(train_run, rows, times, min(times.values()), None, len(times), final=True), do_flush=True)
    best = pick_best(rows)
    return json.dumps({"best_step": best["step"] if best else None, "evaluated": sorted(rows)})


# {{docs-fragment run}}
if __name__ == "__main__":
    import evals

    flyte.init_from_config()
    if "--leaderboard" in sys.argv:
        run = flyte.run(refresh_leaderboard, train_run=sys.argv[sys.argv.index("--leaderboard") + 1])
    else:
        # The evaluator must be deployed, with its trigger active, before the first checkpoint lands.
        flyte.deploy(evals.eval_env)
        if "--smoke" in sys.argv:
            run = flyte.run(train_and_watch, profile="smoke", train_examples=1_600, n_checkpoints=2, wait_minutes=20)
        else:
            run = flyte.run(train_and_watch)
    print(run.url)
# {{/docs-fragment run}}
