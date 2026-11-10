# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
# ]
# main = "find_features"
# params = "layers=[3] widths=[4096] n_tokens=500000 steps=600 features_per_sae=8"
# ///
"""Find interpretable features in a language model with sparse autoencoders, and label them with Claude.

    harvest_activations (GPU, cached)    Pythia-70m over FineWeb: residual stream at several layers -> Dir per layer
      -> N x train_sae (GPU each)        one (layer, width) per task: TopK SAEs at k = 16, 32, 64, then
                                         variance explained, L0, dead features, and loss with the SAE spliced in
      -> M x label_features (CPU)        Claude explains sampled features and is scored on held-out contexts
    -> report (feature browser)  ->  publish the most interpretable SAE as an artifact

The harvest is cached on its inputs. Re-running with another width, k, or step budget reuses
the activations, and every (layer, width) that was already trained, and only trains what is new.

    uv run find_features.py                      # 3 layers x 2 widths, 8M tokens, about 20 minutes
    uv run find_features.py --smoke              # 1 layer x 1 width, 0.5M tokens
    uv run find_features.py --widths 4096 16384 32768   # reuses the harvest and the first two widths
"""

import argparse
import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import autointerp
import flyte
import flyte.artifacts as artifacts
import flyte.report
import reports
from flyte.io import Dir, File

# The model to study, and the Claude model that labels its features. Both are task inputs
# (harvest_activations' `model`, label_features' `claude_model`), so changing either one is a
# cache miss rather than a stale cached result.
MODEL = "EleutherAI/pythia-70m-deduped"
CLAUDE = "claude-opus-5-5"
# The name of the secret that holds your Anthropic API key. Set ANTHROPIC_SECRET when you
# launch to use a secret with another name.
ANTHROPIC_SECRET = os.getenv("ANTHROPIC_SECRET", "anthropic-api-key")
# Bump when sae.py changes (the SAE, its training, its metrics, or feature sampling). train_sae
# salts its cache with this; cache="auto" alone hashes only the task's own body.
SAE_VERSION = "topk-auxk/2"
# Set SAE_DEVICE=cpu to run a tiny version on a laptop with flyte's local mode.
DEVICE = os.getenv("SAE_DEVICE", "cuda")

# {{docs-fragment envs}}
image = flyte.Image.from_debian_base(name="sae-features").with_pip_packages(
    "torch==2.9.1", "transformers==4.57.3", "datasets==4.4.2", "numpy<3", "anthropic==2.6.0",
)

# One GPU per task: the harvest, and every SAE trainer. sae.py is imported inside the tasks.
gpu_env = flyte.TaskEnvironment(
    name="sae-gpu",
    image=image,
    resources=flyte.Resources(cpu=6, memory="40Gi", gpu="L4:1", disk="80Gi"),
    cache="auto",
    include=["sae.py"],
)

# Autointerp is API calls, so CPU only. Only this environment holds the API key.
interp_env = flyte.TaskEnvironment(
    name="sae-autointerp",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=[flyte.Secret(key=ANTHROPIC_SECRET, as_env_var="ANTHROPIC_API_KEY")],
    cache="auto",
)

driver_env = flyte.TaskEnvironment(
    name="sae-driver",
    image=image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    # The driver declares label_features when it calls it, so it needs the same secret name.
    env_vars={"ANTHROPIC_SECRET": ANTHROPIC_SECRET},
    depends_on=[gpu_env, interp_env],
)
# {{/docs-fragment envs}}


@dataclass
class Harvest:
    model: str
    dataset: str
    layers: list[int]
    ctx_len: int
    d_model: int
    n_seqs: int
    acts: list[Dir]  # one per layer: shard_*.npy, float16 [sequences, ctx_len, d_model]
    tokens: File  # int32 [sequences, ctx_len]


@dataclass
class SAEStats:
    layer: int
    width: int
    k: int
    fve: float
    l0: float
    dead_frac: float
    ce_clean: float
    ce_sae: float
    ce_zero: float
    loss_recovered: float
    density_hist: dict = field(default_factory=dict)


@dataclass
class Trained:
    layer: int
    width: int
    ks: list[int]
    stats: list[SAEStats]
    weights: list[File]  # one per k
    curve: list[dict]
    train_s: float
    examples: File | None = None  # sampled features with their contexts, for the k that gets labeled
    label_k: int = 0


# {{docs-fragment harvest}}
@gpu_env.task
async def harvest_activations(
    layers: list[int],
    n_tokens: int = 8_000_000,
    model: str = MODEL,
    ctx_len: int = 128,
    dataset: str = "HuggingFaceFW/fineweb:sample-10BT",
    shard_seqs: int = 4096,
) -> Harvest:
    """Run Pythia over FineWeb and save the residual stream after each layer in ``layers``.

    Each context is the first ``ctx_len`` tokens of one document. Activations are float16 shards,
    one directory per layer, so a trainer downloads only the layer it needs.
    """
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    lm = AutoModelForCausalLM.from_pretrained(model).to(DEVICE).eval()
    name, _, config = dataset.partition(":")
    stream = load_dataset(name, name=config or None, split="train", streaming=True)

    n_seqs = n_tokens // ctx_len
    rows: list[list[int]] = []
    texts: list[str] = []
    for doc in stream:
        texts.append(doc["text"])
        if len(texts) == 1000:
            enc = tok(texts, truncation=True, max_length=ctx_len)["input_ids"]
            rows += [ids for ids in enc if len(ids) == ctx_len]
            texts = []
            if len(rows) >= n_seqs:
                break
    tokens = np.array(rows[:n_seqs], dtype=np.int32)
    print(f"{len(tokens):,} contexts of {ctx_len} tokens from {dataset}")

    captured: dict[int, torch.Tensor] = {}
    for layer in layers:
        lm.gpt_neox.layers[layer].register_forward_hook(
            lambda _m, _i, out, layer=layer: captured.__setitem__(layer, out[0] if isinstance(out, tuple) else out))

    root = Path(tempfile.mkdtemp())
    for layer in layers:
        (root / f"layer_{layer}").mkdir()
    d_model = lm.config.hidden_size
    with torch.no_grad():
        for shard, s in enumerate(range(0, len(tokens), shard_seqs)):
            chunk = tokens[s:s + shard_seqs]
            out = {layer: np.empty((len(chunk), ctx_len, d_model), dtype=np.float16) for layer in layers}
            for b in range(0, len(chunk), 256):
                lm(input_ids=torch.as_tensor(chunk[b:b + 256], dtype=torch.long, device=DEVICE))
                for layer in layers:
                    out[layer][b:b + 256] = captured[layer].clamp(-6e4, 6e4).half().cpu().numpy()
            for layer in layers:
                np.save(root / f"layer_{layer}" / f"shard_{shard:03d}.npy", out[layer])
    np.save(root / "tokens.npy", tokens)
    print(f"saved {len(layers)} layers x {len(tokens) * ctx_len:,} tokens")

    acts = await asyncio.gather(*[Dir.from_local(str(root / f"layer_{layer}")) for layer in layers])
    return Harvest(model=model, dataset=dataset, layers=layers, ctx_len=ctx_len, d_model=d_model,
                   n_seqs=len(tokens), acts=list(acts), tokens=await File.from_local(str(root / "tokens.npy")))
# {{/docs-fragment harvest}}


async def load_layer(harvest: Harvest, layer: int):
    """Download one layer's shards and put them on the GPU, with the token ids."""
    import numpy as np
    import torch

    local = Path(await harvest.acts[harvest.layers.index(layer)].download())
    shards = sorted(local.rglob("shard_*.npy"))
    acts = torch.cat([torch.from_numpy(np.load(p)).to(DEVICE) for p in shards])
    tokens = np.load(await harvest.tokens.download())
    return acts, tokens


# {{docs-fragment train}}
@gpu_env.task(report=True, cache=flyte.Cache("auto", salt=SAE_VERSION))
async def train_sae(
    harvest: Harvest,
    layer: int,
    width: int,
    ks: list[int],
    steps: int = 8000,
    batch_size: int = 4096,
    label_k: int = 32,
    n_features: int = 40,
    n_top: int = 12,
    n_sampled: int = 8,
    held_out: float = 0.05,
    seed: int = 0,
) -> Trained:
    """Train one TopK SAE per k on one layer, evaluate each on held-out text, and sample features to label."""
    import sae as S
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = True
    acts, tokens = await load_layer(harvest, layer)
    n_train = int(len(acts) * (1 - held_out))
    print(f"layer {layer}: {n_train:,} training and {len(acts) - n_train:,} held-out contexts, width {width}, k {ks}")

    started = time.time()
    curve_so_far: list[dict] = []

    def on_log(point: dict) -> None:  # stream the training curves into this task's report
        curve_so_far.append(point)
        flyte.report.replace(reports.training(layer, width, ks, steps, curve_so_far, time.time() - started))
        flyte.report.flush()

    # Training runs in a worker thread, so the report can stream from the callback.
    saes, scale, curve = await asyncio.to_thread(S.train, acts[:n_train], width, ks, steps, batch_size,
                                                 seed=seed, on_log=on_log)
    train_s = time.time() - started
    print(f"trained {len(ks)} SAEs for {steps} steps in {train_s / 60:.1f} min")

    # Evaluate on held-out contexts: reconstruction, sparsity, and what the model loses with the SAE in place.
    model = AutoModelForCausalLM.from_pretrained(harvest.model).to(DEVICE).eval()
    eval_tokens = torch.as_tensor(tokens[n_train:][:512], dtype=torch.long, device=DEVICE)
    ce = S.spliced_losses(model, eval_tokens, layer, saes, scale)
    stats = []
    for sae in saes:
        m = S.reconstruction_metrics(sae, acts[n_train:], scale)
        st = SAEStats(layer=layer, width=width, k=sae.k, fve=m["fve"], l0=m["l0"], dead_frac=m["dead_frac"],
                      ce_clean=ce["clean"], ce_sae=ce[f"k{sae.k}"], ce_zero=ce["zero"],
                      loss_recovered=S.loss_recovered(ce["clean"], ce[f"k{sae.k}"], ce["zero"]),
                      density_hist=S.density_histogram(m["density"]))
        print(f"  k={sae.k}: FVE {st.fve:.3f}, L0 {st.l0:.1f}, dead {st.dead_frac:.1%}, "
              f"loss {st.ce_clean:.3f} -> {st.ce_sae:.3f} (zero {st.ce_zero:.3f}), recovered {st.loss_recovered:.1%}")
        stats.append(st)

    out = Path(tempfile.mkdtemp())
    weights = []
    for sae in saes:
        path = out / f"sae_L{layer}_w{width}_k{sae.k}.pt"
        torch.save({"state_dict": sae.state_dict(), "scale": scale, "layer": layer, "width": width, "k": sae.k,
                    "d_model": sae.d_model, "model": harvest.model, "hook": f"gpt_neox.layers.{layer} output"}, path)
        weights.append(await File.from_local(str(path)))

    examples = None
    if n_features and label_k in ks:
        tok = AutoTokenizer.from_pretrained(harvest.model)
        feats = S.feature_examples(saes[ks.index(label_k)], acts, tokens, scale, lambda t: tok.decode([t]),
                                   n_features=n_features, n_top=n_top, n_sampled=n_sampled, seed=seed + layer)
        path = out / "examples.json"
        path.write_text(json.dumps(feats))
        examples = await File.from_local(str(path))
        print(f"  sampled {len(feats)} features for autointerp")
    return Trained(layer=layer, width=width, ks=ks, stats=stats, weights=weights, curve=curve,
                   train_s=round(train_s, 1), examples=examples, label_k=label_k if examples else 0)
# {{/docs-fragment train}}


# {{docs-fragment claude}}
class Unanswered(Exception):
    """Claude gave no complete JSON answer: a refusal, or output cut off at max_tokens."""


async def ask_claude(model: str, system: str, prompt: str, schema: dict, max_tokens: int = 4096) -> str:
    import anthropic

    client = anthropic.AsyncAnthropic(max_retries=8)
    message = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": prompt}],
    )
    # Any thinking counts toward max_tokens, so check that the answer finished before reading it.
    if message.stop_reason != "end_turn":
        raise Unanswered(f"stop_reason={message.stop_reason}")
    text = next((b.text for b in message.content if b.type == "text"), None)
    if text is None:
        raise Unanswered("no text block")
    return text


@flyte.trace
async def explain_feature(model: str, layer: int, feature: int, prompt: str) -> str:
    """Claude's explanation of one feature. Traced, so a retried task replays it instead of paying again."""
    return await ask_claude(model, autointerp.EXPLAIN_SYSTEM, prompt, autointerp.EXPLAIN_SCHEMA)


@flyte.trace
async def detect_feature(model: str, layer: int, feature: int, prompt: str) -> str:
    """Claude's guess of which held-out excerpts the explanation predicts. Traced, like the explanation."""
    return await ask_claude(model, autointerp.DETECT_SYSTEM, prompt, autointerp.DETECT_SCHEMA)
# {{/docs-fragment claude}}


# {{docs-fragment label}}
@interp_env.task(retries=1, cache=flyte.Cache("auto", salt=autointerp.VERSION))
async def label_features(layer: int, width: int, k: int, examples: File, claude_model: str = CLAUDE,
                         concurrency: int = 8) -> str:
    """Explain each sampled feature, then score every explanation on the feature's held-out contexts.

    Each feature is also scored with another feature's explanation. That control shows what the
    test gives an explanation that is fluent but wrong. A feature whose calls fail is recorded
    as unlabeled, so one bad answer never fails the whole task.
    """
    with open(await examples.download()) as f:
        features = json.load(f)
    gate = asyncio.Semaphore(concurrency)
    unlabeled: list[dict] = []

    async def explain(feat: dict) -> dict | None:
        try:
            async with gate:
                answer = json.loads(await explain_feature(claude_model, layer, feat["feature"], autointerp.explain_prompt(feat)))
            return {**feat, **answer}
        except Exception as exc:  # Unanswered, malformed JSON, or an API error that outlived the retries
            unlabeled.append({"feature": feat["feature"], "error": f"explain: {exc}"})
            return None

    labeled = [f for f in await asyncio.gather(*[explain(f) for f in features]) if f is not None]
    shuffled = [labeled[(i + 1) % len(labeled)]["explanation"] for i in range(len(labeled))]

    async def score(feat: dict, explanation: str) -> float | None:
        test = autointerp.detect_set(feat, seed=feat["feature"])
        try:
            async with gate:
                answer = await detect_feature(claude_model, layer, feat["feature"], autointerp.detect_prompt(explanation, test))
            return autointerp.detection_score(answer, test)
        except Exception as exc:
            unlabeled.append({"feature": feat["feature"], "error": f"detect: {exc}"})
            return None

    real = await asyncio.gather(*[score(f, f["explanation"]) for f in labeled])
    control = await asyncio.gather(*[score(f, e) for f, e in zip(labeled, shuffled)])
    scored = []
    for feat, s, c in zip(labeled, real, control):
        if s is None or c is None:
            continue
        feat.update(score=s, control=c)
        scored.append(feat)
        print(f"  #{feat['feature']}: {s:.2f} (control {c:.2f})  {feat['explanation']}")
    if scored:
        print(f"layer {layer}: mean detection {sum(f['score'] for f in scored) / len(scored):.2f}, "
              f"mismatched control {sum(f['control'] for f in scored) / len(scored):.2f}")
    unlabeled = list({u["feature"]: u for u in unlabeled}.values())  # one entry per feature
    for u in unlabeled:
        print(f"  #{u['feature']} unlabeled: {u['error']}")
    return json.dumps({"layer": layer, "width": width, "k": k, "features": scored, "unlabeled": unlabeled})
# {{/docs-fragment label}}


# {{docs-fragment publish}}
async def publish(best: Trained, stats: SAEStats, labeled: dict, run_name: str) -> File:
    card = await artifacts.Card.create_from.aio(content=reports.card(best, stats, labeled, run_name), format="html",
                                                card_type="model")
    scores = [f["score"] for f in labeled["features"]]
    metadata = artifacts.Metadata.create_model_metadata(
        name="pythia-70m-topk-sae",
        description=(f"TopK SAE on Pythia-70m-deduped layer {stats.layer}: {stats.width} features, k={stats.k}, "
                     f"{stats.fve:.0%} variance explained, {stats.loss_recovered:.0%} loss recovered"),
        framework="PyTorch",
        model_type="TopK sparse autoencoder",
        architecture=f"d_model 512 -> {stats.width} -> 512, TopK k={stats.k}",
        task="interpretability",
        modality=("text",),
        serial_format="pt",
        card=card,
        attrs={"layer": str(stats.layer), "width": str(stats.width), "k": str(stats.k), "fve": f"{stats.fve:.4f}",
               "loss_recovered": f"{stats.loss_recovered:.4f}", "dead_frac": f"{stats.dead_frac:.4f}",
               "detection": f"{sum(scores) / len(scores):.3f}"},
    )
    return artifacts.new(best.weights[best.ks.index(stats.k)], metadata)
# {{/docs-fragment publish}}


# {{docs-fragment driver}}
@driver_env.task(report=True, produces_artifacts=True)
async def find_features(
    layers: list[int] = [1, 3, 5],
    widths: list[int] = [4096, 16384],
    ks: list[int] = [16, 32, 64],
    n_tokens: int = 8_000_000,
    steps: int = 8000,
    label_k: int = 32,
    label_width: int = 16384,
    features_per_sae: int = 40,
    claude_model: str = CLAUDE,
) -> File:
    """Harvest activations, train an SAE grid in parallel, label features with Claude, and publish the best SAE."""
    if label_width not in widths:
        label_width = max(widths)
    started = time.time()
    await flyte.report.replace.aio(reports.page_start(layers, widths, ks, n_tokens), do_flush=True)

    harvest = await harvest_activations(layers=layers, n_tokens=n_tokens)
    trained: list[Trained] = await asyncio.gather(*[
        train_sae(harvest, layer, width, ks, steps=steps, label_k=label_k,
                  n_features=features_per_sae if width == label_width else 0)
        for layer in layers for width in widths
    ])
    await flyte.report.replace.aio(reports.overview(trained, [], harvest, time.time() - started), do_flush=True)

    labeled = [json.loads(s) for s in await asyncio.gather(*[
        label_features(t.layer, t.width, t.label_k, t.examples, claude_model) for t in trained if t.examples is not None
    ])]
    await flyte.report.replace.aio(reports.overview(trained, labeled, harvest, time.time() - started))
    flyte.report.get_tab("Feature browser").replace(reports.browser(labeled))
    await flyte.report.flush.aio()

    # The SAE whose features Claude explained best, by mean detection score.
    mean = lambda lab: sum(f["score"] for f in lab["features"]) / len(lab["features"])
    best_labels = max((lab for lab in labeled if lab["features"]), key=mean)
    best = next(t for t in trained if (t.layer, t.width) == (best_labels["layer"], best_labels["width"]))
    stats = next(s for s in best.stats if s.k == best_labels["k"])
    return await publish(best, stats, best_labels, flyte.ctx().action.run_name)
# {{/docs-fragment driver}}


# {{docs-fragment run}}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="1 layer, 1 width, 0.5M tokens, 8 features")
    parser.add_argument("--widths", type=int, nargs="+", default=[4096, 16384])
    args = parser.parse_args()
    flyte.init_from_config()
    if args.smoke:
        run = flyte.run(find_features, layers=[3], widths=[4096], n_tokens=500_000, steps=600, features_per_sae=8)
    else:
        run = flyte.run(find_features, widths=args.widths)
    print(run.url)
# {{/docs-fragment run}}
