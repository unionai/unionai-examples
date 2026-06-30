# Reinforcement Learning for LLMs (GRPO + LoRA) on Union

This is a hands-on tutorial that demonstrates a **reinforcement-learning loop for LLMs** — the same
kind of loop the major labs use to train reasoning models. It is implemented **entirely on Flyte's
SDK**: Flyte orchestrates the tasks and applies tricks like **reusable-container warm pools** to keep
the loop fast. The benefit of Flyte really shows here — it **autoscales to the resources you have**,
**isolates concurrent runs** from each other, resumes after failures, and streams progress to a live
report. And once you have Union/Flyte installed, getting this example running is trivial: it's one
`python` command.

By the end you'll understand, and have running:

- the **RL-for-LLMs loop** (sample → generate → score → update → repeat) and the **GRPO** objective,
- how Flyte keeps a **vLLM engine warm** across iterations so you never reload the model,
- how a small **LoRA adapter** is trained with one policy-gradient step and handed back to the
  generator,
- how generation and reward scoring **pipeline** so they overlap, and
- how the whole run is **resumable** with a live progress **report**.

The full, runnable code is in [`rl_grpo_lora.py`](./rl_grpo_lora.py); this README walks through it
piece by piece. It has been validated end-to-end on a Union demo cluster (Qwen3-0.6B, L4 GPUs).

---

## Why Flyte with Union is a great fit for RL training

RL training loops are genuinely awkward to run well. They mix wildly different work — GPU text
generation, cheap CPU scoring, GPU gradient steps — in a tight loop; they run long enough that
failures are a certainty, not a risk; and they're hard to watch while they run. Flyte with Union is
built for exactly this shape of problem, which is why the loop in this tutorial is nothing more than a
`for` loop of ordinary Python functions. You write plain Flyte tasks; Union runs them — provisioning
the GPUs, keeping the warm pools alive, and serving the UI and reports.

What you get, essentially for free:

- **The right hardware for each step.** Generation, reward, and training have different needs, so each
  is its own task environment with its own resources — GPU for rollouts and the trainer, cheap CPU for
  reward and the driver. You never pay for a GPU to run a reward function.
- **Warm pools for the expensive parts.** Loading a model into vLLM is slow, so the generator runs in
  a *reusable* environment: Flyte holds a pool of warm replicas with the model already resident, and
  every iteration reuses them. It's the single biggest speedup in an RL-for-LLMs loop, and it costs one
  line of config.
- **Autoscaling to the resources you have.** The warm pool scales between a min and max replica count
  with demand, so the rollout fan-out spreads across whatever GPUs are available and scales back down
  when the loop is idle — you don't size a cluster up front.
- **Isolation across concurrent runs.** Every run is its own set of containers with its own warm pool,
  state, and report. You (and your teammates) can run many experiments at once without them stepping on
  each other.
- **Long runs that survive failures.** RL runs for hours or days; spot preemptions and OOMs happen.
  `flyte.Checkpoint` resumes the loop mid-run, task retries recover transient errors, and the pool
  recovers replicas on its own — no control plane to write.
- **Observability and data plumbing, handled.** Every rollout, reward, and gradient step is a tracked
  task with logs and lineage; `flyte.group` organizes iterations in the DAG and `report=True` streams a
  live report. The LoRA adapter flows trainer → generator as a `flyte.io.Dir`, and the base model is
  prefetched into object storage once — no shared filesystem, no manual file shuffling.
- **One system, laptop to multi-node.** The same code runs on a single GPU or scales out by changing an
  environment — no second framework, no separate cluster to stand up and babysit.

The rest of this tutorial shows each of these in action.

---

## 1. The idea: RL for LLMs in one loop

Reinforcement learning fine-tunes a language model against a **reward** instead of against fixed target
text. Whatever the lab or the algorithm, the loop has the same shape:

```
   sample prompts
        │
        ▼
   generate several candidate answers per prompt        ← the "policy" (our LLM) acts
        │
        ▼
   score each answer with a reward function             ← how good was it?
        │
        ▼
   nudge the policy toward higher-reward answers         ← the gradient step
        │
        ▼
   repeat with the improved policy
```

### What is GRPO?

**GRPO** (Group Relative Policy Optimization) is the algorithm behind reasoning models like
DeepSeek-R1. Its one big idea: to judge whether an answer was "good," compare it against **other
answers to the same prompt**, instead of training a separate value network to predict a baseline (as
PPO does). That makes it simple and cheap to run — a perfect fit for a tutorial.

For each prompt we sample a **group** of `G` answers and compute a *group-relative advantage*:

```
advantage(answer_i) = (reward_i − mean(rewards in group)) / (std(rewards in group) + ε)
```

Answers above their group's average get a positive advantage (make them more likely); answers below it
get a negative one (make them less likely). The objective we maximize is:

```
J(θ) = mean over answers [ advantage_i · (average log-probability the policy assigns to answer_i) ]
```

That's the whole idea — no critic, no replay buffer. The tutorial implements this objective directly
so you can read exactly what every line does (see [§5.4](#54-the-grpo-update-train_step)). The full
GRPO paper adds a PPO-style clipped ratio and a KL penalty to a reference model; we leave both out to
keep the math legible — fine for the single-gradient-step-per-iteration setup here, and noted in the
code.

### Why LoRA?

Updating *all* of a model's weights each iteration is expensive, and it makes the handoff between the
trainer and the generator enormous. So we freeze the base model and train a small **LoRA adapter** — a
pair of low-rank matrices layered on top of the frozen weights. The adapter is only a few
**megabytes**, and that's what makes the warm-engine trick work: the generator keeps the big frozen
base resident and just swaps in the tiny adapter each iteration.

### Coming from Ray / RLlib?

If you've built RL with Ray, the mental map is direct:

| In Ray/RLlib | Here (Union + flyte-sdk) |
|---|---|
| Rollout worker actors | the **warm vLLM pool** (`generate`) |
| Learner / trainer | the **`train_step`** task |
| Driver / Tune loop | a plain **`async` driver task** (`train_rl`) |
| `ray.remote` calls | calling another `@task` (`await generate(...)`) |

The difference is there's no separate cluster to launch and operate. Each box is just a Python function
with a decorator; Flyte schedules them, moves data between them, retries them, and shows them in a UI.

---

## 2. How the work is laid out

The loop is four task environments plus a one-time model prefetch:

| Environment | Hardware | Job |
|---|---|---|
| `generate` (rollout) | GPU, **warm pool** | run vLLM, produce candidate answers with the current adapter |
| `score_group` (reward) | CPU | grade a group of answers (rule-based / verifiable) |
| `train_step` (trainer) | GPU | one GRPO step, emit the new LoRA adapter |
| `train_rl` (driver) | CPU | run the `for` loop, wire everything together, checkpoint |

The interesting part is the **warm pool**. Loading even a small model into a vLLM engine takes time, so
if `generate` were a fresh container on every call you'd pay that cost on every rollout. Instead
`generate` runs in a **reusable** environment: Flyte keeps a pool of warm replicas alive between calls,
each holding the loaded engine in memory, and **autoscales** that pool with demand. Everything else is
an ordinary ephemeral pod — cheap to start, and stateless between iterations.

### Warm-pool topology

Only the rollout generator is a warm pool (🔥). The driver, reward, and trainer are ephemeral (❄):

```mermaid
flowchart TB
    P["flyte.prefetch.hf_model(Qwen3-0.6B)<br/><i>runs once, before the loop</i>"] --> B["base model Dir<br/>in object store"]

    D["DRIVER · train_rl<br/>❄ ephemeral · CPU · one pod for the whole run<br/>async <code>for it in range(N)</code>:<br/>fan out → score → GRPO step → checkpoint"]

    subgraph POOL["🔥 WARM POOL · rl-grpo-rollout · GPU — ReusePolicy(replicas=1..4, concurrency=1, idle_ttl=300)"]
      direction LR
      R1["replica 1<br/>vLLM + BASE<br/>(frozen, resident)"]
      R2["replica 2<br/>vLLM + BASE<br/>(frozen, resident)"]
      R3["replica 3..4<br/>autoscaled on load"]
    end

    S["score_group × groups<br/>❄ ephemeral · CPU<br/>one task per prompt group (as_completed)"]
    T["train_step<br/>❄ ephemeral · GPU<br/>one GRPO step per iteration"]

    D -- "generate(prompt, adapter, version)" --> POOL
    POOL -- "rollouts" --> S
    S -- "rewards" --> T
    T == "new LoRA adapter (few MB, flyte.Dir)" ==> D
    D -. "attached next iteration via LoRARequest" .-> POOL

    B -. "loaded once per replica" .-> POOL
    B -. "downloaded per step" .-> T

    classDef warm fill:#fde68a,stroke:#b45309,stroke-width:2px,color:#1a1a2e;
    classDef ephem fill:#e0f2fe,stroke:#0369a1,color:#1a1a2e;
    classDef store fill:#ede9fe,stroke:#6d28d9,color:#1a1a2e;
    class R1,R2,R3 warm;
    class D,S,T ephem;
    class P,B store;
    style POOL fill:#fffbeb,stroke:#b45309,stroke-width:3px;
```

🔥 = warm / reused across iterations (`ReusePolicy`) &nbsp;·&nbsp; ❄ = ephemeral (new container per
call). In the validated run, the `generate` actions ran as Flyte **`actor`** tasks (the warm pool),
while `init_adapter` / `score_group` / `train_step` ran as ordinary **`python`** pods.

---

## 3. Getting started

Once Union/Flyte is installed, running this is trivial — a single command. You'll need:

- A Union/Flyte deployment with **GPU** capacity (the tutorial uses `L4:1`; any single modern GPU is
  enough for a small model).
- A **Hugging Face token** stored as a Union secret. The example reads a secret named `hf-token`; create
  it with `flyte create secret hf-token` (or point `HF_SECRET` in the code at an existing one).
- `flyte-sdk` pointed at your endpoint (`flyte create config ...`).

Then:

```bash
python rl_grpo_lora.py
```

That prefetches the base model and launches the training loop. The very first run also builds the
container image (vLLM + flashinfer + PEFT), which takes a few minutes; every run after reuses it, so
you're straight into training.

---

## 4. The model weights: prefetch once

Both vLLM (the generator) and Transformers/PEFT (the trainer) need the base model's weights. Rather
than have every task pull from Hugging Face, Flyte **prefetches once** into object storage and hands
the resulting directory to the tasks as a `flyte.io.Dir`:

```python
import flyte.prefetch

run = flyte.prefetch.hf_model(repo="Qwen/Qwen3-0.6B", hf_token_key="hf-token")
run.wait()
base_dir = run.outputs()[0]          # the model Dir, passed into train_rl(base=base_dir)
```

> **Note — plain weights, not vLLM-sharded.** `hf_model` can pre-shard for vLLM, but that layout isn't
> readable by the Transformers/PEFT trainer. On a single GPU (`tensor_parallel_size=1`) vLLM loads
> plain Hugging Face weights directly with no downside, so we prefetch plain weights and share one
> directory between the generator and the trainer. Pre-sharding only pays off for multi-GPU rollout
> replicas, which would then want a separate copy for the trainer.

---

## 5. Walkthrough

### 5.1 Rollouts on a warm vLLM pool (`generate`)

This is the core technique, and where Flyte's warm pool earns its keep. The environment is marked
**reusable**, so Flyte holds warm replicas between calls. The expensive per-replica work — loading the
engine and downloading each adapter — is wrapped in `@alru_cache`, so it runs once and is reused for
every call that replica handles:

```python
rollout_env = flyte.TaskEnvironment(
    name="rl-grpo-rollout",
    image=image,
    resources=flyte.Resources(cpu=4, memory="24Gi", gpu=flyte.GPU("L4", 1), shm="auto"),
    reusable=flyte.ReusePolicy(replicas=(1, 4), concurrency=1, idle_ttl=300, scaledown_ttl=120),
    secrets=[HF_SECRET],
)

@alru_cache(maxsize=1)                      # build the engine ONCE per warm replica
async def _load_engine(base_uri: str) -> Any:
    from vllm import LLM
    local_base = await flyte.io.Dir.from_existing_remote(base_uri).download()
    return LLM(model=local_base, enable_lora=True, max_lora_rank=LORA_RANK, ...)

@alru_cache(maxsize=None)                   # download each adapter version ONCE per replica
async def _adapter_local_path(adapter_uri: str) -> str:
    return await flyte.io.Dir.from_existing_remote(adapter_uri).download()

@rollout_env.task
async def generate(base: flyte.io.Dir, question: str, answer: str,
                   adapter: flyte.io.Dir, version: int, group_id: int) -> list[Rollout]:
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    engine: Any = await _load_engine(base.path)              # cached: loaded once, then warm
    adapter_path: str = await _adapter_local_path(adapter.path)

    lora = LoRARequest(f"policy-v{version}", version + 1, adapter_path)
    sampling = SamplingParams(n=GROUP_SIZE, temperature=1.0, max_tokens=MAX_NEW_TOKENS)
    outputs: Any = await asyncio.to_thread(engine.generate, [build_prompt(question)],
                                           sampling, lora_request=lora)
    return [Rollout(group_id=group_id, question=question, completion=o.text, answer=answer)
            for o in outputs[0].outputs]
```

The `ReusePolicy(replicas=(1, 4), ...)` is what makes this autoscale: Flyte runs as few as one warm
replica when idle and spins up to four as the rollout fan-out demands, then scales back down. A few
more details worth noticing:

- **`@alru_cache` keyed on the remote URI** does the warm-state caching declaratively — `_load_engine`
  (`maxsize=1`) builds the vLLM engine the first time and returns the same instance forever after;
  `_adapter_local_path` downloads each adapter version exactly once. We key on the `Dir.path` string
  (hashable) rather than the `Dir` object.
- **`enable_lora=True`** reserves adapter slots when the engine starts, and the frozen base loads once.
- Each call attaches the iteration's adapter with **`LoRARequest(name, id, path)`** — vLLM applies
  `W + (B·A)·scale` on the fly. The base in GPU memory is never touched; "swapping weights" is just
  pointing at a new adapter directory with a new id. (`id` must be ≥ 1, hence `version + 1`.)
- **`asyncio.to_thread`** runs vLLM's blocking `generate()` off the event loop, so the reusable
  replica's background heartbeat stays responsive while the GPU is busy — the standard way to call
  blocking code from an `async` task. (We don't reach for `flyte.extras.DynamicBatcher` here: it shines
  when *many concurrent producers* feed one GPU, whereas this env runs `concurrency=1` and each call
  already submits a full group of `GROUP_SIZE` sequences as one vLLM batch.)
- One call returns a whole **group** of `GROUP_SIZE` completions for one prompt — exactly the group
  GRPO needs to compute relative advantages.

### 5.2 Reward (`score_group`)

The reward is a plain CPU task. We use a **verifiable** reward — a tiny arithmetic dataset where the
answer can be checked exactly — because it's the cleanest way to watch RL actually working. We score a
**whole prompt group per call** rather than one task per rollout:

```python
def _reward(rollout: Rollout) -> float:           # pure Python: format bonus + exact-match
    reward = 0.2 if "####" in rollout.completion else 0.0
    if _extract_answer(rollout.completion) == rollout.answer:
        reward += 1.0
    return reward

@reward_env.task
async def score_group(rollouts: list[Rollout]) -> list[float]:
    return [_reward(r) for r in rollouts]
```

Why per group? The rule-based reward is microseconds of work, so a task *per rollout* would pay
container startup over and over for trivial compute (with `GROUP_SIZE=6` that's 24 tiny pods an
iteration). Scoring at the **group** granularity — the unit `generate` already returns — keeps reward
an observable, pipelined Flyte task while cutting the pod count by `GROUP_SIZE`×.

> A warm pool wouldn't be the right fix here: a pool amortizes *expensive per-replica state* (like a
> model in GPU memory), and `score_group` has none. When the reward becomes **model-based** (an
> LLM-as-judge or a reward model), it gains that state — and then it *should* run on a warm vLLM pool,
> exactly like the generator. Picking the right tool per task is part of what Flyte makes easy.

### 5.3 Pipelining generation and reward

Instead of waiting for *all* rollouts before scoring (a barrier), we launch every rollout at once and
score each group **the moment it finishes**, so reward overlaps generation that's still in flight:

```python
rollout_futs = [asyncio.create_task(generate(base, q, a, adapter, version, gid))
                for gid, (q, a) in enumerate(prompts)]

rollouts, reward_futs = [], []
for fut in asyncio.as_completed(rollout_futs):    # yields in completion order
    group = await fut
    rollouts.extend(group)
    reward_futs.append(asyncio.create_task(score_group(group)))   # score this group now

group_rewards = await asyncio.gather(*reward_futs)        # aligned with append order
rewards = [r for gr in group_rewards for r in gr]         # flatten → aligned with rollouts
```

This is just `asyncio` — `create_task` to fan out, `as_completed` to drain, `gather` to collect — but
because each `await generate(...)` / `score_group(...)` is a Flyte task call, the overlap happens
*across containers*, and Flyte spreads it over the warm pool's autoscaled replicas. Scoring at group
granularity (one reward task per group, the unit `generate` produces) keeps that overlap with far fewer
pods.

### 5.4 The GRPO update (`train_step`)

The trainer resumes the previous adapter (frozen base, trainable LoRA), computes group-relative
advantages, takes **one** policy-gradient step, and saves the new adapter:

```python
@train_env.task
async def train_step(base, rollouts, rewards, adapter, version) -> tuple[flyte.io.Dir, float, int]:
    model = PeftModel.from_pretrained(base_model, local_adapter, is_trainable=True)  # resume adapter
    advantages = _group_normalized_advantages(rollouts, rewards)                     # GRPO baseline

    optimizer.zero_grad()
    for rollout, advantage in zip(rollouts, advantages):
        # log-prob the policy assigns to the completion tokens
        seq_log_prob = completion_log_prob(model, rollout)
        loss = -advantage * seq_log_prob                  # push up good answers, down bad ones
        loss.backward()                                   # accumulate across the batch
    optimizer.step()                                      # one GRPO step

    model.save_pretrained(out_dir)                        # writes only the small adapter
    return await flyte.io.Dir.from_local(out_dir), mean_loss, contributing
```

`_group_normalized_advantages` is the GRPO formula from [§1](#what-is-grpo): standardize each reward
against its prompt group. `save_pretrained` on a PEFT model writes only `adapter_config.json` +
`adapter_model.safetensors` (a few MB) — and that `Dir` *is* the entire trainer → generator handoff.

> **Why a custom step instead of a library trainer?** Libraries like TRL's `GRPOTrainer` own the whole
> loop — they generate completions and call the reward function *internally*. That's handy for a
> self-contained job, but it would bypass our warm vLLM pool and the as-completed pipelining, which are
> the whole point of running this on Flyte. Driving one explicit gradient step keeps generation,
> reward, and the update as separate, observable Flyte tasks.

### 5.5 The driver loop (`train_rl`)

The driver is a normal `async` task that owns the `for` loop. Each iteration it fans out rollouts,
scores them, takes a GRPO step, then **checkpoints** loop state and **publishes a report**:

```python
@driver_env.task(report=True)
async def train_rl(base: flyte.io.Dir, num_iterations: int = NUM_ITERATIONS) -> flyte.io.Dir:
    cp = flyte.ctx().checkpoint
    # resume from a previous attempt if one exists ...
    adapter = await init_adapter(base)          # iteration 0 starts from a fresh adapter

    for it in range(start_iter, num_iterations):
        with flyte.group(f"iter-{it}"):         # groups this iteration's tasks in the UI
            rollouts, rewards = ...             # fan out + as_completed (see §5.3)
            adapter, loss, _ = await train_step(base, rollouts, rewards, adapter, it + 1)
            await cp.save(loop_state)           # resumable: survives preemption
            await _publish_report(history, status="running")    # live HTML report
    return adapter
```

This is where Flyte's reliability shows up with almost no code:

- **`flyte.Checkpoint`** persists `{iteration, adapter location, report history}` each step. If the
  driver is preempted (spot reclaim, OOM), it resumes mid-run instead of starting over.
- **`flyte.group("iter-N")`** nests each iteration's tasks in the UI so the DAG stays readable.

### 5.6 The live report

`report=True` plus [`report_helpers.py`](./report_helpers.py) gives you a self-contained HTML report,
re-published every iteration, with reward / accuracy / format-rate / loss charts, a per-iteration
table, and the best sample completion. It's pure Python (inline SVG, no plotting dependency), so the
CPU driver stays light. Open it from the run's **Report** tab in the Union UI and watch training move
in real time.

---

## 6. What this validates

Running `python rl_grpo_lora.py` against a Union demo cluster (Qwen3-0.6B, L4 GPUs, 3 GRPO iterations)
exercises the whole loop on real hardware:

- prefetch → `init_adapter` → **12 warm-vLLM rollouts** (4 prompts × 3 iterations) → **12 group reward
  tasks** → **3 GRPO steps** → final LoRA adapter (`v3`) returned as a `flyte.io.Dir`,
- with the live report published each iteration, the driver checkpointing per step, and the rollout
  tasks running as warm, autoscaled **`actor`** replicas.

(With a 0.6B model and a toy dataset this proves the *machinery*, not convergence — scale the model and
dataset for a real learning signal.)

---

## 7. Going further

The example is deliberately the smallest thing that runs end to end. Because it's all plain Flyte, each
of these is a small change, not a rewrite:

- **A real task and a bigger policy.** Swap `BASE_MODEL_REPO` and `DATASET` for a larger model and a
  real verifiable-reward dataset (math, code, tool use). The loop code is unchanged.
- **Model-based reward.** Replace the rule in `score_group` with a call to a second warm vLLM
  environment (an LLM judge) — the same warm-pool pattern as the generator.
- **Multi-GPU rollouts.** Raise `tensor_parallel_size` and prefetch a vLLM-sharded copy for the
  generator (keeping a plain copy for the trainer).
- **Multi-node training.** When the policy outgrows one GPU, move `train_step` to a
  `ClusteredTaskEnvironment` with `TorchRun`; the body stays nearly the same.
- **Full-weight RL.** If LoRA capacity isn't enough, train all parameters and hand off the full model
  directory instead of an adapter.
- **Serving the result.** Merge the final adapter into the base (`merge_and_unload()`) and serve it
  with `VLLMAppEnvironment`.
- **The full GRPO objective.** Add the PPO-style clipped ratio and a KL penalty to a reference model
  for stability over many steps.

---

## 8. Files & references

- [`rl_grpo_lora.py`](./rl_grpo_lora.py) — the full example: prefetch, warm-vLLM rollouts, reward,
  GRPO `train_step`, and the async driver loop.
- [`report_helpers.py`](./report_helpers.py) — dependency-free HTML/SVG report toolkit.

Patterns this tutorial builds on, from the flyte-sdk examples:

- `examples/genai/vllm/vllm_app.py` — the `flyte.prefetch.hf_model` + vLLM image recipe.
- `examples/streaming/basic_as_completed.py` — the reusable-env + `as_completed` pipelining pattern.
- `examples/checkpoint/unsloth_sft_checkpoint.py` — TRL/PEFT LoRA + `flyte.Checkpoint`.

---

The takeaway: an RL training loop — warm GPU pools, CPU reward fan-out, a resumable driver, live
reporting — usually means standing up and operating a distributed system. On Flyte with Union it's a
handful of decorated Python functions that autoscale to your hardware and stay isolated per run. Start
with this small example, then scale the model, the reward, and the hardware without changing the shape
of your code.
