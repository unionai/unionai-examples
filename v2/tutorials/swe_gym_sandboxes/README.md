# SWE-Gym RL rollouts on Union sandboxes

A port of Modal's [SWE-Gym example](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/swe_gym)
(`helpers/rollouts.py`, `helpers/environments.py`) to Union sandboxes. The rollout logic is
Modal's; only the `environments` layer is new. Your driver stays your own Python process.

```
your driver (laptop / other cloud / Flyte task)          Union
  model client + keys  ──────────────────────────▶  (nothing: keys stay with the driver)
  generate_rollout ── envs.sandbox(inst) ─────────▶  microVM on a CoW fork of the instance's testbed
                     envs.command(sb, cmd, 120) ──▶    bash in /testbed, no network, no secrets
                     envs.patch(sb) ──────────────▶    git diff
  grade_rollout ──── envs.grade(inst, patch) ─────▶  another fork: apply patch + hidden tests
```

## Where to look

Read it in this order. Each step names the function that does the work.

| Step | Where | What you will find |
|---|---|---|
| 1. The job | [`driver_laptop.py`](driver_laptop.py): `main`, `Job.run_group` | The whole run: get each instance's testbed, run a group of attempts per instance, grade each one, print rewards. |
| 2. A sandbox | [`union_environments.py`](union_environments.py): `sandbox`, `command`, `patch`, `grade` | Everything that touches a sandbox. A sandbox is a microVM on its own copy of the testbed; `command` is one `sb.run`. |
| 3. A testbed | [`union_environments.py`](union_environments.py): `_build`, `_validated`, `_published` | Build it once in a sandbox with network, `sb.commit()` it, check it with the gold patch, publish it for the next run. |
| 4. One attempt | [`rollouts.py`](rollouts.py): `generate_rollout`, `grade_rollout` | The agent loop: the model writes a turn, its bash commands run in the sandbox, the output goes back, until it stops. Modal's code, made async. |
| 5. The model | [`rollouts.py`](rollouts.py): `make_sampler`, `TinkerSampler`, `OpenAICompatSampler`, `OracleSampler` | Where your model plugs in. `OracleSampler` is a scripted stand-in for smoke tests. |
| 6. The data | [`swe_gym_tasks.py`](swe_gym_tasks.py): `DEFAULT_IDS`, `SPECS`, `build_script`, `eval_script`, `grade_log` | Which instances, how each repository is installed, how the hidden tests are run and scored. |
| 7. As a Flyte run | [`driver_flyte.py`](driver_flyte.py): `build_testbed`, `rollout`, `main` | The same job with cached builds, retries per attempt and traced grading. Optional. |

New to this kind of system? The
[SWE-Gym rollouts on VM sandboxes](https://www.union.ai/docs/v2/union/tutorials/reinforcement-learning/swe-gym-sandboxes/)
tutorial explains what it is for, why it is shaped this way and how the agent loop works.

## How it works: one testbed, sealed, forked per attempt

Modal starts each sandbox from the instance's Docker image. Here each instance's testbed is
built **once**, in a sandbox with network: clone at `base_commit`, install, strip the git
history after the base commit. Then its disk is committed as a `SandboxVolume`, which is
never written again. Every attempt, and every grading run, is a new microVM on a
**copy-on-write fork** of that volume:

- parallel attempts at one instance start at once, each in a few seconds, from the same bytes
- no attempt sees another's writes: each fork is private, and the testbed itself is untouched
- grading uses a fresh fork too, so nothing the agent did, outside its patch, leaks into grading
- a testbed is built once across runs: `driver_laptop.py` publishes each one as an artifact
  (`SandboxEnvironment(publish=...)`) and starts from it next time
  (`SandboxVolume.from_artifact`); in `driver_flyte.py` the build is a cached task

A testbed is checked before it is used: the gold patch must make the hidden tests pass on a
copy of it. A published testbed that fails the check is rebuilt.

From a laptop, the sandbox SDK reaches each sandbox through Union's data proxy, as the user
who is signed in. Nothing else runs in the cluster on your behalf: no task, no app.

## Setup

```sh
uv venv -p 3.12 .venv-rl && source .venv-rl/bin/activate
uv pip install "flyteplugins-union>=0.21.2"
```

- A Union data plane with microVM support enabled. It needs special node configuration and
  setup: talk to your Union.ai solutions team.
- A flyte config for your Union control plane (your current one, or `--config <cfg>`).
  `driver_flyte.py` also needs it to build images remotely: `image: {builder: remote}`.
- For a run with nobody at the keyboard, sign in with an API key (`flyte create api-key`,
  then `FLYTE_API_KEY`). A browser sign-in that expires mid-run waits for you.
- Sampler extras: `pip install tinker "transformers[chat-template]"` for `--sampler tinker`
  or `--sampler openai` (both use Qwen3.5-9B's tokenizer). `--smoke` needs neither.

Model credentials live only where the driver runs: `TINKER_API_KEY` / `TINKER_BASE_URL`
(or `OPENAI_API_KEY`) in the laptop's environment; for `driver_flyte.py`, a Union secret
`TINKER_API_KEY` given to the driver tasks only. Sandboxes get no secrets, and only the
sandbox that builds a testbed has network.

## Run

```sh
# Laptop driver, as on Modal: sandboxes on Union, everything else here
python driver_laptop.py --project <P> --smoke
python driver_laptop.py --project <P> --sampler tinker
python driver_laptop.py --project <P> --sampler openai --base-url http://my-vllm:8000/v1

# Flyte driver: the whole thing runs in the cluster
python driver_flyte.py --project <P> --smoke
python driver_flyte.py --project <P> --sampler tinker
```

`--smoke` is 2 instances x 4 attempts with `OracleSampler`, so it needs no model or GPU: even
attempts apply the gold patch (reward 1), odd attempts give up (reward 0). Rewards of
`1 0 1 0` per instance show testbeds, forks, patch capture and hidden-test grading all work.

```
sandboxes of this job are labelled swe-gym-run=0612-1432-a3f9

instance                         attempt reward turns  stop
getmoto__moto-7023                     0    1.0     3  final
getmoto__moto-7023                     1    0.0     2  final
getmoto__moto-7023                     2    1.0     3  final
getmoto__moto-7023                     3    0.0     2  final
python__mypy-16781                     0    1.0     3  final
python__mypy-16781                     1    0.0     2  final
python__mypy-16781                     2    1.0     3  final
python__mypy-16781                     3    0.0     2  final

mean reward 0.50 over 8 attempts
group 0 getmoto__moto-7023: mixed=True (GRPO advantages +0.50, -0.50, +0.50, -0.50)
group 1 python__mypy-16781: mixed=True (GRPO advantages +0.50, -0.50, +0.50, -0.50)
```

`stop` is why the attempt ended: `final`, `turn_limit`, `context_limit` or `generation_limit`.
Each attempt's trajectory, patch and test output are under `--out`, with the timings of every
step in `events.jsonl`, as in Modal's example.

### Finding a job's sandboxes

Outside a task every sandbox is a run of its own. `driver_laptop.py` puts one label on all the
sandboxes of a job (`SandboxEnvironment(labels=...)`) and prints it, so they can be listed
together:

```sh
flyte get run --project <P> --domain development --with-label swe-gym-run=0612-1432-a3f9
```

`--run-id` sets the label's value, to give a job a name of your own. In `driver_flyte.py` the
sandboxes are actions of the Flyte run, which groups them already.

## Union vs Modal

| | Modal sandboxes | Union sandboxes (this port) |
|---|---|---|
| Isolation | gVisor container | Firecracker microVM: its own kernel; network enforced outside the VM |
| Per-instance environment | custom image per instance (`Image.from_registry`) | **not yet** (see below): the platform's base image, and a testbed built once into a sealed volume |
| Fork / snapshot | filesystem and memory snapshots | `sb.commit()` → `SandboxVolume`, CoW-forked by any number of sessions on any node; `sb.fork()`; `sb.snapshot()` with memory and processes |
| Network | `block_network=True` | `Network.blocked()` (default), `.pypi()`, `.allow(domains/CIDRs)`, `.internet()` |
| Driving from outside | the SDK, from anywhere | the SDK, from anywhere, as your Union sign-in |
| Durability | your driver's process | Flyte, if you want it: cached testbed builds, per-attempt retries, traced grading, every run and sandbox in the console |

## What Union doesn't have yet: a custom image per sandbox

`SandboxEnvironment(image=...)` raises: a session runs the platform's Python base (Debian 13).
Modal's example starts each sandbox from the instance's SWE-Gym Docker image.

The port builds the same environment instead. `swe_gym_tasks.build_script` rebuilds each
instance's testbed from source in a sandbox with network (apt, `uv` for the right Python, the
harness's install line), with a shim at `/opt/miniconda3/bin/activate` so the rollout's
`source /opt/miniconda3/bin/activate testbed` runs unchanged, then commits it.

What that costs:

- a few minutes per instance, once
- repos that need conda or heavy native builds (pandas, dask, MONAI) aren't in the subset
- the environment can differ from the SWE-Gym image, which is why each testbed is checked
  with the gold patch, and PASS_TO_PASS tests that fail on it aren't graded

What would remove it: `SandboxEnvironment(image="docker.io/xingyaoww/sweb.eval.x86_64.<id>:latest")`
(or a `flyte.Image`), booting a microVM whose root filesystem is that image, with
`sb.commit()` and forks working on top of it as they do on the base image.
