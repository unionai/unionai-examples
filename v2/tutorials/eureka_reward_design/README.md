# Eureka-style reward design with Claude and sandboxes

Claude writes reward functions for MountainCar, whose own reward gives PPO no signal. Each
candidate trains PPO policies on several seeds inside sandbox sessions, which block the network,
limit the filesystem, and cap memory and time. Each policy is graded outside the sandbox. The
best candidate's training statistics go back to Claude for the next generation. At the end, the
winner and the baselines are trained again on fresh seeds, and the winning reward is published
as an artifact.

The tutorial that walks through this code is at
https://www.union.ai/docs/v2/union/tutorials/reinforcement-learning/eureka-reward-design/.

## Files

| File | What it does |
| --- | --- |
| `eureka.py` | Image, task environments, sandboxed evaluation, the Claude call, and the driver |
| `harness.py` | Runs inside each sandbox to train PPO with the candidate reward; its `evaluate` grades the policy outside the sandbox |
| `prompts.py` | The task description, reward signature, baselines, hostile examples, and reflection |
| `reports.py` | The run report, the Candidates tab, and the artifact card |
| `viz.py` | Shared HTML and SVG building blocks for reports and cards |

## Requirements

- A Union.ai cluster. The sandbox sessions need `unionai-sandbox`, which is Union.ai only. The
  example pins `unionai-sandbox==0.0.1b15`, a pre-release, so its API can change between
  versions. It uses the sandbox's `userns` backend, which runs in an ordinary unprivileged pod
  on nodes that allow unprivileged user namespaces. If the backend can't start, the session
  fails instead of running the code without isolation.
- [uv](https://docs.astral.sh/uv/), which installs the script's pinned dependencies, and a
  Flyte config for your cluster (`flyte create config`).
- An Anthropic API key stored as a secret named `anthropic-api-key` (or set `ANTHROPIC_SECRET`
  to the name of an existing secret when you launch):

  ```bash
  flyte create secret anthropic-api-key
  ```

## Run it

```bash
uv run eureka.py            # 4 generations of 6 candidates
uv run eureka.py --smoke    # 2 generations of 3 candidates
```
