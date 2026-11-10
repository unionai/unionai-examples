# Eureka-style reward design with Claude and sandboxes

Claude writes reward functions for MountainCar, whose own reward gives PPO no signal. Each
candidate trains PPO policies on several seeds inside sandbox sessions, which block the network,
limit the filesystem, and cap memory and time. The best candidate's training statistics go back
to Claude for the next generation. The winning reward is published as an artifact.

The tutorial that walks through this code is at
https://www.union.ai/docs/v2/union/tutorials/reinforcement-learning/eureka-reward-design/.

## Files

| File | What it does |
| --- | --- |
| `eureka.py` | Image, task environments, sandboxed evaluation, the Claude call, and the driver |
| `harness.py` | Runs inside each sandbox: trains PPO with the candidate reward and grades the policy |
| `prompts.py` | The task description, reward signature, baselines, hostile examples, and reflection |
| `reports.py` | The run report, the Candidates tab, and the artifact card |
| `viz.py` | Shared HTML and SVG building blocks for reports and cards |

## Requirements

- A Union cluster. The sandbox sessions need `unionai-sandbox`, which is Union only.
- An Anthropic API key stored as a secret named `anthropic-api-key` (or set `ANTHROPIC_SECRET`
  to the name of an existing secret when you launch):

  ```bash
  flyte create secret anthropic-api-key
  ```

## Run it

```bash
python eureka.py            # 4 generations of 6 candidates
python eureka.py --smoke    # 2 generations of 3 candidates
```
