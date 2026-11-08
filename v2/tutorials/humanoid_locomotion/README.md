# Humanoid locomotion: teach a Unitree G1 to run

Train a running gait for the Unitree G1 humanoid with Brax PPO over MuJoCo MJX. MJX compiles
the physics into the same XLA program as the learner, so 8,192 robots step in parallel on one
GPU. The training report streams the reward curve and films the policy as it learns, three
CPU replays compare the trained policy with a random one, and the policy is published as an
artifact with a model card.

The tutorial that walks through this code is at
https://www.union.ai/docs/v2/union/tutorials/reinforcement-learning/humanoid-locomotion/.

## Files

| File | What it does |
| --- | --- |
| `learn_to_run.py` | Image, task environments, training, replays, publishing, and the driver |
| `g1.py` | The G1 running environment, PPO config, rollouts, filming, and video encoding |
| `reports.py` | The live training report, the final report, and the model card |
| `viz.py` | Shared HTML and SVG building blocks for reports and cards |

## Requirements

A Union cluster with an NVIDIA L40S (or another GPU with at least 40 GB of memory; change
`gpu_env` in `learn_to_run.py`).

## Run it

```bash
python learn_to_run.py                     # 150M steps, about 20 minutes of training
python learn_to_run.py --smoke             # 20M steps: fast, but the robot still falls
python learn_to_run.py --checkpoint <uri>  # replay and report an existing checkpoint
```
