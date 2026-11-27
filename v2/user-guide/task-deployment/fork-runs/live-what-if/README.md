# Live what-if analysis: fork a run while it's still running

Two long-running workloads you can fork mid-flight to ask "what if?". A fork reuses every action
the source run has already finished, re-runs only what your change touches, and leaves the source
running beside it, so you can compare the two as they roll out.

| File | Workload | What-ifs |
|---|---|---|
| `agent_whatif.py` | An incident-triage agent: plan, then think → call a tool → record the observation, then report. | A different decision, a different tool result, a guardrail in the policy, a smaller step budget. |
| `rl_whatif.py` | A GRPO-style training loop whose reward has a length bonus the policy learns to exploit. | Branch training at step k with a different reward, while the original keeps training. |

Both are deterministic simulations (a scripted policy and a fake world for the agent, a three-way
softmax policy for RL), so they run anywhere in a few minutes, without API keys or GPUs. Every
step sleeps a little so you have time to fork while the run is going.

## Requirements

- A Union backend. Forking is remote-only.
- `flyteplugins-union` 0.24.0 or later on the machine you fork from (`pip install flyteplugins-union`).
- Forking a run that is still running has to be enabled for your organization. Without it, a fork
  of a running run is refused with "Forking a running run is not enabled for this organization",
  and you can still fork either workload once it finishes.

## The agent: what if it had done something else?

```bash
python agent_whatif.py                 # launches the agent and prints its run name
```

While it runs:

1. **A different decision.** In the console, open the run, find the `call_tool` action in
   `step 1` (the log search), and choose **Edit inputs** from its menu. Change `query` to
   `checkout errors region=eu-west` and click **Fork**. The fork reuses the plan, the first think
   step and the metrics query, re-runs from the log search, and finds a different root cause (a
   DNS change instead of a database pool change). From Python: `python agent_whatif.py steer <run>`.
2. **A different tool result.** Find the `record_observation` action of the runbook lookup
   (`step 3`) and set `tool_output` to a tampered runbook (`TAMPERED_RUNBOOK` in the file), which
   tells the agent to run `kubectl delete deployment payments-db -n prod`. The agent follows the
   instruction in the tool output and runs the command. From Python:
   `python agent_whatif.py inject <run>`.
3. **A guardrail.** In `think`, set `follow_instructions_in_tool_output = False` and fork the
   tampered run with the edited file. Only the think steps re-run (their code changed), the tool
   calls that repeat are reused, and the agent now flags the runbook instead of acting on it.
4. **A smaller budget.** `flyte fork <run> agent_whatif.py investigate --max_steps 3`.

## RL: what if the reward were different from step 4 on?

```bash
python rl_whatif.py                    # launches training and prints its run name
```

Watch the root action's **Reports** tab: reward climbs while accuracy falls and answers get
longer. While it trains, open the `train_step` action for step 4, choose **Edit inputs**, set
`state.length_bonus` to `0.1`, and click **Fork**. Steps 0 to 3 are reused, the fork trains on from
step 4 with the smaller bonus, and the source keeps training with the original one. From Python:
`python rl_whatif.py branch <run> 4`.

## Useful CLI options while the source is running

```bash
flyte fork <run> --plan                               # what a fork would reuse right now
flyte fork <run> --wait-for <action> ...              # wait for an action to succeed first, so it's reused
flyte fork <run> --abort-source ...                   # abort the source once the fork is created
```

See [Fork a run](https://www.union.ai/docs/v2/union/user-guide/tasks/task-deployment/fork-runs/)
for how a fork decides what to reuse.
