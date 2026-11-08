# Closed-loop molecular discovery

An active learning loop over 20,000 drug-like molecules from ZINC250k. Each round, an ensemble
of MLPs on Morgan fingerprints proposes a batch of molecules by a lower confidence bound, a
scientist approves the batch (or a timeout does), and a warm pool of CPU workers measures each
molecule's HOMO-LUMO gap with GFN2-xTB. A random search with the same budget runs alongside as
the control. Every measurement is cached per molecule, every round is a cached task, and the
smallest-gap molecules are published as a dataset artifact.

The tutorial that walks through this code is at
https://www.union.ai/docs/v2/union/tutorials/frontier-ai/closed-loop-discovery/.

## Files

| File | What it does |
| --- | --- |
| `discovery.py` | Image, task environments, the pool, the oracle task, proposals, the approval gate, rounds, and the driver |
| `oracle.py` | The GFN2-xTB measurement: conformers, xTB optimization, retries, and failure reasons |
| `surrogate.py` | Fingerprints, the MLP ensemble, and the diverse batch selection |
| `reports.py` | The report and its tabs, the approval prompt, the CSV, and the dataset card |
| `viz.py` | Shared HTML and SVG building blocks for reports and cards |

## Requirements

A Union cluster. Everything runs on CPUs.

## Run it

```bash
python discovery.py            # 10 rounds of 64 over 20,000 molecules
python discovery.py --smoke    # 2 rounds of 16 over 2,000 molecules
```

Each round waits for approval of its batch. Approve or reject it in the UI, or from Python:

```python
import flyte
from flyte.remote import Condition

flyte.init_from_config()
Condition.get("approve-round-3", run_name="<run name>").signal(True)
```

If nobody answers within `approval_timeout_s` (180 s by default), the batch is approved
automatically. Pass `approval_timeout_s=0` to run without the gate.
