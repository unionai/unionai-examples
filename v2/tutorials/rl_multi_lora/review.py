"""
Approve or reject a finished training run from the terminal.

    python review.py <run-name> --approve
    python review.py <run-name> --reject

Each ``train_rl`` run pauses on a condition named ``approve-<skill>`` after its final eval. You can
approve it from the run page in the UI; this script does the same thing with ``flyte.remote``.
"""

from __future__ import annotations

import argparse
import pathlib

import flyte
import flyte.remote

# {{docs-fragment signal}}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_name")
    decision = parser.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", action="store_true")
    args = parser.parse_args()

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    pending = [c for c in flyte.remote.Condition.listall(run_name=args.run_name) if c.name.startswith("approve-")]
    if not pending:
        raise SystemExit(f"Run {args.run_name} has no review waiting. Is it still training?")
    for condition in pending:
        condition.signal(args.approve)
        print(f"{condition.name}: {'approved' if args.approve else 'rejected'}")
# {{/docs-fragment signal}}
