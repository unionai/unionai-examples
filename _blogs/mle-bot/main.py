"""MLE Bot — CLI entry point.

Usage:
    # 1. Generate the synthetic demo dataset
    uv run main.py generate-data

    # 2. Submit the MLE agent to your Flyte cluster
    uv run main.py run \\
        --data data/predictive_maintenance.csv \\
        --problem "Predict pump failures 24 hours before they happen based on sensor readings" \\
        --target failure_24h \\
        --time-column timestamp

    # 3. Run with more iterations and save the model card
    uv run main.py run \\
        --data data/predictive_maintenance.csv \\
        --problem "Predict pump failures 24 hours before they happen" \\
        --target failure_24h \\
        --time-column timestamp \\
        --max-iterations 3 \\
        --output results/model_card.md

Requires:
    - ~/.flyte/config.yaml pointing at your cluster
    - The secret "openai-api-key" registered in the cluster's secret store
"""

import argparse
import asyncio
import os
import sys


def cmd_generate_data(args) -> None:
    from mle_bot.synthetic_data import generate_predictive_maintenance
    generate_predictive_maintenance(
        n_machines=args.machines,
        n_days=args.days,
        output_path=args.output,
        seed=args.seed,
    )


async def cmd_run(args) -> None:
    import flyte
    from flyte.io import File
    from mle_bot.agent import mle_agent_task

    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    # Connect to the cluster using ~/.flyte/config.yaml
    flyte.init_from_config()

    data_file = await File.from_local(os.path.abspath(args.data))

    print(f"\nSubmitting to Flyte cluster...")
    print(f"  Problem : {args.problem}")
    print(f"  Target  : {args.target}")
    print(f"  Dataset : {args.data}")

    run = await flyte.run.aio(
        mle_agent_task,
        data=data_file,
        problem_description=args.problem,
        target_column=args.target,
        time_column=args.time_column or "",
        max_iterations=args.max_iterations,
    )

    print(f"\nJob submitted: {run.url}")
    print("Streaming logs (Ctrl+C to detach)...\n")

    try:
        async for line in run.get_logs.aio():
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        print(f"\n\nDetached from logs. Job is still running.")
        print(f"Track it at: {run.url}")
        return

    await run.wait.aio()

    outputs = await run.outputs.aio()
    model_card = outputs[0]  # mle_agent_task returns a single str

    print("\n" + "=" * 60)
    print("MODEL CARD")
    print("=" * 60)
    print(model_card)

    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(model_card)
        print(f"\nModel card saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="MLE Bot — AI ML Engineer powered by Flyte durable sandbox executions"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate-data command
    gen_parser = subparsers.add_parser("generate-data", help="Generate synthetic predictive maintenance dataset")
    gen_parser.add_argument("--output", default="data/predictive_maintenance.csv", help="Output CSV path")
    gen_parser.add_argument("--machines", type=int, default=20, help="Number of machines to simulate")
    gen_parser.add_argument("--days", type=int, default=365, help="Days to simulate per machine")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # run command — submits to Flyte cluster
    run_parser = subparsers.add_parser("run", help="Submit the MLE agent to your Flyte cluster")
    run_parser.add_argument("--data", required=True, help="Path to CSV dataset (uploaded automatically)")
    run_parser.add_argument("--problem", required=True, help="Natural language problem description")
    run_parser.add_argument("--target", required=True, help="Target column name")
    run_parser.add_argument("--time-column", default="", help="Timestamp column for time-based split (optional)")
    run_parser.add_argument("--max-iterations", type=int, default=3, help="Max experiment iterations")
    run_parser.add_argument("--output", default="", help="Save model card to this path when done (optional)")
    run_parser.add_argument(
        "--inject-failure",
        action="store_true",
        default=False,
        help="Demo mode: corrupt the first experiment so the agent visibly self-heals",
    )

    args = parser.parse_args()

    if args.command == "generate-data":
        cmd_generate_data(args)
    elif args.command == "run":
        asyncio.run(cmd_run(args))


if __name__ == "__main__":
    main()
