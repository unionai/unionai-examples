import datetime
from pathlib import Path

import flyte
import flyte.sandbox
from flyte.io import File
from flyte.sandbox import sandbox_environment

# sandbox_environment provides the base runtime for code sandboxes.
# Include it in depends_on so the sandbox runtime is available when tasks execute.
env = flyte.TaskEnvironment(
    name="sandbox-demo",
    image=flyte.Image.from_debian_base(name="sandbox-demo"),
    depends_on=[sandbox_environment],
)

# Auto-IO mode: pure computation
sum_sandbox = flyte.sandbox.create(
    name="sum-to-n",
    code="total = sum(range(n + 1)) if conditional else 0",
    inputs={"n": int, "conditional": bool},
    outputs={"total": int},
)

# Auto-IO mode with packages
_stats_code = """\
import numpy as np
nums = np.array([float(v) for v in values.split(",")])
mean = float(np.mean(nums))
std  = float(np.std(nums))
window_end = dt + delta
"""

stats_sandbox = flyte.sandbox.create(
    name="numpy-stats",
    code=_stats_code,
    inputs={
        "values": str,
        "dt": datetime.datetime,
        "delta": datetime.timedelta,
    },
    outputs={"mean": float, "std": float, "window_end": datetime.datetime},
    packages=["numpy"],
)

# Verbatim mode: full script control
_etl_script = """\
import json, pathlib

payload = json.loads(pathlib.Path("/var/inputs/payload").read_text())
total = sum(payload["values"])

pathlib.Path("/var/outputs/total").write_text(str(total))
"""

etl_sandbox = flyte.sandbox.create(
    name="etl-script",
    code=_etl_script,
    inputs={"payload": File},
    outputs={"total": int},
    auto_io=False,
)

# Command mode: shell pipeline
linecount_sandbox = flyte.sandbox.create(
    name="line-counter",
    command=[
        "/bin/bash",
        "-c",
        "grep -c . /var/inputs/data_file > /var/outputs/line_count || echo 0 > /var/outputs/line_count",
    ],
    inputs={"data_file": File},
    outputs={"line_count": str},
)


@env.task
async def create_text_file() -> File:
    path = Path("/tmp/data.txt")
    path.write_text("line 1\n\nline 2\n")
    return await File.from_local(str(path))


@env.task
async def payload_generator() -> File:
    path = Path("/tmp/payload.json")
    path.write_text('{"values": [1, 2, 3, 4, 5]}')
    return await File.from_local(str(path))


@env.task
async def run_pipeline() -> dict:
    # Auto-IO: sum 1..10 = 55
    total = await sum_sandbox.run.aio(n=10, conditional=True)

    # Auto-IO with numpy
    mean, std, window_end = await stats_sandbox.run.aio(
        values="1,2,3,4,5",
        dt=datetime.datetime(2024, 1, 1),
        delta=datetime.timedelta(days=1),
    )

    # Verbatim ETL
    payload = await payload_generator()
    etl_total = await etl_sandbox.run.aio(payload=payload)

    # Command mode: line count
    data_file = await create_text_file()
    line_count = await linecount_sandbox.run.aio(data_file=data_file)

    return {
        "sum_1_to_10": total,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "window_end": window_end.isoformat(),
        "etl_sum_1_to_10": etl_total,
        "line_count": line_count,
    }


if __name__ == "__main__":
    flyte.init_from_config()

    r = flyte.run(run_pipeline)
    print(f"run url: {r.url}")
