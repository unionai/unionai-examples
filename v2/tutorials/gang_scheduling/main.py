# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.11.2",
# ]
# main = "gang_scheduling_demo"
# params = ""
# ///
"""
Gang scheduling on Union: every worker starts together, or none of them start.

A distributed training job is only useful when ALL of its workers are running. If a scheduler
starts some of them and not the rest, the started workers sit on expensive GPUs doing nothing,
and two jobs competing for the same GPUs can each end up half-started, waiting on each other.

This demo submits two training jobs ("team-a" and "team-b") at the same moment. Each one is a
gang of GANG_WORKERS GPU workers, and the two together need more GPUs than the cluster has. On
Union:

1. one gang is admitted as a whole and all of its workers start together;
2. the other gang waits without starting a single worker or holding a GPU, then starts as a whole the moment the
   first gang releases its GPUs.

The run's report draws both gangs on one timeline and checks those two claims from timestamps
the workers record themselves.

Run it (size GANG_WORKERS so that 2 x GANG_WORKERS is more than the GPUs your pool can give):

    GANG_WORKERS=3 GANG_GPU=L4:1 uv run --script main.py
"""

import asyncio
import os
import time

import flyte
import flyte.report
from flyte.clustered import ClusteredTaskEnvironment, ClusterFailurePolicy, TorchRun
from pydantic import BaseModel

# {{docs-fragment knobs}}
WORKERS = int(os.getenv("GANG_WORKERS", "3"))  # pods (one GPU each) per training job
GPU = os.getenv("GANG_GPU", "L4:1")  # accelerator per worker; "" runs a CPU-only rehearsal
QUEUE = os.getenv("GANG_QUEUE", "shared")  # the cluster's Kueue LocalQueue
# {{/docs-fragment knobs}}

# The containers import this file too, so hand them the same knobs.
knobs = {"GANG_WORKERS": str(WORKERS), "GANG_GPU": GPU, "GANG_QUEUE": QUEUE}
BACKEND = "nccl" if GPU else "gloo"
image = (
    flyte.Image.from_debian_base(name="gang-scheduling-demo")
    # CUDA 12.6 wheels run on any NVIDIA driver from 560 on; CPU wheels keep the rehearsal image small.
    .with_pip_packages(
        "torch", index_url="https://download.pytorch.org/whl/cu126" if GPU else "https://download.pytorch.org/whl/cpu"
    )
    # numpy: torch's all_gather_object needs it. kubernetes: the driver serializes the workers' pod template.
    .with_pip_packages("numpy", "kubernetes")
)

# {{docs-fragment gang-env}}
gang = ClusteredTaskEnvironment(
    name="gang_scheduling_demo_workers",
    image=image,
    env_vars=knobs,
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu=GPU) if GPU else flyte.Resources(cpu=1, memory="2Gi"),
    replicas=WORKERS,
    nproc_per_node=1,
    runtime=TorchRun(rdzv_backend="static", max_restarts=0),
    failure_policy=ClusterFailurePolicy(max_restarts=0),
    # This label is the whole opt-in: the job is admitted as one gang by the cluster's queue.
    pod_template=flyte.PodTemplate(labels={"kueue.x-k8s.io/queue-name": QUEUE}),
)
# {{/docs-fragment gang-env}}

driver = flyte.TaskEnvironment(
    name="gang_scheduling_demo_driver",
    image=image,
    env_vars=knobs,
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    depends_on=[gang],
)


class Worker(BaseModel):
    job: str
    rank: int
    pod: str
    device: str
    started: float  # the worker's container started (before torchrun waits for the other pods)
    joined: float  # every worker of the gang has connected
    finished: float


def container_started() -> float:
    """Wall-clock time this container started: when its PID 1 started.

    torchrun's static rendezvous waits for every pod before it runs `train`, so the time `train` begins
    would hide a pod that came up early and sat waiting for the rest.
    """
    try:
        with open("/proc/stat") as f:
            boot = next(float(line.split()[1]) for line in f if line.startswith("btime "))
        with open("/proc/1/stat") as f:
            ticks = int(f.read().rsplit(")", 1)[1].split()[19])  # field 22: starttime, after "pid (comm)"
        return boot + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, StopIteration, ValueError, IndexError):
        return time.time()


# {{docs-fragment train}}
@gang.task
async def train(job: str, steps: int = 90) -> list[Worker]:
    """Data-parallel training of a small model. Every rank runs this function."""
    import socket

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP

    started = container_started()
    if BACKEND == "nccl":
        torch.cuda.set_device(0)
    device = torch.device("cuda", 0) if BACKEND == "nccl" else torch.device("cpu")
    dist.init_process_group(backend=BACKEND)  # returns once every worker in the gang has joined
    joined = time.time()
    rank, world = dist.get_rank(), dist.get_world_size()

    model = DDP(nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 1)).to(device))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(512, 64, device=device)
    y = x.sum(dim=1, keepdim=True)
    for step in range(steps):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()  # gradients are averaged across all workers here
        opt.step()
        if rank == 0 and step % 30 == 0:
            print(f"[{job}] step {step}/{steps} loss={loss.item():.4f} world={world}", flush=True)
        await asyncio.sleep(1)  # stands in for a real model's step time so the timeline is readable

    me = Worker(
        job=job,
        rank=rank,
        pod=socket.gethostname(),
        device=torch.cuda.get_device_name(0) if BACKEND == "nccl" else "cpu",
        started=started,
        joined=joined,
        finished=time.time(),
    )
    everyone: list = [None] * world
    dist.all_gather_object(everyone, me.model_dump())
    dist.destroy_process_group()
    return [Worker(**w) for w in everyone]  # rank 0's return value is the task output
# {{/docs-fragment train}}


# {{docs-fragment driver}}
@driver.task(report=True)
async def gang_scheduling_demo(steps: int = 90) -> str:
    """Submit two gangs at once; together they need more GPUs than the cluster has."""
    submitted = time.time()
    team_a, team_b = await asyncio.gather(
        train.override(short_name="team-a")(job="team-a", steps=steps),
        train.override(short_name="team-b")(job="team-b", steps=steps),
    )
    verdict = render_report(submitted, team_a, team_b)
    await flyte.report.flush.aio()
    return verdict
# {{/docs-fragment driver}}


def _fmt(seconds: float) -> str:
    seconds = max(0, round(seconds))
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def render_report(submitted: float, *gangs: list[Worker]) -> str:
    """Draw every worker on one timeline and check the gang-scheduling claims from its timestamps."""
    first, second = sorted(gangs, key=lambda g: min(w.started for w in g))
    first_done = max(w.finished for w in first)
    second_start = min(w.started for w in second)

    # A worker that is up but whose peers are not holds a GPU and does nothing. Gang scheduling keeps this near zero.
    idle = {g[0].job: max(w.joined - w.started for w in g) for g in gangs}
    checks = [
        (
            "pass" if all(s <= 120 for s in idle.values()) else "fail",
            "Every gang started together",
            " · ".join(f"{job}: longest any worker waited for its peers was {_fmt(s)}" for job, s in idle.items()),
        ),
        (
            "pass",
            "The waiting gang held nothing",
            f"{second[0].job} did not start a single worker while {first[0].job} was training; it waited "
            f"{_fmt(second_start - submitted)} and started {_fmt(second_start - first_done)} after the GPUs were freed",
        )
        if second_start >= first_done
        else (
            "info",
            "Both gangs fit at once",
            "They ran side by side, so nothing had to wait. Set GANG_WORKERS above half of the GPU pool "
            "to see one gang wait for the other.",
        ),
    ]

    end = max(w.finished for g in gangs for w in g)
    span = max(end - submitted, 1)
    row, left, width = 26, 150, 760
    rows = [w for g in (first, second) for w in sorted(g, key=lambda w: w.rank)]
    colors = {first[0].job: "#2563eb", second[0].job: "#d97706"}

    def x(t: float) -> float:
        return left + (t - submitted) / span * width

    bars = []
    for i, w in enumerate(rows):
        y = 34 + i * row
        tip = (
            f"{w.pod} on {w.device}: up at +{_fmt(w.started - submitted)}, all peers joined at "
            f"+{_fmt(w.joined - submitted)}, done at +{_fmt(w.finished - submitted)}"
        )
        bars.append(
            f'<text x="{left - 10}" y="{y + 13}" text-anchor="end" font-size="12" fill="#374151">'
            f"{w.job} · worker {w.rank}</text>"
            # faint: up and waiting for peers; solid: training with the whole gang
            f'<rect x="{x(w.started):.1f}" y="{y}" width="{max(x(w.joined) - x(w.started), 0):.1f}" height="18" '
            f'fill="{colors[w.job]}" opacity="0.3"><title>{tip}</title></rect>'
            f'<rect x="{x(w.joined):.1f}" y="{y}" width="{max(x(w.finished) - x(w.joined), 2):.1f}" height="18" '
            f'rx="3" fill="{colors[w.job]}"><title>{tip}</title></rect>'
        )
    height = 34 + len(rows) * row + 30
    step = max(60, int(span / 8) // 60 * 60)
    ticks = "".join(
        f'<line x1="{x(submitted + t):.1f}" y1="26" x2="{x(submitted + t):.1f}" y2="{height - 26}" stroke="#e5e7eb"/>'
        f'<text x="{x(submitted + t):.1f}" y="{height - 10}" text-anchor="middle" font-size="11" '
        f'fill="#6b7280">{_fmt(t)}</text>'
        for t in range(0, int(span) + 1, step)
    )
    timeline = (
        f'<svg viewBox="0 0 {left + width + 20} {height}" width="100%" font-family="system-ui, sans-serif">'
        f'<text x="{left}" y="14" font-size="11" fill="#6b7280">time since both jobs were submitted →  '
        f"(faint = worker up, waiting for its peers · solid = whole gang training)</text>"
        f"{ticks}{''.join(bars)}</svg>"
    )
    icon = {"pass": "✅", "fail": "❌", "info": "ℹ️"}
    check_html = "".join(
        f'<div style="margin:8px 0"><b>{icon[status]} {title}</b>'
        f'<div style="color:#4b5563;margin-left:26px">{detail}</div></div>'
        for status, title, detail in checks
    )
    flyte.report.get_tab("main").log(
        f"""
        <div style="font-family:system-ui,sans-serif;max-width:960px">
          <h2 style="margin-bottom:4px">Gang scheduling: all workers or none</h2>
          <p style="color:#4b5563;margin-top:0">Two training jobs, {WORKERS} workers each
          ({GPU or "CPU"} per worker), submitted at the same moment. Each row is one worker.</p>
          {check_html}
          {timeline}
        </div>"""
    )
    return " | ".join(f"{status.upper()} {title}: {detail}" for status, title, detail in checks)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(gang_scheduling_demo)
    print(run.name)
    print(run.url)
    run.wait()
