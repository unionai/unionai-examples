"""E5 — Volume.fork() as a per-trial world filesystem: fork + mount + read cost vs. a blob download.

Populates a ~1 GB "world" volume once, then measures (a) cold fork, (b) mount of the fork in a
fresh task, (c) full read; and compares with downloading the same bytes as a flyte.io.Dir.

Run (CPU only; cluster needs the FUSE device plugin):
    flyte --config ~/.flyte/demo-config.yaml --project ketan --domain development \
        run e5_volume_fork.py main --n-files 100 --file-mb 10
"""

import json
import os
import time
import uuid
from pathlib import Path

import flyte
from flyte.io import Dir
from flyteplugins.union.io import ROVolume

image = (
    flyte.Image.from_debian_base(install_flyte=False, name="skyrl-e5-vol")
    .with_pip_packages("flyteplugins-union>=0.8.2")
    .with_local_v2()
)

env = flyte.TaskEnvironment(
    name="skyrl-e5-vol",
    pod_template=flyte.PodTemplate().allow_fuse(),
    image=image,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)


def _fill(root: Path, n_files: int, file_mb: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        (root / f"doc_{i:04d}.bin").write_bytes(os.urandom(file_mb * 1024 * 1024))


def _read_all(root: Path) -> int:
    return sum(len(p.read_bytes()) for p in root.rglob("*") if p.is_file())


@env.task
async def populate_world(name: str, n_files: int, file_mb: int) -> ROVolume:
    from flyteplugins.union.io import Volume

    vol = Volume.new(name=name)
    t0 = time.monotonic()
    await vol.mount()
    t_mount = time.monotonic() - t0
    t0 = time.monotonic()
    root = Path(vol.mount_path) / "world"
    print("populate mount_path:", vol.mount_path, flush=True)
    _fill(root, n_files, file_mb)
    t_write = time.monotonic() - t0
    t0 = time.monotonic()
    ro = await vol.finalize(message="world v1")
    print(json.dumps({"populate": {"mount_s": t_mount, "write_s": t_write, "finalize_s": time.monotonic() - t0}}))
    return ro


@env.task
async def fork_and_read(parent: ROVolume, fork_name: str) -> dict:
    t0 = time.monotonic()
    forked = await parent.fork(name=fork_name)
    t_fork = time.monotonic() - t0
    t0 = time.monotonic()
    await forked.mount()
    t_mount = time.monotonic() - t0
    t0 = time.monotonic()
    root = Path(forked.mount_path) / "world"
    print("fork mount_path:", forked.mount_path, flush=True)
    nbytes = _read_all(root)
    t_read = time.monotonic() - t0
    # a write proves the fork is independent
    (root / "agent_output.txt").write_text("trial wrote here")
    return {"fork_s": round(t_fork, 2), "mount_s": round(t_mount, 2), "read_s": round(t_read, 2), "bytes": nbytes}


@env.task
async def make_dir(n_files: int, file_mb: int) -> Dir:
    root = Path("/tmp/world_dir")
    _fill(root, n_files, file_mb)
    return await Dir.from_local(root)


@env.task
async def download_dir(d: Dir) -> dict:
    t0 = time.monotonic()
    local = await d.download()
    t_dl = time.monotonic() - t0
    return {"download_s": round(t_dl, 2), "bytes": _read_all(Path(local))}


@env.task
async def main(n_files: int = 100, file_mb: int = 10) -> dict:
    run_id = uuid.uuid4().hex[:8]
    parent = await populate_world(f"skyrl-e5-{run_id}", n_files, file_mb)
    forks = [fork_and_read(parent, f"skyrl-e5-{run_id}-trial{i}") for i in range(3)]
    import asyncio

    fork_results = await asyncio.gather(*forks)
    d = await make_dir(n_files, file_mb)
    dl = await download_dir(d)
    out = {"volume_fork_trials": fork_results, "blob_dir_download": dl}
    print(json.dumps(out, indent=2), flush=True)
    return out


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
