# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.15.0",
#    "kubernetes",
# ]
# ///

"""Test RDMA between two MI355X nodes from pods on the host network.

From a pod on the cluster network, the Pensando NICs show up but every RDMA
connection fails when the queue pair moves to ready-to-receive: the NIC
addresses belong to the host's network namespace, which the pod is not in.
Knative apps cannot use the host network on this cluster, so this runs as two
Flyte tasks instead, a and b, each on its own node with hostNetwork set.

Each task records its RDMA devices and addresses, publishes them under the
results prefix, and waits for the other. Then a drives these tests against b:

- RDMA write bandwidth on each NIC pair, all pairs at once, and one cross-rail pair.
- RDMA write latency at 64 KB and 4 MB.
- Mooncake transfer engine reads and writes over RDMA, from host memory and from GPU memory.
- The same Mooncake transfers over TCP, for comparison with the pod network.

Both tasks exit when a finishes, which releases the nodes.

    uv run kv_rdma_probe.py --results <object store prefix>
"""

import argparse
import asyncio
import json
import os

import flyte
from kubernetes.client import (
    V1Capabilities,
    V1Container,
    V1EmptyDirVolumeSource,
    V1PodSpec,
    V1ResourceRequirements,
    V1SecurityContext,
    V1Toleration,
    V1Volume,
    V1VolumeMount,
)

from common import GPUS_PER_NODE, QUEUE
from kimi_app import VLLM_BASE

RDMA_RESOURCE = "rdma/rdma_shared_device_ib"
MOONCAKE = "mooncake-transfer-engine-rocm==0.3.14"

# The vLLM ROCm image already carries AMD's ionic verbs provider and ROCm, which
# the Mooncake ROCm wheel links against. Its entrypoint is vllm, so a small shim
# in front of the real binary passes Flyte's task command through.
SHIM_B64 = "IyEvYmluL3NoCmlmIFsgIiQxIiA9IGEwIF07IHRoZW4gZXhlYyAiJEAiOyBmaQpleGVjIC9vcHQvcnVudGltZS12ZW52L2Jpbi92bGxtLXJlYWwgIiRAIgo="  # base64 of a sh script: a0 runs as given, anything else goes to vllm
image = (
    flyte.Image.from_base(VLLM_BASE)
    .clone(name="kv-rdma", python_version=(3, 12), extendable=True)
    .with_apt_packages("ibverbs-utils", "perftest", "iperf3", "iproute2", "pciutils")
    .with_commands(
        [
            "mv /opt/runtime-venv/bin/vllm /opt/runtime-venv/bin/vllm-real",
            f"echo {SHIM_B64} | base64 -d > /opt/runtime-venv/bin/vllm && chmod +x /opt/runtime-venv/bin/vllm",
            f"/opt/runtime-venv/bin/python -m pip install --target /opt/mc {MOONCAKE}"
            f" || uv pip install --python /opt/runtime-venv/bin/python --target /opt/mc {MOONCAKE}",
        ]
    )
    .with_pip_packages("flyte==2.15.0", "kubernetes")
)


def node_pod_template() -> flyte.PodTemplate:
    res = {"amd.com/gpu": str(GPUS_PER_NODE), RDMA_RESOURCE: "1"}
    return flyte.PodTemplate(
        primary_container_name="primary",
        pod_spec=V1PodSpec(
            host_network=True,
            dns_policy="ClusterFirstWithHostNet",
            containers=[
                V1Container(
                    name="primary",
                    resources=V1ResourceRequirements(requests=res, limits=res),
                    security_context=V1SecurityContext(capabilities=V1Capabilities(add=["IPC_LOCK"])),
                    volume_mounts=[V1VolumeMount(name="shm", mount_path="/dev/shm")],
                )
            ],
            volumes=[V1Volume(name="shm", empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="64Gi"))],
            tolerations=[V1Toleration(key="amd.com/gpu", operator="Exists", effect="NoSchedule")],
        ),
    )


node_env = flyte.TaskEnvironment(
    name="kv-rdma-node",
    image=image,
    resources=flyte.Resources(cpu="48", memory="256Gi"),
    pod_template=node_pod_template(),
    queue=QUEUE,
)

driver_env = flyte.TaskEnvironment(
    name="kv-rdma",
    # The driver only waits on the two node tasks, so it skips the large vLLM image.
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("kubernetes"),
    resources=flyte.Resources(cpu="1", memory="2Gi"),
    depends_on=[node_env],
    queue=QUEUE,
)


def _sh(cmd: str, timeout: int = 120) -> tuple[int, str]:
    import subprocess

    try:
        p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"


def _devices() -> list[dict]:
    """Pensando devices with their first routable RoCE v2 GID (the fabric is IPv6)."""
    import glob
    import re

    devs = []
    for d in sorted(glob.glob("/sys/class/infiniband/ionic*")):
        name = os.path.basename(d)
        out = _sh(f"ibv_devinfo -v -d {name}")[1]
        gids = [
            (int(m.group(1)), m.group(2))
            for m in re.finditer(r"GID\[\s*(\d+)\]:\s+(\S+), RoCE v2", out)
            if not m.group(2).startswith("fe80")
        ]
        state = "PORT_ACTIVE" in out
        devs.append({"dev": name, "active": state, "gid_index": gids[0][0] if gids else None, "gid": gids[0][1] if gids else None})
    return devs


class _Store:
    """JSON files under the results prefix."""

    def __init__(self, res: str):
        self.res = res.rstrip("/")

    async def put(self, name: str, obj) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(obj, f, indent=1)
        await flyte.storage.put(f.name, f"{self.res}/{name}")

    async def get(self, name: str):
        path = f"{self.res}/{name}"
        try:
            flyte.storage.get_underlying_filesystem(path=path).invalidate_cache()
        except Exception:
            pass
        if not await flyte.storage.exists(path):
            return None
        with open(await flyte.storage.get(path)) as f:
            return json.load(f)


def _bw(out: str):
    import re

    rows = [line.split() for line in out.splitlines() if re.match(r"^\s*\d+\s+\d+\s+[\d.]+\s+[\d.]+", line)]
    return float(rows[-1][3]) if rows else None


def _lat(out: str):
    import re

    rows = [line.split() for line in out.splitlines() if re.match(r"^\s*\d+\s+\d+\s+[\d.]+", line)]
    return {"typical_us": float(rows[-1][4]), "p99_us": float(rows[-1][7])} if rows and len(rows[-1]) >= 8 else None


MC_ENV = (
    "cd /opt/mc/mooncake; export LD_LIBRARY_PATH=/opt/mc/mooncake:/opt/mc/mooncake.libs:$LD_LIBRARY_PATH "
    "MC_GID_INDEX=1 MC_ENABLE_DEST_DEVICE_AFFINITY=1 MC_NUM_TCP_IO_THREADS=16 MC_TCP_ENABLE_CONNECTION_POOL=1 "
    "MC_TCP_LANES_PER_PEER=8 MC_TCP_MAX_CONNECTIONS_PER_PEER=16"
)
BW_PORT, LAT_PORT = 18515, 18600
GiB = 1 << 30
# Mooncake cases: name -> (protocol, GPU memory, NICs ("all" or a list), buffer bytes, operations[, extra env]).
OPS_FULL = (("read", 4194304), ("write", 4194304), ("read", 1048576), ("read", 262144))
OPS_SHORT = (("read", 4194304), ("write", 4194304))
MOONCAKE_CASES = {
    "full": {
        "rdma-dram": ("rdma", False, "all", 8 * GiB, OPS_FULL),
        "rdma-vram": ("rdma", True, "all", 8 * GiB, OPS_SHORT),
        "tcp-dram": ("tcp", False, None, 8 * GiB, OPS_FULL),
    },
    # Narrows down why registering host memory fails on the Pensando NICs.
    "mooncake": {
        "dram-no-relaxed-order": ("rdma", False, "all", 1 * GiB, OPS_SHORT, "MC_IB_PCI_RELAXED_ORDERING=0"),
        "dram-mr-64m": ("rdma", False, "all", 1 * GiB, OPS_SHORT, "MC_MAX_MR_SIZE=67108864"),
        "dram-no-ro-mr-64m": ("rdma", False, "all", 1 * GiB, OPS_SHORT, "MC_IB_PCI_RELAXED_ORDERING=0 MC_MAX_MR_SIZE=67108864"),
        "dram-no-ro-8g": ("rdma", False, "all", 8 * GiB, OPS_SHORT, "MC_IB_PCI_RELAXED_ORDERING=0"),
    },
}
BW_ARGS = "-s 4194304 -q 8 -D 10"


@node_env.task
async def node(role: str, results: str, mode: str = "full") -> str:
    import re
    import socket
    import subprocess
    import time

    store = _Store(results)
    peer_role = "b" if role == "a" else "a"
    ip = _sh("ip -4 route get 1.1.1.1 | grep -o 'src [0-9.]*' | cut -d' ' -f2")[1].strip()
    me = {
        "role": role,
        "host": socket.gethostname(),
        "ip": ip,
        "devices": _devices(),
        "ip_addr": _sh("ip -br addr")[1],
        "ip6_route": _sh("ip -6 route | head -40")[1],
        "rdma_link": _sh("rdma link show")[1],
        "mr_limits": _sh("ibv_devinfo -v -d ionic_0 | grep -i -E 'max_mr|page_size_cap|device_cap_flags' ; ulimit -l")[1],
    }
    print(json.dumps({k: me[k] for k in ("role", "host", "ip")}), [(d["dev"], d["gid"]) for d in me["devices"]], flush=True)
    devs = [d for d in me["devices"] if d["active"] and d["gid_index"] is not None]
    names = ",".join(d["dev"] for d in devs)

    cases = MOONCAKE_CASES[mode]

    def nic_flag(nics):
        if nics is None:
            return ""
        return f"--device_name={names if nics == 'all' else ','.join(nics)}"

    target_logs = {}
    if role == "b":
        procs = []
        if mode == "full":
            # perftest servers exit after one test, so each runs in a loop.
            for i, d in enumerate(devs):
                loop = f"while true; do ib_write_bw -d {d['dev']} -i 1 -x {d['gid_index']} -p {BW_PORT + i} -F --report_gbits {BW_ARGS} >/dev/null 2>&1; done"
                procs.append(subprocess.Popen(["bash", "-c", loop]))
            for j, size in enumerate((65536, 4194304)):
                d = devs[0]
                loop = f"while true; do ib_write_lat -d {d['dev']} -i 1 -x {d['gid_index']} -p {LAT_PORT + j} -F -s {size} -n 2000 >/dev/null 2>&1; done"
                procs.append(subprocess.Popen(["bash", "-c", loop]))
            procs.append(subprocess.Popen(["bash", "-c", "exec iperf3 -s >/dev/null 2>&1"]))
        targets = {}
        for k, (name, case) in enumerate(cases.items()):
            proto, vram, nics, buf, _ops = case[:5]
            cenv = case[5] if len(case) > 5 else ""
            log = f"/tmp/te-target-{name}.log"
            target_logs[name] = log
            cmd = (
                f"{MC_ENV} {cenv}; exec ./transfer_engine_bench --mode=target --metadata_server=P2PHANDSHAKE "
                f"--local_server_name={ip}:{12345 + 10 * k} --protocol={proto} {nic_flag(nics)} "
                f"--use_vram={'true' if vram else 'false'} --gpu_id={1 + k if vram else 0} --buffer_size={buf} > {log} 2>&1"
            )
            procs.append(subprocess.Popen(["bash", "-c", cmd]))
            time.sleep(12)
            m = re.findall(r"listening on [0-9.]+:(\d+)", open(log).read())
            targets[name] = {"rpc_port": int(m[-1]) if m else None, "log": open(log).read()[-1500:]}
        me["mooncake_targets"] = targets

    await store.put(f"peers/{role}.json", me)
    peer = None
    for _ in range(360):
        peer = await store.get(f"peers/{peer_role}.json")
        if peer:
            break
        await asyncio.sleep(10)
    if not peer:
        return f"{role}: peer never appeared"

    if role == "b":
        # Serve until a says it is done.
        for _ in range(720):
            if await store.get("DONE.json"):
                break
            await asyncio.sleep(10)
        # The targets' logs show the receiving side of any failed transfer.
        return json.dumps({name: open(log).read()[-3000:] for name, log in target_logs.items()})

    results_doc = {"a": me, "b": peer, "tests": []}

    async def record(name, rc, out, **extra):
        row = {"test": name, "rc": rc, **extra, "out": out[-2500:]}
        results_doc["tests"].append(row)
        print(time.strftime("%H:%M:%S"), json.dumps({k: v for k, v in row.items() if k != "out"}), flush=True)
        await store.put("results.json", results_doc)

    theirs = [d for d in peer["devices"] if d["active"] and d["gid_index"] is not None]
    n = min(len(devs), len(theirs))
    bip = peer["ip"]

    async def perftest(name, pairs, tool="ib_write_bw", args=BW_ARGS, port_of=lambda j: BW_PORT + j, parse=_bw):
        # pairs: list of (my index, their index)
        time.sleep(3)
        cmds = [
            f"{tool} -d {devs[i]['dev']} -i 1 -x {devs[i]['gid_index']} -p {port_of(j)} -F --report_gbits {args} {bip}"
            for i, j in pairs
        ]
        ps = [subprocess.Popen(["bash", "-c", c], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for c in cmds]
        outs = []
        for p in ps:
            try:
                outs.append(p.communicate(timeout=90)[0])
            except subprocess.TimeoutExpired:
                p.kill()
                outs.append("timeout")
        vals = [parse(o) for o in outs]
        rc = 0 if all(v is not None for v in vals) else 1
        extra = {"per_pair": vals, "pairs": [f"{devs[i]['dev']}->{theirs[j]['dev']}" for i, j in pairs]}
        if parse is _bw and rc == 0:
            extra["total_gbits"] = round(sum(vals), 1)
        await record(name, rc, "\n----\n".join(outs) if rc else "", **extra)
        return rc

    if mode != "full":
        n = 0
    rc, out = _sh(f"iperf3 -c {bip} -P 16 -t 15 -J", timeout=60) if mode == "full" else (0, "")
    try:
        g = round(json.loads(out)["end"]["sum_received"]["bits_per_second"] / 1e9, 1)
    except Exception:
        g = None
    if mode == "full":
        await record("tcp-host-network-iperf3-P16", rc, "" if g else out, gbits=g)

    if n and await perftest("rdma-write-bw-pair0", [(0, 0)]) == 0:
        for i in range(1, n):
            await perftest(f"rdma-write-bw-pair{i}", [(i, i)])
        await perftest("rdma-write-bw-all-pairs", [(i, i) for i in range(n)])
        if n > 1:
            await perftest("rdma-write-bw-cross-rail-0to1", [(0, 1)])
        for j, size in enumerate((65536, 4194304)):
            await perftest(f"rdma-write-lat-{size}", [(0, 0)], tool="ib_write_lat", args=f"-s {size} -n 2000",
                     port_of=lambda _j, j=j: LAT_PORT + j, parse=_lat)

    targets = peer.get("mooncake_targets", {})
    for k, (name, case) in enumerate(cases.items()):
        proto, vram, nics, buf, ops = case[:5]
        cenv = case[5] if len(case) > 5 else ""
        t = targets.get(name) or {}
        if not t.get("rpc_port"):
            await record(f"mooncake-{name}", 1, f"target did not start: {t.get('log', '')}")
            continue
        for op, bs in ops:
            cmd = (
                f"{MC_ENV} {cenv}; timeout 90 ./transfer_engine_bench --mode=initiator --metadata_server=P2PHANDSHAKE "
                f"--local_server_name={ip}:{12400 + len(results_doc['tests'])} --segment_id={bip}:{t['rpc_port']} "
                f"--protocol={proto} {nic_flag(nics)} --use_vram={'true' if vram else 'false'} "
                f"--gpu_id={1 + k if vram else 0} --buffer_size={buf} --operation={op} --block_size={bs} "
                f"--batch_size=32 --threads=16 --duration=15 --report_unit=GB"
            )
            rc, out = _sh(cmd, timeout=120)
            m = re.findall(r"throughput ([\d.]+) GB/s", out)
            await record(f"mooncake-{name}-{op}-{bs}", 0 if m else 1, "" if m else out, gbytes_per_s=float(m[-1]) if m else None)

    await store.put("DONE.json", {"ts": time.time()})
    # Failed rows keep the tail of their output, so the run's outputs show why.
    return json.dumps([{k: (v[-1500:] if k == "out" else v) for k, v in r.items() if k != "out" or r["rc"]} for r in results_doc["tests"]])


@driver_env.task
async def probe(results: str, mode: str = "full") -> list[str]:
    return list(await asyncio.gather(node("a", results, mode), node("b", results, mode)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Object store prefix for the reports")
    parser.add_argument("--mode", default="full", choices=sorted(MOONCAKE_CASES), help="full, or only the Mooncake cases")
    a = parser.parse_args()
    flyte.init_from_config()
    run = flyte.with_runcontext(queue=QUEUE).run(probe, results=a.results, mode=a.mode)
    print(run.name, run.url)
