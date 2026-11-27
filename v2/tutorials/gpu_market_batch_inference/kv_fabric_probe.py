# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.13.1",
#    "kubernetes",
# ]
# ///

"""Measure the network between two MI355X nodes before building a shared KV tier.

Deploys two apps, a and b, which the market places on two separate GPU nodes.
Each pod records what it sees: the RDMA devices handed in by the shared device
plugin, their RoCE GIDs, the verbs providers in the image, and the KV transfer
packages in vLLM's environment. The pods find each other through a small JSON
file under the results prefix. Then a drives the tests against b:

- TCP over the pod network with iperf3, the path a TCP-only KV store would use.
- RDMA write bandwidth on each NIC pair, then on all pairs at once.
- One cross-rail pair, which shows whether NIC i can reach a different NIC on the peer.
- RDMA write latency at KV-chunk sizes.

Everything lands under the results prefix as it finishes. Afterwards both pods
stay idle until the apps are stopped.

    uv run kv_fabric_probe.py --results <object store prefix>
"""

import argparse
import os

import flyte
import flyte.app
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

from common import GPUS_PER_NODE
from kimi_app import CACHE_DIR, image as vllm_image

PORT = 8080
RDMA_RESOURCE = "rdma/rdma_shared_device_ib"

image = vllm_image.with_apt_packages(
    "ibverbs-utils", "perftest", "infiniband-diags", "iperf3", "iproute2", "pciutils", "ethtool"
)

PRELUDE = r"""
set -uo pipefail
export AWS_ACCESS_KEY_ID="$FLYTE_AWS_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$FLYTE_AWS_SECRET_ACCESS_KEY" AWS_REGION=auto
printf '%s' "$3" > /tmp/probe.py
exec /opt/runtime-venv/bin/python -u /tmp/probe.py "$1" "$2"
"""

# Uses only the standard library, so it runs in the image's Python as is.
PROBE = r'''
import glob, http.server, json, os, re, socket, socketserver, subprocess, sys, threading, time

ROLE, RES = sys.argv[1], sys.argv[2].rstrip("/")
PEER = "b" if ROLE == "a" else "a"
OUT = f"/work/{ROLE}"
os.makedirs(OUT, exist_ok=True)
S5 = ["s5cmd", "--endpoint-url", os.environ["FLYTE_AWS_ENDPOINT"], "--log", "error"]
PORT = int(os.environ["SERVE_PORT"])
BASE_PORT = 18515


def log(*a):
    line = " ".join(str(x) for x in a)
    print(time.strftime("%H:%M:%S"), line, flush=True)
    with open(f"{OUT}/progress.log", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {line}\n")


def sh(cmd, timeout=120):
    try:
        p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, f"timeout after {timeout}s: {(e.stdout or b'')[-2000:]!r}"


def upload(path, sub=""):
    subprocess.run([*S5, "cp", path, f"{RES}/{sub}{ROLE}/"], check=False)


def put_json(name, obj, sub=""):
    path = f"{OUT}/{name}"
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    upload(path, sub)


def fetch(remote):
    p = subprocess.run([*S5, "cat", remote], capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


# Health and control endpoint. The peer asks b to start perftest servers here.
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/serve?"):
            q = dict(kv.split("=", 1) for kv in self.path.split("?", 1)[1].split("&"))
            for spec in q["pairs"].split(","):
                dev, gid, port = spec.split(":")
                cmd = f"{q['tool']} -d {dev} -i 1 -x {gid} -p {port} -F --report_gbits {q['args'].replace('+', ' ')}"
                subprocess.Popen(["bash", "-c", f"exec {cmd} > {OUT}/server-{port}.log 2>&1"])
            time.sleep(2)
        body = b"ok\n"
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)


socketserver.ThreadingTCPServer.allow_reuse_address = True
threading.Thread(target=socketserver.ThreadingTCPServer(("0.0.0.0", PORT), Handler).serve_forever, daemon=True).start()
log("role", ROLE, "host", socket.gethostname())


def read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return None


def rdma_devices():
    """Each device's port state, rate and the RoCE v2 GID to use.

    Reads GIDs through verbs: sysfs shows an empty table inside the pod. The
    backend fabric is IPv6, so take the first RoCE v2 GID that is not link-local.
    """
    devs = []
    for d in sorted(glob.glob("/sys/class/infiniband/*")):
        name = os.path.basename(d); port = f"{d}/ports/1"
        info = {"dev": name, "state": read(f"{port}/state"), "rate": read(f"{port}/rate"),
                "link_layer": read(f"{port}/link_layer"), "gids": []}
        rc, out = sh(f"ibv_devinfo -v -d {name}")
        for m in re.finditer(r"GID\[\s*(\d+)\]:\s+(\S+), (RoCE v\d)", out):
            info["gids"].append({"index": int(m.group(1)), "gid": m.group(2), "type": m.group(3)})
        v2 = [g for g in info["gids"] if g["type"] == "RoCE v2" and not g["gid"].startswith("fe80")]
        info["roce_v2_gid"] = v2[0]["index"] if v2 else None
        devs.append(info)
    return devs


# 1. What this pod can see.
INVENTORY = {
    "kernel": "uname -r; cat /etc/os-release | head -3",
    "nics_pci": "lspci -nn | grep -i -E '1dd8|ethernet|infiniband' | head -40",
    "dev_infiniband": "ls -l /dev/infiniband",
    "sys_infiniband": "ls /sys/class/infiniband",
    "verbs_providers": "ls /etc/libibverbs.d; ls /usr/lib/x86_64-linux-gnu/libibverbs/ 2>/dev/null; dpkg -l | grep -i -E 'rdma|ibverbs|ionic' ",
    "ibv_devices": "ibv_devices",
    "ibv_devinfo": "ibv_devinfo -v | head -150",
    "rdma_link": "rdma link show; rdma resource show 2>/dev/null | head",
    "ip": "ip -br addr; ip route",
    "limits": "ulimit -l; cat /proc/self/status | grep -i cap; nproc; free -g",
    "py_packages": "/opt/runtime-venv/bin/python -c 'import importlib.metadata as m; [print(d.metadata[\"Name\"], d.version) for d in m.distributions()]' | sort | grep -i -E 'vllm|torch|lmcache|mooncake|nixl|mori|ucx|aiter|triton|rixl|amd'",
    "kv_connectors": "/opt/runtime-venv/bin/python -c 'from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory as F; print(sorted(F._registry))'",
    "native_libs": "ls /opt/rocm/lib 2>/dev/null | grep -i -E 'rccl|mori|ionic|ucx|nixl' ; find / -xdev \\( -name 'libionic*' -o -name 'libmori*' -o -name 'libnixl*' -o -name 'libucp*' \\) 2>/dev/null | head",
}
inv = {}
for k, cmd in INVENTORY.items():
    rc, out = sh(cmd)
    inv[k] = {"rc": rc, "out": out[-6000:]}


def verbs_devices():
    rc, out = sh("ibv_devices")
    lines = out.splitlines()
    return [l.split()[0] for l in lines[2:] if l.strip()] if rc == 0 else []


# The Pensando verbs provider (libionic) is in rdma-core 61 and later. Older
# distro images lack it, so fall back to AMD's package for this Ubuntu release,
# preferring the bundle that matches the host driver.
def install_libionic():
    repo = "https://repo.radeon.com/amdainic/pensando/ubuntu"

    def get(url):
        rc, out = sh(f"curl -fsSL --max-time 60 {url}")
        if rc:
            raise RuntimeError(f"{url}: {out[-200:]}")
        return out
    codename = sh(". /etc/os-release; echo $VERSION_CODENAME")[1].strip()
    host = {m: read(f"/sys/module/{m}/version") for m in ("ionic", "ionic_rdma")}
    index = get(f"{repo}/")
    bundles = [b.rstrip("/") for b in re.findall(r'href="(\d[^"/]*)/"', index)]
    hv = " ".join(v for v in host.values() if v)
    match = [b for b in bundles if b in hv or hv.startswith(b.split("-")[0])]
    order = (match[-1:] or []) + list(reversed(bundles))
    tried = []
    for b in order:
        try:
            pk = get(f"{repo}/{b}/dists/{codename}/main/binary-amd64/Packages")
        except Exception as e:
            tried.append((b, repr(e)[:80])); continue
        files = re.findall(r"^Filename: (pool/\S*/libionic1_[^/\s]+\.deb)$", pk, re.M)
        if not files:
            tried.append((b, "no libionic1")); continue
        deb = f"/tmp/{os.path.basename(files[-1])}"
        rc, out = sh(f"curl -fsSL --max-time 120 -o {deb} {repo}/{b}/{files[-1]} && dpkg -i --force-depends {deb}")
        return {"host_driver": host, "codename": codename, "bundle": b, "deb": os.path.basename(deb),
                "dpkg_rc": rc, "dpkg_out": out[-1500:], "tried": tried}
    return {"host_driver": host, "codename": codename, "error": "no libionic1 package found", "tried": tried}


verbs = {"system": verbs_devices()}
if not verbs["system"] and glob.glob("/sys/class/infiniband/*"):
    try:
        verbs["libionic_install"] = install_libionic()
    except Exception as e:
        verbs["libionic_install"] = {"error": repr(e)}
    verbs["after_install"] = verbs_devices()
    verbs["ibv_devinfo_after"] = sh("ibv_devinfo | head -60")[1]
inv["verbs_setup"] = verbs
log("verbs devices", verbs.get("after_install", verbs["system"]))
devs = rdma_devices()
me = {"role": ROLE, "host": socket.gethostname(), "pod_ip": socket.gethostbyname(socket.gethostname()),
      "devices": devs, "ts": time.time()}
put_json("inventory.json", inv)
put_json("peer.json", me, sub="peers/")
log("devices", [(d["dev"], d["state"], d["rate"], d["roce_v2_gid"]) for d in devs])

# 2. Wait for the peer, which may get its node later.
peer = None
for _ in range(360):
    txt = fetch(f"{RES}/peers/{PEER}/peer.json")
    if txt:
        peer = json.loads(txt); break
    time.sleep(10)
log("peer", peer and (peer["host"], peer["pod_ip"]))

results = {"me": me, "peer": peer, "tests": []}


def record(name, rc, out, **extra):
    row = {"test": name, "rc": rc, **extra, "out": out[-3000:]}
    results["tests"].append(row)
    log(name, {k: v for k, v in row.items() if k != "out"})
    put_json("results.json", results)


def bw_gbits(out):
    # perftest's result row: bytes iterations peak avg msg_rate. Take avg Gb/s.
    rows = [l.split() for l in out.splitlines() if re.match(r"^\s*\d+\s+\d+\s+[\d.]+\s+[\d.]+", l)]
    return float(rows[-1][3]) if rows else None


def lat_us(out):
    rows = [l.split() for l in out.splitlines() if re.match(r"^\s*\d+\s+\d+\s+[\d.]+", l)]
    # bytes iterations t_min t_max t_typical t_avg t_stdev 99% 99.9%
    return {"typical": float(rows[-1][4]), "p99": float(rows[-1][7])} if rows and len(rows[-1]) >= 8 else None


def peer_serve(tool, pairs, args):
    import urllib.request
    q = f"tool={tool}&args={args.replace(' ', '+')}&pairs={','.join(pairs)}"
    urllib.request.urlopen(f"http://{peer['pod_ip']}:{PORT}/serve?{q}", timeout=30).read()


if ROLE == "a" and peer:
    mine = [d for d in devs if d["dev"].startswith("ionic") and d["roce_v2_gid"] is not None and d["state"] and "ACTIVE" in d["state"]]
    theirs = [d for d in peer["devices"] if d["dev"].startswith("ionic") and d["roce_v2_gid"] is not None and d["state"] and "ACTIVE" in d["state"]]
    log("usable pairs", len(mine), len(theirs))

    # TCP over the pod network.
    rc, out = sh(f"iperf3 -c {peer['pod_ip']} -P 16 -t 15 -J", timeout=60)
    try:
        gbps = round(json.loads(out)["end"]["sum_received"]["bits_per_second"] / 1e9, 1)
    except Exception:
        gbps = None
    record("tcp-pod-network-iperf3-P16", rc, out if gbps is None else "", gbits=gbps)

    BW = "-s 4194304 -q 8 -D 10"
    port = BASE_PORT

    def pair_test(name, pairs_mine, pairs_theirs, tool="ib_write_bw", args=BW, parse=bw_gbits):
        global port
        specs, cmds = [], []
        for dm, dt in zip(pairs_mine, pairs_theirs):
            specs.append(f"{dt['dev']}:{dt['roce_v2_gid']}:{port}")
            cmds.append(f"{tool} -d {dm['dev']} -i 1 -x {dm['roce_v2_gid']} -p {port} -F --report_gbits {args} {peer['pod_ip']}")
            port += 1
        try:
            peer_serve(tool, specs, args)
        except Exception as e:
            record(name, 1, f"peer did not start servers: {e!r}"); return
        procs = [subprocess.Popen(["bash", "-c", c], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for c in cmds]
        outs = []
        for p in procs:
            try:
                outs.append(p.communicate(timeout=90)[0])
            except subprocess.TimeoutExpired:
                p.kill(); outs.append("timeout")
        vals = [parse(o) for o in outs]
        rc = 0 if all(v is not None for v in vals) else 1
        extra = {"per_pair": vals, "pairs": [f"{a['dev']}->{b['dev']}" for a, b in zip(pairs_mine, pairs_theirs)]}
        if parse is bw_gbits and rc == 0:
            extra["total_gbits"] = round(sum(vals), 1)
        record(name, rc, "\n----\n".join(outs) if rc else outs[0][-1500:], **extra)

    if mine and theirs:
        n = min(len(mine), len(theirs))
        for i in range(n):
            pair_test(f"rdma-write-bw-pair{i}", [mine[i]], [theirs[i]])
        pair_test("rdma-write-bw-all-pairs", mine[:n], theirs[:n])
        if n > 1:
            pair_test("rdma-write-bw-cross-rail-0to1", [mine[0]], [theirs[1]])
        for size in (65536, 4194304):
            pair_test(f"rdma-write-lat-{size}", [mine[0]], [theirs[0]], tool="ib_write_lat", args=f"-s {size} -n 2000", parse=lat_us)
    else:
        record("rdma", 1, "no ACTIVE ionic device with a routable RoCE v2 GID on one side")
    put_json("DONE", {"ts": time.time()})
elif ROLE == "b":
    subprocess.Popen(["bash", "-c", f"exec iperf3 -s > {OUT}/iperf3-server.log 2>&1"])
    put_json("results.json", results)

# Stay up so the app stays healthy until it is stopped.
while True:
    time.sleep(3600)
'''


def probe_app(role: str, results_remote: str) -> flyte.app.AppEnvironment:
    res = {"amd.com/gpu": str(GPUS_PER_NODE), RDMA_RESOURCE: "1"}
    pod_spec = V1PodSpec(
        containers=[
            V1Container(
                name="primary",
                resources=V1ResourceRequirements(requests=res, limits=res),
                # Registering memory for RDMA locks it in RAM.
                security_context=V1SecurityContext(capabilities=V1Capabilities(add=["IPC_LOCK"])),
                volume_mounts=[
                    V1VolumeMount(name="shm", mount_path="/dev/shm"),
                    V1VolumeMount(name="cache", mount_path=CACHE_DIR),
                ],
            )
        ],
        volumes=[
            V1Volume(name="shm", empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="128Gi")),
            V1Volume(name="cache", empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="1700Gi")),
        ],
        tolerations=[V1Toleration(key="amd.com/gpu", operator="Exists", effect="NoSchedule")],
    )
    return flyte.app.AppEnvironment(
        name=f"kv-fabric-{role}",
        image=image,
        command=["bash", "-c", PRELUDE, "probe", role, results_remote, PROBE],
        port=PORT,
        # Sized like the serving app so follow-up tests can load the model here.
        resources=flyte.Resources(cpu="48", memory="2500Gi", disk="200Gi"),
        env_vars={"SERVE_PORT": str(PORT), "VLLM_ROCM_USE_AITER": "1"},
        secrets=flyte.Secret(key="kimi-vllm-api-key", as_env_var="VLLM_API_KEY"),
        scaling=flyte.app.Scaling(replicas=(1, 1)),
        requires_auth=False,
        cluster_pool=os.getenv("BATCH_CLUSTER_POOL"),
        pod_template=flyte.PodTemplate(
            pod_spec=pod_spec,
            annotations={"serving.knative.dev/progress-deadline": "10800s"},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Object store prefix for the reports")
    parser.add_argument("--roles", default="a,b")
    a = parser.parse_args()

    flyte.init_from_config()
    for role in a.roles.split(","):
        app = flyte.serve(probe_app(role, a.results))
        print(role, app.url if hasattr(app, "url") else app)
