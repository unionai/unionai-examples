# Batch inference on a GPU market that survives preemption

Some GPU clouds sell whole 8-GPU AMD MI355X nodes at a market price. When the
price goes above your limit, the node is taken back, and when it drops you get
one again. This tutorial runs batch inference on that market. The run goes
faster when more nodes are granted, waits when there are none, and resumes
without regenerating finished work.

There are two ways to run it, depending on the model:

- A model that fits on one GPU, such as gpt-oss-120b, runs inside the batch
  itself: `batch_inference.py` serves it from a reusable GPU environment, eight
  engines per node.
- Kimi K3 needs a whole node for one copy of the model, so it runs as an app on
  each node (`kimi_app.py`), and the batch runs on CPU tasks that send prompts
  to those apps (`batch_app.py`).

## Requirements

- A Union cluster on the market's nodes. If it is reached through a queue and
  not as the org's default cluster, set `BATCH_QUEUE` to the queue's name. If
  the apps should run in a cluster pool other than the default, set
  `BATCH_CLUSTER_POOL`.
- A limit price or capacity set in the market's console, so GPU pods get a
  node. A grant is a whole node.
- A registry the market's nodes can pull from. If it isn't the
  builder's default registry, set `BATCH_REGISTRY` (for example
  `ghcr.io/<you>`) and `BATCH_REGISTRY_SECRET` to an image pull secret the builder
  can push with:

  ```bash
  flyte create secret <name> --type image_pull --from-docker-config --registries ghcr.io --cluster-pool <pool>
  ```

  The cluster pulls the images without credentials, so the packages have to be
  public.
- For Kimi K3, a secret named `kimi-vllm-api-key` holding the API key the apps
  check requests against.

Check that a node can be scheduled. It should report 8 GPUs; if it stays in
"waiting for resources", no node is granted.

```bash
uv run gpu_probe.py
```

## Run gpt-oss-120b

Prefetch the model into the cluster's object store. The run records a model
artifact whose path is the `--model` argument below.

```bash
uv run prefetch.py openai/gpt-oss-120b
```

Run the batch. 2,000 prompts make 8 shards of 256, and each shard sends its
prompts in traced sub-batches of 128. Replicas scale from one node up to
`BATCH_MAX_NODES` (default 2). The driver's report shows shards done, tokens per
second overall and over the last two minutes, active replicas, retried shards
with the batches they replayed, and model load time.

```bash
uv run batch_inference.py --model <model path> --num-prompts 2000
```

## Run Kimi K3

Prefetch the weights, about 1.5 TB:

```bash
uv run prefetch.py moonshotai/Kimi-K3 --artifact-name kimi-k3
```

Deploy one app per node. Each app copies the weights from object storage into
node memory with parallel reads (about 3 minutes), downloads the kernels that
earlier replicas compiled, and starts vLLM with a 700 GiB CPU KV cache. It is
serving about 7.5 minutes after its container starts, plus the market's wait
for a node. If vLLM crashes, its container restarts with the weights still in
memory, and a watchdog restarts vLLM if it stops producing tokens while
requests are running. The image is `vllm/vllm-openai-rocm:kimi-k31-5c14fe8`;
set `BATCH_VLLM_BASE` to use another build.

```bash
BATCH_APP_NAME=kimi-0 uv run kimi_app.py --model <model path> --extra-args "--max-num-batched-tokens 2048"
BATCH_APP_NAME=kimi-1 uv run kimi_app.py --model <model path> --extra-args "--max-num-batched-tokens 2048"
```

The `--max-num-batched-tokens 2048` works around a crash in the Kimi K3 ROCm
image: under sustained load its MoE kernels fail with
`hipErrorLaunchOutOfResources` on vLLM steps of 7,900 tokens or more. The cap
avoided it through 75 minutes of load, at about 40% of the uncapped
throughput. Leave it out once the image is fixed.

Run the batch against the apps' URLs, which `kimi_app.py` prints:

```bash
uv run batch_app.py --endpoint <app 0 url> <app 1 url> --workload documents --docs 600 --questions 8
```

- The driver waits until every app answers, then splits the prompts into shards
  and sends each shard to one app.
- With `--routing prefix` (the default), prompts that share a prefix, here
  questions about the same document, go to the same node and run back to back,
  so the node computes each document once. `--routing spread` deals prompts out
  evenly instead. On two nodes, prefix routing ran 4,800 prompts 1.58x faster.
- Shards are cached and retried, and each sub-batch of 64 is a trace, so a
  retried shard replays its finished sub-batches. A shard whose app stops
  answering gives up after 2 minutes and retries with growing pauses, which
  covers the market replacing a node without holding a CPU the whole time.

`--workload coding` sends short coding prompts instead of documents.
`routing_bench.py` compares prefix and spread routing on the same apps from a
single task, without shards or traces.

## Show an interruption

- Add `--crash-shard 1` to either batch script to kill shard 1 halfway through
  its first attempt. The retry replays the finished sub-batches and continues.
- Stop a Kimi K3 app in the middle of a run and start it again. Its node goes
  away, the app comes back on a new one, and the shards that were sending to it
  pick up where they stopped.
- Lower the limit price below the market price during a run. Nodes are taken
  back and work waits. Raise it again and the run finishes.

## Other scripts

- `kimi_sweep.py` copies the weights onto one node once, then starts vLLM with
  each of several settings in turn and measures the KV cache under load,
  without redeploying.
- `kimi_stream_app.py` serves Kimi K3 with flyteplugins-vllm's streaming
  loader. With `--node-cache-dir`, the first GPU on the node downloads the
  checkpoint once and every GPU loads from there; see its docstring for the
  read settings and the plugin version it needs.

## Experiments

These measure the cluster rather than run a batch. Each needs two GPU nodes.

- `kv_fabric_probe.py` measures the network between two app pods: TCP over the
  pod network, and RDMA on the Pensando NICs.
- `kv_rdma_probe.py` runs the same tests from Flyte tasks on the host network,
  which RDMA needs on this cluster, plus Mooncake transfers. RDMA reached
  390 Gb/s per NIC, but Mooncake could not register large host-memory buffers
  on these NICs.
- `kv_pool_test.py` runs two Kimi K3 nodes that share one Mooncake KV pool
  over TCP, and measures how much a prefix computed on one node speeds up the
  other (about 2x), with network throughput per phase.
