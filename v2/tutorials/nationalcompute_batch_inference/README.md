# Batch inference on National Compute that survives preemption

National Compute sells whole 8-GPU AMD MI355X nodes at a market price. When the
price goes above your limit, the node is taken back, and when it drops you get
one again. This tutorial runs batch inference on that market. The run goes
faster when more nodes are granted, waits when there are none, and resumes
without regenerating finished work.

## How it works

- `batch` runs on a CPU node. It generates coding prompts, splits them into
  shards and fans out.
- `infer_shard` runs on a reusable GPU environment where one replica is one
  full node. Each replica starts a vLLM server once, and its weights stream
  from object storage into GPU memory with the `flyteplugins-vllm` model
  streamer. Replicas scale from one node up to `NC_MAX_NODES` (default 2).
- Shards are cached and retried. Each 128-prompt sub-batch is a `@flyte.trace`,
  so a retried shard replays its finished sub-batches.

gpt-oss-120b runs as eight engines per node. A model that needs the whole node,
such as Kimi K3, runs with `--tensor-parallel 8`.

## Requirements

- A Union cluster on National Compute, reached through a queue. Set
  `NC_QUEUE` to its name.
- A limit price or capacity on the cluster's Capacity page in the National
  Compute console, so GPU pods get a node. A grant is a whole node.
- A registry the National Compute nodes can pull from. If it isn't the
  builder's default registry, set `NC_REGISTRY` (for example
  `ghcr.io/<you>`) and `NC_REGISTRY_SECRET` to an image pull secret the builder
  can push with:

  ```bash
  flyte create secret <name> --type image_pull --from-docker-config --registries ghcr.io --cluster-pool <pool>
  ```

  The cluster pulls the images without credentials, so the packages have to be
  public.

Each script builds its images without a queue first, then submits to the
National Compute queue with the image URIs.

## Run it

Check that a node can be scheduled. It should report 8 GPUs; if it stays in
"waiting for resources", no node is granted.

```bash
uv run gpu_probe.py
```

Prefetch the model into the cluster's object store. The run records a model
artifact whose path is the `--model` argument below.

```bash
uv run prefetch.py openai/gpt-oss-120b
```

Run the batch. 2,000 prompts make 8 shards of 256, and each shard sends its
prompts in traced sub-batches of 128. The driver's report shows shards done,
tokens per second and model load time.

```bash
uv run batch_inference.py --model <model path> --num-prompts 2000
```

## Show an interruption

- Add `--crash-shard 3` to kill a replica halfway through shard 3. The retry
  replays the finished sub-batches and continues.
- Lower the limit price below the market price during a run. Nodes are taken
  back and shards wait in the queue. Raise it again and the run finishes.

Other flags: `--shard-size` sets prompts per shard, `--extra-args "..."` passes
flags to `vllm serve`, and `--max-tokens` sets the completion length.
