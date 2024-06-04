# Batch Inference

Efficiently perform parallel batch inference on Union Cloud with map tasks.

## Setup

```bash
python -m venv ~/venvs/whisper-batch-inference
source ~/venvs/whisper-batch-inference/bin/activate
```

### JAX

```bash
pip install -r requirements.txt
```

### PyTorch

```bash
pip install -r requirements_pytorch.txt
```

## Container Build

### JAX

#### Apple Silicon

```bash
docker buildx build --platform linux/amd64 --cache-to type=local,dest=/tmp/whisper-batch-inference -t <registry>/whisper-jax:0.0.1 --push .
```

#### AMD

```bash
docker build . -t <registry>/whisper-jax:0.0.1
docker push <registry>/whisper-jax:0.0.1
```

### PyTorch

#### Apple Silicon

```bash
docker buildx build --platform linux/amd64 --cache-to type=local,dest=/tmp/whisper-torch -t <registry>/whisper-torch:0.0.1 -f Dockerfile.pytorch --push .
```

#### AMD

```bash
docker build . -f Dockerfile.pytorch -t <registry>/whisper-torch:0.0.1
docker push <registry>/whisper-torch:0.0.1
```

## Run on Flyte

Install [flytectl](https://docs.flyte.org/projects/flytectl/en/latest/).

### PyTorch Simple

```bash
export FLYTECTL_CONFIG=...
pyflyte register --image <registry>/whisper-torch:0.0.1 whisper/workflows/inference_hf_pytorch_simple.py
```

### JAX simple

```bash
export FLYTECTL_CONFIG=...
pyflyte register --image <registry>/whisper-jax:0.0.1 whisper/workflows/inference_jax_simple.py
```

### JAX batch inference

```bash
export FLYTECTL_CONFIG=...
pyflyte register --image <registry>/whisper-jax:0.0.1 whisper
```
