# Union Examples

This is a repository of runnable examples for [Union](https://docs.union.ai).
Use it as a reference for learning how to build repeatable and scalable AI/ML
workflows.

## Usage

### Flyte 1

Install the `union` SDK:

```shell
pip install union
```

This should also install `flytekit` for Flyte v1. Then run an example in the
`tutorials` directory, for instance:

```shell
union run --remote tutorials/sentiment_classifier/sentiment_classifier.py main
```

### Flyte 2

Install the `flyte` SDK:

```shell
pip install --pre flyte
```

Create a config file in the root directory:

```shell
flyte create config \
--output ~/.flyte/config.yaml \
--endpoint demo.hosted.unionai.cloud \
--domain flytesnacks \
--project development \
--builder remote
```

Swap out `demo.hosted.unionai.cloud` for your endpoint of choice.
Then run an example in the `integrations-v2` directory, for instance:

```shell
uv run integrations-v2/flyte-plugins/openai/openai/agents_tools.py main
```
