################################################
# Source: https://github.com/bentoml/llm-bench #
################################################
import argparse
import functools
import os

from transformers import AutoTokenizer

from userdef import UserDef as BaseUserDef
from huggingface_hub import login

max_tokens = 512

login(token="<HF_TOKEN>")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

base_url = os.environ.get("BASE_URL", "http://localhost:8000")


@functools.lru_cache(maxsize=8)
def get_prompt_set(min_input_length=0, max_input_length=500):
    """
    return a list of prompts with length between min_input_length and max_input_length
    """
    import json
    import requests
    import os

    # check if the dataset is cached
    if os.path.exists("databricks-dolly-15k.jsonl"):
        print("Loading cached dataset")
        with open("databricks-dolly-15k.jsonl", "r") as f:
            dataset = [json.loads(line) for line in f.readlines()]
    else:
        print("Downloading dataset")
        raw_dataset = requests.get(
            "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
        )
        content = raw_dataset.content
        open("databricks-dolly-15k.jsonl", "wb").write(content)
        dataset = [json.loads(line) for line in content.decode().split("\n")]
        print("Dataset downloaded")

    for d in dataset:
        d["question"] = d["context"] + d["instruction"]
        d["input_tokens"] = len(tokenizer(d["question"])["input_ids"])
        d["output_tokens"] = len(tokenizer(d["response"]))
    return [
        d["question"]
        for d in dataset
        if min_input_length <= d["input_tokens"] <= max_input_length
    ]


prompts = get_prompt_set(30, 150)


class UserDef(BaseUserDef):
    BASE_URL = base_url
    PROMPTS = prompts

    @classmethod
    def make_request(cls):
        import json
        import random

        prompt = random.choice(cls.PROMPTS)

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        url = f"{cls.BASE_URL}/completions"
        data = {
            "model": "meta/llama3-8b-instruct",
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        return url, headers, json.dumps(data)

    @staticmethod
    def parse_response(chunk: bytes):
        text = chunk.decode("utf-8").strip()
        return tokenizer.encode(text, add_special_tokens=False)


if __name__ == "__main__":
    import asyncio
    from common import start_benchmark_session

    # arg parsing
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument("--max_users", type=int, required=True)
    parser.add_argument("--session_time", type=float, default=None)
    parser.add_argument("--ping_correction", action="store_true")
    args = parser.parse_args()

    asyncio.run(start_benchmark_session(args, UserDef))
