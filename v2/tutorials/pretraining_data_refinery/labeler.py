"""The LLM labeler: the FineWeb-Edu prompt, served by vLLM on the task's GPU.

The server is a subprocess of the reusable GPU container. The first labeling call on a replica
starts it and waits for the model to load; every later call on that replica finds it running.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import time
from dataclasses import dataclass

# Bump when the prompt or the decoding settings change (see classifier.VERSION).
VERSION = "fineweb-edu-prompt/1"
PORT = 8000
MAX_DOC_CHARS = 6000  # about 1,500 tokens of the document

# {{docs-fragment prompt}}
# The FineWeb-Edu annotation prompt (Penedo et al. 2024, and the
# HuggingFaceFW/fineweb-edu-classifier model card). Llama-3-70B-Instruct answered it for
# 450,000 FineWeb pages; here Qwen2.5-14B-Instruct answers it for a few thousand.
PROMPT = """Below is an extract from a web page. Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.
- Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.
- Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students.
- Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren't too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.
- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.

The extract:
{extract}.

After examining the extract:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Educational score:  <total points>\""""

SCORE = re.compile(r"Educational score:\s*\**\s*(\d)")


def parse_score(answer: str) -> int | None:
    """The last "Educational score: N" in the answer, or None if the model did not give one."""
    found = SCORE.findall(answer)
    return min(int(found[-1]), 5) if found else None
# {{/docs-fragment prompt}}


@dataclass
class Server:
    model: str
    process: subprocess.Popen
    startup_s: float
    log_path: str


_server: Server | None = None


def tail(path: str, lines: int = 30) -> str:
    """The error lines from the vLLM log, then its last lines."""
    with open(path, errors="replace") as f:
        log = f.readlines()
    errors = [ln for ln in log if "Error" in ln or "error:" in ln][-20:]
    return "".join(errors) + "...\n" + "".join(log[-lines:])


# {{docs-fragment server}}
def get_server(model: str, timeout_s: int = 1200) -> tuple[Server, bool]:
    """Start vLLM once per container. Returns the server and whether this call started it."""
    global _server
    import httpx

    if _server is not None and _server.process.poll() is None:
        if _server.model == model:
            return _server, False
        _server.process.terminate()  # a different labeler was asked for: swap models
        _server.process.wait(timeout=120)
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model,
           "--port", str(PORT), "--max-model-len", "8192", "--gpu-memory-utilization", "0.90",
           "--max-num-seqs", "256"]
    log_path = "/tmp/vllm.log"
    started = time.monotonic()
    process = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    while time.monotonic() - started < timeout_s:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {process.returncode}:\n{tail(log_path)}")
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
                _server = Server(model, process, time.monotonic() - started, log_path)
                print(f"vLLM ready in {_server.startup_s:.0f}s", flush=True)
                return _server, True
        except httpx.HTTPError:
            pass
        time.sleep(3)
    process.kill()
    raise TimeoutError(f"vLLM did not start in {timeout_s}s:\n{tail(log_path)}")
# {{/docs-fragment server}}


# {{docs-fragment label}}
async def label_all(texts: list[str], model: str, concurrency: int = 256) -> list[dict]:
    """Send every document to the server at once (bounded), greedy decoding, and parse the score."""
    import httpx

    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}", timeout=600) as client:

        async def one(text: str) -> dict:
            body = {
                "model": model,
                "messages": [{"role": "user", "content": PROMPT.format(extract=text[:MAX_DOC_CHARS])}],
                "temperature": 0.0,
                "max_tokens": 300,
            }
            async with sem:
                r = await client.post("/v1/chat/completions", json=body)
            r.raise_for_status()
            out = r.json()
            answer = out["choices"][0]["message"]["content"]
            return {
                "llm_score": parse_score(answer),
                "justification": answer.strip(),
                "prompt_tokens": out["usage"]["prompt_tokens"],
                "completion_tokens": out["usage"]["completion_tokens"],
            }

        return await asyncio.gather(*[one(t) for t in texts])
# {{/docs-fragment label}}
