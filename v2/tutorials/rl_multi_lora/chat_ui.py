"""
Side-by-side chat: the same prompt streams to two models at once, each answer gets a ✅/❌ badge.

Deployed by ``serve.py``. To run it locally against deployed (or any OpenAI-compatible) endpoints:

    FLYTE_API_KEY=... python chat_ui.py \\
        --endpoint base=https://<base-app-url> --endpoint rl-countdown=https://<rl-app-url>

gradio and openai are imported lazily so ``serve.py`` can import this module without them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from typing import Any

import tasks

TASK_LABELS = {"Countdown": "countdown", "JSON extraction": "json"}
PLACEHOLDERS = {
    "countdown": "Numbers: 7, 3, 9, 2\nTarget: 24",
    "json": "Paste a messy customer email here, or click 'Random problem'.",
}


def _badge(passed: bool | None, reason: str) -> str:
    icon = {True: "✅", False: "❌", None: "ℹ️"}[passed]
    return f"### {icon} {reason}"


def _display(text: str) -> str:
    """Show <think>/<answer> tags as markdown; the Chatbot would otherwise sanitize them away."""
    # A code block, not italics: the reasoning is full of '*' from arithmetic.
    text = re.sub(r"<think>\s*", "💭 **Thinking**\n```text\n", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*</think>\s*", "\n```\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<answer>\s*", "**Answer:** `", text, flags=re.IGNORECASE)
    return re.sub(r"\s*</answer>", "`", text, flags=re.IGNORECASE)


TRAINING_SAMPLES = 4  # shown per page; a training iteration uses PROMPTS_PER_ITER (16) problems


def training_samples_md(task: str, iteration: int) -> str:
    """The first problems of one training iteration's batch, exactly as train.py generates them."""
    parts = [f"#### Iteration {iteration}: first {TRAINING_SAMPLES} problems of the batch"]
    for i, (q, a) in enumerate(tasks.train_batch(task, iteration, TRAINING_SAMPLES), start=1):
        truth = json.loads(a)
        parts.append(f"**Problem {i}**: what the model sees\n```text\n{q}\n```")
        if task == "countdown":
            solution = tasks.solve_countdown(truth["numbers"], truth["target"])
            parts.append(f"What the checker accepts (one valid solution, never shown to the model): `{solution}`")
        else:
            parts.append(
                f"Ground truth the reward compares against (never shown to the model):\n"
                f"```json\n{json.dumps(truth, indent=2)}\n```"
            )
    return "\n\n".join(parts)


def build_demo(endpoints: dict[str, str], api_key: str | None = None) -> Any:
    """Build the Gradio app. ``endpoints`` maps vLLM model id (served model name) -> base URL."""
    import gradio as gr
    from openai import AsyncOpenAI

    api_key = api_key or os.getenv("FLYTE_API_KEY") or "unused"
    clients = {m: AsyncOpenAI(base_url=f"{url}/v1", api_key=api_key, timeout=120) for m, url in endpoints.items()}
    models = list(endpoints)
    # Ground truth for every generated/preset problem shown in this session, keyed by the exact text.
    known: dict[str, str] = {q: a for t in tasks.TASKS for q, a in tasks.PRESETS[t]}

    def default_pair(task: str) -> tuple[str, str]:
        left = "base" if "base" in models else models[0]
        right = f"rl-{task}" if f"rl-{task}" in models else models[-1]
        return left, right

    async def stream(model: str, task: str, user: str, temperature: float, queue: asyncio.Queue, side: int) -> None:
        try:
            response = await clients[model].chat.completions.create(
                model=model,
                messages=tasks.messages(task, user),  # one-shot, exactly as in training
                temperature=temperature,
                max_tokens=1024,
                stream=True,
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    await queue.put((side, chunk.choices[0].delta.content))
        except Exception as e:  # noqa: BLE001 - surface any endpoint failure in the chat column
            await queue.put((side, f"\n\n⚠️ {type(e).__name__}: {e}"))
        finally:
            await queue.put((side, None))

    async def send(task_label: str, left: str, right: str, user: str, temperature: float):
        task = TASK_LABELS[task_label]
        user = user.strip()
        if not user:
            return
        texts = ["", ""]
        histories = [[{"role": "user", "content": user}, {"role": "assistant", "content": ""}] for _ in range(2)]
        badges = ["### ⏳ generating…", "### ⏳ generating…"]
        yield *histories, *badges

        queue: asyncio.Queue = asyncio.Queue()
        workers = [
            asyncio.create_task(stream(model, task, user, temperature, queue, side))
            for side, model in enumerate((left, right))
        ]
        remaining = 2
        while remaining:
            side, delta = await queue.get()
            if delta is None:
                remaining -= 1
                badges[side] = _badge(*tasks.verdict(task, user, texts[side], known.get(user)))
            else:
                texts[side] += delta
                histories[side][1]["content"] = _display(texts[side])
            yield *histories, *badges
        await asyncio.gather(*workers)

    def random_problem(task_label: str) -> str:
        task = TASK_LABELS[task_label]
        q, a = tasks.make_problem(task, random.Random())
        known[q] = a
        return q

    def preset(task_label: str, index: int) -> str:
        return tasks.PRESETS[TASK_LABELS[task_label]][index][0]

    def reference(task_label: str, user: str) -> str:
        task = TASK_LABELS[task_label]
        if task == "countdown":
            parsed = tasks.parse_countdown_message(user)
            if parsed is None:
                return ""
            solution = tasks.solve_countdown(*parsed)
            return f"`{solution} = {parsed[1]}`" if solution else "No solution exists for this puzzle."
        answer = known.get(user.strip())
        return (
            f"```json\n{json.dumps(json.loads(answer), indent=2)}\n```"
            if answer
            else "No ground truth for custom text."
        )

    def switch_task(task_label: str):
        task = TASK_LABELS[task_label]
        left, right = default_pair(task)
        return (
            gr.update(value=left),
            gr.update(value=right),
            gr.update(value="", placeholder=PLACEHOLDERS[task], lines=3 if task == "countdown" else 10),
            tasks.SYSTEM_PROMPTS[task],
            [],
            [],
            "",
            "",
        )

    async def warm_up() -> str:
        """Prefill every preset on every model once, so prefix caches and CUDA graphs are hot."""
        jobs = [
            clients[m].chat.completions.create(model=m, messages=tasks.messages(t, q), max_tokens=1)
            for m in models
            for t in tasks.TASKS
            for q, _ in tasks.PRESETS[t]
        ]
        results = await asyncio.gather(*jobs, return_exceptions=True)
        failed = [r for r in results if isinstance(r, BaseException)]
        return f"Warmed {len(results) - len(failed)}/{len(results)} requests" + (f" ({failed[0]})" if failed else "")

    left0, right0 = default_pair("countdown")
    with gr.Blocks(title="Before vs. after RL") as demo:
        gr.Markdown("# Before vs. after RL")
        with gr.Tab("Chat"):
            gr.Markdown(
                "The same prompt goes to both models. **Left:** the base model. **Right:** the same model "
                "after a short GRPO run on this skill."
            )
            task_label = gr.Radio(list(TASK_LABELS), value="Countdown", label="Skill")
            with gr.Accordion("System prompt (sent to both models, same as in training)", open=True):
                chat_system = gr.Textbox(tasks.COUNTDOWN_SYSTEM, show_label=False, interactive=False, max_lines=12)
            with gr.Row():
                with gr.Column():
                    left_model = gr.Dropdown(models, value=left0, label="Model", filterable=False)
                    left_chat = gr.Chatbot(height=460, show_label=False)
                    left_badge = gr.Markdown()
                with gr.Column():
                    right_model = gr.Dropdown(models, value=right0, label="Model", filterable=False)
                    right_chat = gr.Chatbot(height=460, show_label=False)
                    right_badge = gr.Markdown()
            user = gr.Textbox(label="Prompt", placeholder=PLACEHOLDERS["countdown"], lines=3)
            with gr.Row():
                send_btn = gr.Button("Send to both", variant="primary")
                random_btn = gr.Button("🎲 Random problem")
                preset_btns = [gr.Button(f"Preset {i + 1}", size="sm") for i in range(len(tasks.PRESETS["countdown"]))]
            with gr.Accordion("Reference answer", open=False):
                ref = gr.Markdown()
            with gr.Accordion("Settings", open=False):
                temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature (0 = deterministic)")
                warm_btn = gr.Button("Warm up all models")
                warm_status = gr.Markdown()

        with gr.Tab("Training data"):
            gr.Markdown(
                "RL needs no labeled answers, just problems and a checker. These are the exact problems "
                "`train.py` samples at each iteration (fresh ones every iteration; the held-out eval set "
                "is excluded)."
            )
            with gr.Row():
                data_task = gr.Radio(list(TASK_LABELS), value="Countdown", label="Skill")
                data_iter = gr.Number(value=0, precision=0, minimum=0, label="Training iteration")
                next_btn = gr.Button("Next iteration →")
            with gr.Accordion("System prompt", open=False):
                data_system = gr.Textbox(tasks.COUNTDOWN_SYSTEM, show_label=False, interactive=False, max_lines=12)
            data_reward = gr.Markdown(f"**Reward:** {tasks.REWARD_RULES['countdown']}")
            data_samples = gr.Markdown(training_samples_md("countdown", 0))

        outputs = [left_chat, right_chat, left_badge, right_badge]
        inputs = [task_label, left_model, right_model, user, temperature]
        send_btn.click(send, inputs, outputs)
        user.submit(send, inputs, outputs)
        random_btn.click(random_problem, task_label, user)
        for i, btn in enumerate(preset_btns):
            btn.click(lambda label, i=i: preset(label, i), task_label, user)
        user.change(reference, [task_label, user], ref)
        task_label.change(
            switch_task,
            task_label,
            [left_model, right_model, user, chat_system, left_chat, right_chat, left_badge, right_badge],
        )
        warm_btn.click(warm_up, None, warm_status)

        def show_data(task_label: str, iteration: float | None):
            task = TASK_LABELS[task_label]
            return (
                tasks.SYSTEM_PROMPTS[task],
                f"**Reward:** {tasks.REWARD_RULES[task]}",
                training_samples_md(task, int(iteration or 0)),
            )

        data_outputs = [data_system, data_reward, data_samples]
        data_task.change(show_data, [data_task, data_iter], data_outputs)
        data_iter.change(show_data, [data_task, data_iter], data_outputs)
        next_btn.click(lambda i: int(i or 0) + 1, data_iter, data_iter)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--endpoint", action="append", required=True, metavar="MODEL=URL", help="vLLM model id and base URL"
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    endpoints = dict(e.split("=", 1) for e in args.endpoint)
    build_demo({m: u.rstrip("/") for m, u in endpoints.items()}).queue().launch(server_port=args.port)
