# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b6",
#    "datasets==4.0.0",
#    "litellm==1.75.0",
# ]
# ///

import asyncio
import html
import re
from dataclasses import dataclass
from typing import AsyncIterator, Union

import flyte
import flyte.report
import pandas as pd

env = flyte.TaskEnvironment(
    name="prompt-sweep",
    image=flyte.Image.from_uv_script(__file__, name="prompt-sweep", pre=True),
    # TODO: replace with your OpenAI API key
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    resources=flyte.Resources(cpu=1),
)


@env.task
async def data_prep(
    dataset_name: str,
    dataset_split: str = "geometric_shapes",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_split)["test"].shuffle(seed=1234)

    def make_df(slice_):
        return pd.DataFrame({"question": slice_["input"], "answer": slice_["target"]})

    df_train = make_df(dataset.select(range(100)))
    df_test = make_df(dataset.select(range(100, 200)))

    return df_train, df_test


@flyte.trace
async def generate_target_model_response(
    model_name: str, prompt: str, question: str
) -> AsyncIterator[str]:
    from litellm import acompletion

    stream = await acompletion(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        temperature=0,
        timeout=600,
        max_tokens=1000,
        stream=True,
    )

    async for chunk in stream:
        content = chunk.choices[0].delta.get("content", "")
        if content:
            yield content


@flyte.trace
async def generate_review_model_response(model_name: str, prompt: str) -> str:
    from litellm import acompletion

    response = await acompletion(
        model=model_name,
        messages=[{"role": "system", "content": prompt}],
        temperature=0,
        timeout=600,
        max_tokens=10,
    )
    return response.choices[0].message["content"].strip().lower()


async def generate_and_review(
    index: int,
    question: str,
    answer: str,
    prompt: str,
    model_name: str,
    review_model: str,
) -> dict:
    response = "".join(
        [
            chunk
            async for chunk in generate_target_model_response(
                model_name, prompt, question
            )
        ]
    )

    review_prompt = f"""You are a review model tasked with evaluating the correctness of a response to a navigation problem.
The response may contain detailed steps and explanations, but the final answer is the key point.
Please determine if the final answer provided in the response is correct based on the ground truth number.
Respond with 'True' if the final answer is correct and 'False' if it is not.
Only respond with 'True' or 'False', nothing else.

Model Response:
{response}

Ground Truth:
{answer}
"""

    verdict = await generate_review_model_response(review_model, review_prompt)
    verdict_clean = verdict.strip().lower()

    if verdict_clean not in {"true", "false"}:
        verdict_clean = "not sure"

    return {
        "index": index,
        "model_response": response,
        "is_correct": verdict_clean == "true",
    }


async def run_grouped_task(
    i, index, question, answer, prompt, model_name, review_model, semaphore
):
    async with semaphore:
        with flyte.group(name=f"row-{i}"):
            return await generate_and_review(
                index=index,
                question=question,
                answer=answer,
                prompt=prompt,
                model_name=model_name,
                review_model=review_model,
            )


@env.task(report=True)
async def evaluate_prompt(
    df: pd.DataFrame, prompt: str, model_name: str, review_model: str, concurrency: int
) -> float:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []

    for i, row in enumerate(df.itertuples(index=True)):
        tasks.append(
            run_grouped_task(
                i=i,
                index=row.Index,
                question=row.question,
                answer=row.answer,
                prompt=prompt,
                model_name=model_name,
                review_model=review_model,
                semaphore=semaphore,
            )
        )

    results = await asyncio.gather(*tasks)

    await flyte.report.log.aio("<table border='1' cellspacing='0' cellpadding='4'>")
    await flyte.report.log.aio(
        "<thead><tr><th>Question</th><th>Answer</th><th>Model Response</th><th>Correct</th></tr></thead><tbody>"
    )

    for result in results:
        idx = result["index"]
        df.at[idx, "model_response"] = result["model_response"]
        df.at[idx, "is_correct"] = result["is_correct"]

        await flyte.report.log.aio(
            f"""
            <tr>
              <td>{html.escape(df.at[idx, 'question'])}</td>
              <td>{html.escape(df.at[idx, 'answer'])}</td>
              <td>{result['model_response']}</td>
              <td>{result['is_correct']}</td>
            </tr>"""
        )

    await flyte.report.log.aio("</tbody></table>", do_flush=True)

    accuracy = df["is_correct"].mean()
    return accuracy.item()


@flyte.trace
async def generate_optimizer_model_response(prompt: str, model: str) -> str:
    from litellm import acompletion

    response = await acompletion(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
        timeout=600,
    )
    return response.choices[0].message["content"].strip()


@dataclass
class PromptResult:
    prompt: str
    accuracy: float


@env.task(report=True)
async def prompt_optimizer(
    df_train: pd.DataFrame,
    starting_prompt: str,
    target_model: str,
    review_model: str,
    generation_model: str,
    max_iterations: int,
    concurrency: int,
) -> tuple[str, float]:
    prompt_accuracies: list[PromptResult] = []

    # Step 1: Evaluate the starting prompt first
    with flyte.group(name="baseline_evaluation"):
        starting_accuracy = await evaluate_prompt(
            df_train, starting_prompt, target_model, review_model, concurrency
        )
        prompt_accuracies.append(
            PromptResult(prompt=starting_prompt, accuracy=starting_accuracy)
        )

    # Step 2: Optimize with generated prompts
    while len(prompt_accuracies) <= max_iterations:
        with flyte.group(name=f"prompt_optimization_step_{len(prompt_accuracies)}"):
            prompt_scores_str = "\n".join(
                f"{result.prompt}: {result.accuracy:.2f}"
                for result in sorted(prompt_accuracies, key=lambda x: x.accuracy)
            )
            metaprompt = f"""
<EXPLANATION>
I have some prompts along with their corresponding accuracies.
The prompts are arranged in ascending order based on their accuracy, where higher accuracy indicate better quality.
</EXPLANATION>

<PROMPTS>
{prompt_scores_str}
</PROMPTS>

Each prompt was used together with a problem statement around geometric shapes.

<EXAMPLE>
<QUESTION>
This SVG path element <path d="M 55.57,80.69 L 57.38,65.80 M 57.38,65.80 L 48.90,57.46 M 48.90,57.46 L 45.58,47.78 M 45.58,47.78 L 53.25,36.07 L 66.29,48.90 L 78.69,61.09 L 55.57,80.69"/> draws a Options: (A) circle (B) heptagon (C) hexagon (D) kite (E) line (F) octagon (G) pentagon (H) rectangle (I) sector (J) triangle
</QUESTION>
<ANSWER>
(B)
</ANSWER>
</EXAMPLE>

<TASK>
Write a new prompt that will achieve an accuracy as high as possible and that is different from the old ones.
</TASK>


<RULES>
- It is very important that the new prompt is distinct from ALL the old ones!
- Ensure that you analyse the prompts with a high accuracy and reuse the patterns that worked in the past
- Ensure that you analyse the prompts with a low accuracy and avoid the patterns that didn't worked in the past
- Think out loud before creating the prompt. Describe what has worked in the past and what hasn't. Only then create the new prompt.
- Use all available information like prompt length, formal/informal use of language, etc for your analysis.
- Be creative, try out different ways of prompting the model. You may even come up with hypothetical scenarios that might improve the accuracy.
- You are generating system prompts. This means that there should be no placeholders in the prompt, as they cannot be filled at runtime. Instead focus on general instructions that will help the model to solve the task.
- Write your new prompt in double square brackets. Use only plain text for the prompt text and do not add any markdown (i.e. no hashtags, backticks, quotes, etc).
</RULES>
"""

            response = await generate_optimizer_model_response(
                metaprompt, generation_model
            )
            match = re.search(r"\[\[(.*?)\]\]", response, re.DOTALL)
            if not match:
                print("No new prompt found. Skipping.")
                continue

            new_prompt = match.group(1)
            accuracy = await evaluate_prompt(
                df_train, new_prompt, target_model, review_model, concurrency
            )

            prompt_accuracies.append(PromptResult(prompt=new_prompt, accuracy=accuracy))

    best_accuracy = max(prompt_accuracies, key=lambda x: x.accuracy).accuracy
    best_prompt = max(prompt_accuracies, key=lambda x: x.accuracy).prompt

    await flyte.report.log.aio("<table border='1' cellspacing='0' cellpadding='4'>")
    await flyte.report.log.aio(
        "<thead><tr><th>Prompt</th><th>Accuracy</th></tr></thead>"
    )
    await flyte.report.log.aio("<tbody>")
    for prompt_result in sorted(prompt_accuracies, key=lambda x: x.accuracy):
        await flyte.report.log.aio(
            f"""
            <tr>
            <td>{prompt_result.prompt}</td>
            <td>{prompt_result.accuracy:.2f}</td>
            </tr>"""
        )
    await flyte.report.log.aio("</tbody></table>")

    improvement = best_accuracy - starting_accuracy

    await flyte.report.log.aio(f"Best prompt: {best_prompt}")
    await flyte.report.log.aio(f"Best accuracy: {best_accuracy:.2f}")
    await flyte.report.log.aio(f"Improvement: {improvement:.2f}", do_flush=True)

    return best_prompt, best_accuracy


@env.task
async def prompt_sweep(
    dataset_name: str = "lukaemon/bbh",
    dataset_split: str = "geometric_shapes",
    starting_prompt: str = "Solve the given problem about geometric shapes. Think step by step.",
    target_model: str = "gpt-4.1-nano",
    review_model: str = "gpt-4.1-nano",
    generation_model: str = "gpt-4.1",
    max_iterations: int = 3,
    concurrency: int = 10,
) -> dict[str, Union[str, float]]:
    df_train, df_test = await data_prep(dataset_name, dataset_split)

    best_prompt, training_accuracy = await prompt_optimizer(
        df_train,
        starting_prompt,
        target_model,
        review_model,
        generation_model,
        max_iterations,
        concurrency,
    )

    with flyte.group(name="test_data_evaluation"):
        baseline_test_accuracy = await evaluate_prompt(
            df_test, starting_prompt, target_model, review_model, concurrency
        )

        test_accuracy = await evaluate_prompt(
            df_test, best_prompt, target_model, review_model, concurrency
        )

    return {
        "best_prompt": best_prompt,
        "training_accuracy": training_accuracy,
        "baseline_test_accuracy": baseline_test_accuracy,
        "test_accuracy": test_accuracy,
    }


if __name__ == "__main__":
    flyte.init_from_config("config.yaml")
    run = flyte.run(prompt_sweep)
    print(run.url)
