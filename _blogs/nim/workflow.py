from flytekit import workflow

from lora import update_lora
from orpo_finetune import FineTuningArgs, llama_8b_instruct_finetune
from serve import model_serving

from constants import HF_REPO_ID


@workflow
def nim_workflow(
    args: FineTuningArgs = FineTuningArgs(new_model=HF_REPO_ID),
) -> list[str]:
    repo_id = llama_8b_instruct_finetune(args=args)

    t1 = update_lora(repo_id=repo_id)
    t2 = model_serving(
        questions=[
            "Explain the central dogma of molecular biology and its significance in understanding the flow of genetic information.",
            "Derive the formula for the volume of a cone using integral calculus.",
            "Analyze the symbolism and literary devices used in William Shakespeare's play 'Hamlet.'",
            "Can you write a short story about a time-traveling detective?",
            "Discuss the ethical considerations surrounding the use of artificial intelligence in healthcare decision-making.",
            "Explain the difference between SQL and NoSQL databases, their respective advantages and use cases.",
            "Analyze the impact of globalization on international trade and economic development.",
            "Describe the process of photosynthesis and its importance in sustaining life on Earth.",
            "Discuss the historical significance of the French Revolution and its impact on modern political ideologies.",
            "Write a function in Python that takes a list of numbers and returns the second-largest number in the list.",
        ],
        repo_id=repo_id,
    )

    t1 >> t2
    return t2
