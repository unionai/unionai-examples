from datetime import datetime, timedelta
from typing import Optional

import union
from flytekit import CronSchedule

from .apps import deepseek_app

CRON_MINUTE = 5

###############################
# ARIZE ONLINE RAG EVALUATION #
###############################
RELEVANCE_EVAL_TEMPLATE = """You are comparing a reference text to a question and trying to determine if the reference text
contains information relevant to answering the question. Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {input}
    ************
    [Reference text]: {reference}
    [END DATA]

Compare the Question above to the Reference text. You must determine whether the Reference text
contains information that can answer the Question. Please focus on whether the very specific
question can be answered by the information in the Reference text.
Your response must be single word, either "relevant" or "unrelated",
and should not contain any text or characters aside from that word.
"unrelated" means that the reference text does not contain an answer to the Question.
"relevant" means the reference text contains an answer to the Question.
"""

CORRECTNESS_EVAL_TEMPLATE = """You are given a question, an answer and reference text. You must determine whether the
given answer correctly answers the question based on the reference text. Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {input}
    ************
    [Reference]: {reference}
    ************
    [Answer]: {output}
    [END DATA]
Your response must be a single word, either "correct" or "incorrect",
and should not contain any text or characters aside from that word.
"correct" means that the question is correctly and fully answered by the answer.
"incorrect" means that the question is not correctly or only partially answered by the answer.
"""

RELEVANCE_RAILS = ["relevant", "unrelated"]
CORRECTNESS_RAILS = ["incorrect", "correct"]


@union.task(secret_requests=union.Secret(key="arize_api_key", env_var="ARIZE_API_KEY"))
def evaluate_rag_arize(
    arize_space_id: str,
    arize_model_id: str,
    arize_project_name: str,
    backfill_from_datetime: Optional[str] = None,
    backfill_to_datetime: Optional[str] = None,
):
    from arize.exporter import ArizeExportClient
    from arize.utils.types import Environments
    from phoenix.evals import OpenAIModel, llm_classify

    client = ArizeExportClient()

    if backfill_from_datetime and backfill_to_datetime:
        start_time = datetime.fromisoformat(backfill_from_datetime)
        end_time = datetime.fromisoformat(backfill_to_datetime)
    else:
        end_time = datetime.now()
        start_time = end_time - timedelta(
            minutes=CRON_MINUTE, seconds=10
        )  # add a few seconds to ensure all spans are captured

    response_df = client.export_model_to_df(
        space_id=arize_space_id,
        model_id=arize_model_id,
        environment=Environments.TRACING,
        start_time=start_time,
        end_time=end_time,
    )

    eval_model = OpenAIModel(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        api_key="random",
        base_url=deepseek_app.query_endpoint(public=False),
    )  # TODO: Call model app from a task

    relevance_eval_df = llm_classify(
        dataframe=response_df,
        template=RELEVANCE_EVAL_TEMPLATE,
        model=eval_model,
        rails=RELEVANCE_RAILS,
        provide_explanation=True,
        include_prompt=True,
        concurrency=4,
    )

    correctness_eval_df = llm_classify(
        dataframe=response_df,
        template=CORRECTNESS_EVAL_TEMPLATE,
        model=eval_model,
        rails=CORRECTNESS_RAILS,
        provide_explanation=True,
        include_prompt=True,
        concurrency=4,
    )

    relevance_eval_df = relevance_eval_df.set_index(response_df["context.span_id"])
    correctness_eval_df = correctness_eval_df.set_index(response_df["context.span_id"])

    client.log_evaluations_sync(relevance_eval_df, project_name=arize_project_name)
    client.log_evaluations_sync(correctness_eval_df, project_name=arize_project_name)


@union.workflow
def arize_online_evaluation(
    arize_space_id: str,
    arize_model_id: str,
    arize_project_name: str,
    backfill_from_datetime: Optional[str] = None,
    backfill_to_datetime: Optional[str] = None,
):
    evaluate_rag_arize(
        arize_space_id,
        arize_model_id,
        arize_project_name,
        backfill_from_datetime,
        backfill_to_datetime,
    )


union.LaunchPlan(
    name="arize_online_evaluation_lp",
    workflow=arize_online_evaluation,
    inputs={
        "arize_space_id": "<YOUR_SPACE_ID>",
        "arize_model_id": "<YOUR_MODEL_ID>",
        "arize_project_name": "arize-rag-evaluation",
    },  # TODO: Input space_id and model_id
    schedule=CronSchedule(f"*/{CRON_MINUTE} * * * *"),
    auto_activate=True,
)


#################################
# PHOENIX ONLINE RAG EVALUATION #
#################################
@union.task(
    secret_requests=union.Secret(key="phoenix_api_key", env_var="PHOENIX_API_KEY")
)
def evaluate_rag_phoenix(
    project_name: str,
    backfill_from_datetime: Optional[str] = None,
    backfill_to_datetime: Optional[str] = None,
):
    import phoenix as px
    from phoenix.evals import (
        HallucinationEvaluator,
        OpenAIModel,
        QAEvaluator,
        RelevanceEvaluator,
        run_evals,
    )
    from phoenix.session.evaluation import (
        get_qa_with_reference,
        get_retrieved_documents,
    )
    from phoenix.trace import DocumentEvaluations, SpanEvaluations

    phoenix_client = px.Client()
    start_time = datetime.now() - timedelta(
        minutes=CRON_MINUTE, seconds=10
    )  # add a few seconds to ensure all spans are captured
    end_time = None

    if backfill_from_datetime and backfill_to_datetime:
        start_time = datetime.fromisoformat(backfill_from_datetime)
        end_time = datetime.fromisoformat(backfill_to_datetime)

    qa_spans_df = get_qa_with_reference(
        phoenix_client,
        start_time=start_time,
        end_time=end_time,
        project_name=project_name,
    )
    retriever_spans_df = get_retrieved_documents(
        phoenix_client,
        start_time=start_time,
        end_time=end_time,
        project_name=project_name,
    )

    eval_model = OpenAIModel(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        api_key="random",
        base_url=deepseek_app.query_endpoint(public=False),
    )  # TODO: Call model app from a task

    hallucination_evaluator = HallucinationEvaluator(eval_model)
    qa_correctness_evaluator = QAEvaluator(eval_model)
    relevance_evaluator = RelevanceEvaluator(eval_model)

    [hallucination_evals_df, qa_correctness_evals_df] = run_evals(
        qa_spans_df,
        [hallucination_evaluator, qa_correctness_evaluator],
    )
    relevance_evals_df = run_evals(
        retriever_spans_df,
        [relevance_evaluator],
    )[0]

    phoenix_client.log_evaluations(
        SpanEvaluations(eval_name="Hallucination", dataframe=hallucination_evals_df),
        SpanEvaluations(eval_name="QA Correctness", dataframe=qa_correctness_evals_df),
        DocumentEvaluations(eval_name="Relevance", dataframe=relevance_evals_df),
    )


@union.workflow
def phoenix_online_evaluation(
    project_name: str,
    backfill_from_datetime: Optional[str] = None,
    backfill_to_datetime: Optional[str] = None,
):
    evaluate_rag_phoenix(project_name, backfill_from_datetime, backfill_to_datetime)


union.LaunchPlan(
    name="phoenix_online_evaluation_lp",
    workflow=phoenix_online_evaluation,
    schedule=CronSchedule(f"*/{CRON_MINUTE} * * * *"),
    inputs={"project_name": "<YOUR_PROJECT_NAME>"},  # TODO: Input project_name
    auto_activate=True,
)
