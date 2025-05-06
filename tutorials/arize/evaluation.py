import os
from datetime import datetime, timedelta
from typing import Optional

import union
from apps import MODEL_ID
from flytekit import CronSchedule
from union.remote import UnionRemote

CRON_MINUTE = 5

###############################
# ARIZE ONLINE RAG EVALUATION #
###############################
RELEVANCE_EVAL_TEMPLATE = """You are comparing a reference text to a question and trying to 
determine if the reference text contains information relevant to answering the question. 
Here is the data:
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

arize_image = union.ImageSpec(
    name="arize-evals",
    packages=[
        "arize-phoenix==8.27.0",
        "fastapi[standard]==0.115.12",
        "litellm==1.68.0",
        "arize==7.40.1",
    ],
    builder="union",
)


@union.task(
    secret_requests=[
        union.Secret(
            key="arize-api-key",
            env_var="ARIZE_API_KEY",
            mount_requirement=union.Secret.MountType.ENV_VAR,
        ),
        union.Secret(
            key="EAGER_API_KEY",
            env_var="UNION_API_KEY",
            mount_requirement=union.Secret.MountType.ENV_VAR,
        ),
    ],
    container_image=arize_image,
)
def evaluate_rag_arize(
    arize_space_id: str,
    arize_model_id: str,
    model_app_name: str,
    backfill_from_datetime: Optional[str] = None,
    backfill_to_datetime: Optional[str] = None,
):
    from arize.exporter import ArizeExportClient
    from arize.pandas.logger import Client
    from arize.utils.types import Environments
    from phoenix.evals import LiteLLMModel, llm_classify

    export_client = ArizeExportClient()

    if backfill_from_datetime and backfill_to_datetime:
        start_time = datetime.fromisoformat(backfill_from_datetime)
        end_time = datetime.fromisoformat(backfill_to_datetime)
    else:
        end_time = datetime.now()
        start_time = end_time - timedelta(
            minutes=CRON_MINUTE, seconds=10
        )  # TODO: add a few seconds to ensure all spans are captured

    response_df = export_client.export_model_to_df(
        space_id=arize_space_id,
        model_id=arize_model_id,
        environment=Environments.TRACING,
        start_time=start_time,
        end_time=end_time,
    )
    response_df["input"] = response_df["attributes.input.value"]
    response_df["output"] = response_df["attributes.output.value"]
    response_df["reference"] = response_df["attributes.retrieval.documents"]

    remote = UnionRemote(
        default_project=union.current_context().execution_id.project,
        default_domain=union.current_context().execution_id.domain,
    )
    app_remote = remote._app_remote
    app_idl = app_remote.get(name=model_app_name)
    url = app_idl.status.ingress.public_url

    os.environ["OPENAI_API_KEY"] = "abc"
    eval_model = LiteLLMModel(
        model=f"openai/{MODEL_ID}",
        model_kwargs={"base_url": f"{url}/v1"},
    )

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

    relevance_eval_df["context.span_id"] = response_df["context.span_id"]
    relevance_eval_df["eval.Relevance.label"] = relevance_eval_df["label"]
    relevance_eval_df["eval.Relevance.explanation"] = relevance_eval_df["explanation"]

    correctness_eval_df["context.span_id"] = response_df["context.span_id"]
    correctness_eval_df["eval.Correctness.label"] = correctness_eval_df["label"]
    correctness_eval_df["eval.Correctness.explanation"] = correctness_eval_df[
        "explanation"
    ]

    arize_client = Client(space_id=arize_space_id, api_key=os.getenv("ARIZE_API_KEY"))
    arize_client.log_evaluations_sync(
        relevance_eval_df,
        project_name=arize_model_id,
        verbose=True,
    )
    arize_client.log_evaluations_sync(
        correctness_eval_df,
        project_name=arize_model_id,
        verbose=True,
    )


@union.workflow
def arize_online_evaluation(
    arize_space_id: str,
    arize_model_id: str = "arize-union",
    model_app_name: str = "vllm-deepseek",
    backfill_from_datetime: Optional[str] = None,
    backfill_to_datetime: Optional[str] = None,
):
    evaluate_rag_arize(
        arize_space_id,
        arize_model_id,
        model_app_name,
        backfill_from_datetime,
        backfill_to_datetime,
    )


union.LaunchPlan.get_or_create(
    name="arize_online_evaluation_lp",
    workflow=arize_online_evaluation,
    default_inputs={"arize_space_id": "<YOUR_SPACE_ID>"},  # TODO: Input space_id
    schedule=CronSchedule(schedule=f"*/{CRON_MINUTE} * * * *"),
    auto_activate=True,
)


#################################
# PHOENIX ONLINE RAG EVALUATION #
#################################
phoenix_image = union.ImageSpec(
    name="phoenix-evals",
    builder="union",
    packages=[
        "arize-phoenix==8.27.0",
        "fastapi[standard]==0.115.12",
        "litellm==1.68.0",
    ],
)


@union.task(
    secret_requests=[
        union.Secret(key="phoenix-api-key", env_var="PHOENIX_API_KEY"),
        union.Secret(key="EAGER_API_KEY", env_var="UNION_API_KEY"),
    ],
    container_image=phoenix_image,
)
def evaluate_rag_phoenix(
    project_name: str,
    phoenix_endpoint: str,
    model_app_name: str,
    backfill_from_datetime: Optional[str] = None,
    backfill_to_datetime: Optional[str] = None,
):
    import phoenix as px
    from phoenix.evals import LiteLLMModel, ToxicityEvaluator, run_evals
    from phoenix.trace import SpanEvaluations
    from phoenix.trace.dsl import SpanQuery

    os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"
    os.environ["PHOENIX_PROJECT_NAME"] = project_name

    phoenix_client = px.Client(endpoint=phoenix_endpoint)
    start_time = datetime.now() - timedelta(
        minutes=CRON_MINUTE, seconds=10
    )  # add a few seconds to ensure all spans are captured
    end_time = None

    if backfill_from_datetime and backfill_to_datetime:
        start_time = datetime.fromisoformat(backfill_from_datetime)
        end_time = datetime.fromisoformat(backfill_to_datetime)

    query = (
        SpanQuery()
        .where(
            "span_kind == 'LLM'",
        )
        .select(
            input="input.value",
            text="output.value",
        )
    )
    qa_spans_df = px.Client().query_spans(
        query, start_time=start_time, end_time=end_time, project_name=project_name
    )

    remote = UnionRemote(
        default_project=union.current_context().execution_id.project,
        default_domain=union.current_context().execution_id.domain,
    )
    app_remote = remote._app_remote
    app_idl = app_remote.get(name=model_app_name)
    url = app_idl.status.ingress.public_url

    os.environ["OPENAI_API_KEY"] = "abc"
    eval_model = LiteLLMModel(
        model=f"openai/{MODEL_ID}",
        model_kwargs={"base_url": f"{url}/v1"},
    )

    toxicity_evaluator = ToxicityEvaluator(eval_model)
    [toxicity_evals_df] = run_evals(qa_spans_df, [toxicity_evaluator], verbose=True)

    phoenix_client.log_evaluations(
        SpanEvaluations(eval_name="Toxicity", dataframe=toxicity_evals_df)
    )


@union.workflow
def phoenix_online_evaluation(
    project_name: str = "phoenix-union",
    endpoint: str = "https://app.phoenix.arize.com",
    model_app_name: str = "vllm-deepseek",
    backfill_from_datetime: Optional[str] = None,
    backfill_to_datetime: Optional[str] = None,
):
    evaluate_rag_phoenix(
        project_name,
        endpoint,
        model_app_name,
        backfill_from_datetime,
        backfill_to_datetime,
    )


union.LaunchPlan.get_or_create(
    name="phoenix_online_evaluation_lp",
    workflow=phoenix_online_evaluation,
    schedule=CronSchedule(schedule=f"*/{CRON_MINUTE} * * * *"),
    auto_activate=True,
)
