from datetime import datetime, timedelta

import union
from flytekit import CronSchedule

from .apps import deepseek_app

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
    arize_space_id: str, arize_model_id: str, arize_project_name: str
):
    from arize.exporter import ArizeExportClient
    from arize.utils.types import Environments
    from phoenix.evals import OpenAIModel, llm_classify

    client = ArizeExportClient()

    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=5)  # Since cron job runs every 5 minutes

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
    arize_space_id: str, arize_model_id: str, arize_project_name: str
):
    evaluate_rag_arize(arize_space_id, arize_model_id, arize_project_name)


union.LaunchPlan(
    name="arize_online_evaluation_lp",
    workflow=arize_online_evaluation,
    schedule=CronSchedule("*/5 * * * *"),  # Run every 5 minutes
    auto_activate=True,
)


#################################
# PHOENIX ONLINE RAG EVALUATION #
#################################
def lookup_traces(session):
    import pandas as pd

    # Get traces into a dataframe
    spans_df = session.get_spans_dataframe()
    trace_df = session.get_trace_dataset()

    if not trace_df:
        return None, None

    evals = trace_df.evaluations
    evaluation_dfs = []
    for eval in evals:
        eval_dict = eval.__dict__
        eval_df = eval_dict["dataframe"]
        # all dataframes have a tuple index where index[0] is uuid, we'll use this to look for them in spans_df
        evaluation_dfs.append(eval_df)

    if spans_df is None:
        return None

    spans_df["date"] = pd.to_datetime(spans_df["end_time"]).dt.date

    # Get today's date
    today_date = datetime.now().date() + timedelta(days=1)

    # Calculate yesterday's date
    yesterday_date = today_date - timedelta(days=1)

    # Filter for entries from the last day (i.e., yesterday and today)
    selected_date_spans_df = spans_df[
        (spans_df["date"] == today_date) | (spans_df["date"] == yesterday_date)
    ]

    return selected_date_spans_df, evaluation_dfs


@union.task(
    secret_requests=union.Secret(key="phoenix_api_key", env_var="PHOENIX_API_KEY")
)
def evaluate_rag_phoenix():
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
    from phoenix.trace import DocumentEvaluations, SpanEvaluations, TraceDataset

    has_active_session = px.active_session() is not None
    if has_active_session:
        # Used only in a python runtime
        session = px.active_session()
    else:
        # The most common path from clean script run, no session will be live
        try:
            # We need to choose an arbitrary UUID to persist the dataset and reload it
            TRACE_DATA_UUID = "b4165a34-2020-4e9b-98ec-26c5d7e954d4"

            tds = TraceDataset.load(TRACE_DATA_UUID)
            px.launch_app(trace=tds)
            session = px.active_session()
        except Exception:
            tds = None
            px.launch_app()
            session = px.active_session()

    px_client = px.Client(
        endpoint=str(session.url)
    )  # Client based on URL & port of the session

    spans, evaluation_dfs = lookup_traces(
        session=session, selected_date=datetime.now().date()
    )

    if spans is not None:
        with_eval = set()
        for eval_df in evaluation_dfs:
            for index in eval_df.index:
                if isinstance(index, tuple):
                    with_eval.add(index[0])
                else:
                    with_eval.add(index)

        # If a single span in a trace has an evaluation, the entire trace is considered to have an evaluation "eval processed"
        trace_with_evals_id_set = set(
            spans[spans["context.span_id"].isin(with_eval)]["context.trace_id"].unique()
        )
        all_traces_id_set = set(spans["context.trace_id"].unique())

        # Get trace IDs without evaluations
        traces_without_evals_id_set = all_traces_id_set - trace_with_evals_id_set
        spans_without_evals_df = spans[~spans["context.span_id"].isin(with_eval)]

        # Get span IDs without evaluations
        spans_without_evals_id_set = set(
            spans_without_evals_df["context.span_id"].unique()
        )

        queries_df = get_qa_with_reference(px_client)

        # Grab Q&A spans without evaluations
        queries_no_evals = queries_df[queries_df.index.isin(spans_without_evals_id_set)]

        retrieved_documents_df = get_retrieved_documents(px_client)

        # Grab retireved documents without evaluations, based on trace ID
        retrieved_documents_no_evals = retrieved_documents_df[
            retrieved_documents_df["context.trace_id"].isin(traces_without_evals_id_set)
        ]

        eval_model = OpenAIModel(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            api_key="random",
            base_url=deepseek_app.query_endpoint(public=False),
        )  # TODO: Call model app from a task

        hallucination_evaluator = HallucinationEvaluator(eval_model)
        qa_correctness_evaluator = QAEvaluator(eval_model)
        relevance_evaluator = RelevanceEvaluator(eval_model)

        hallucination_eval_df, qa_correctness_eval_df = run_evals(
            dataframe=queries_no_evals,
            evaluators=[hallucination_evaluator, qa_correctness_evaluator],
            provide_explanation=True,
            concurrency=10,
        )
        relevance_eval_df = run_evals(
            dataframe=retrieved_documents_no_evals,
            evaluators=[relevance_evaluator],
            provide_explanation=True,
            concurrency=10,
        )[0]

        px_client.log_evaluations(
            SpanEvaluations(eval_name="Hallucination", dataframe=hallucination_eval_df),
            SpanEvaluations(
                eval_name="QA Correctness", dataframe=qa_correctness_eval_df
            ),
            DocumentEvaluations(eval_name="Relevance", dataframe=relevance_eval_df),
        )

        tds = px_client.get_trace_dataset()
        tds._id = TRACE_DATA_UUID
        tds.save()


@union.workflow
def phoenix_online_evaluation():
    evaluate_rag_phoenix()


union.LaunchPlan(
    name="phoenix_online_evaluation_lp",
    workflow=phoenix_online_evaluation,
    schedule=CronSchedule("*/5 * * * *"),  # Run every 5 minutes
    auto_activate=True,
)
