from flytekit import workflow, map_task, LaunchPlan, CronSchedule
from flytekit.core.artifact import Artifact
from typing_extensions import Annotated
from typing import List
from datetime import datetime

from tasks.download_call import download_call
from tasks.query_calls import query_calls
from tasks.query_calls import CallData
from tasks.transcribe import torch_transcribe
from tasks.generate_corpus import generate_corpus
from tasks.generate_corpus import call_data_corpus

# this workflow can be run locally by setting the following secrets:
# export _FSEC_GONG_ACCESS_KEY=<key>
# export _FSEC_GONG_SECRET_KEY=<key>


@workflow
def process_calls(
        start_date: datetime,
        end_date: datetime,
        prev_data: List[CallData]
):
    call_id_list = query_calls(start_date=start_date, end_date=end_date, prev_data=prev_data)
    call_data_list = map_task(download_call, concurrency=2)(call_data=call_id_list)
    transcriptions = map_task(torch_transcribe, concurrency=10)(audio=call_data_list)
    generate_corpus(prev_data=prev_data, new_data=transcriptions)


process_calls_lp = LaunchPlan.get_or_create(
    name="process_calls_lp",
    workflow=process_calls,
    default_inputs={
        "start_date": datetime(2020, 1, 1),
        "end_date": datetime.today(),
        "prev_data": call_data_corpus.query()
        # "prev_data": []
    },
    schedule=CronSchedule(
        schedule="0 7 * * *",
    ),
)


