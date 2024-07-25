from flytekit import workflow, map_task
from datetime import datetime
from tasks.query_calls import query_calls
from tasks.download_call import download_call
from tasks.transcribe import torch_transcribe

# this workflow can be run locally by setting the following secrets:
# export _FSEC_GONG_ACCESS_KEY=<key>
# export _FSEC_GONG_SECRET_KEY=<key>

@workflow
def process_calls(start_date: datetime, end_date: datetime):
    call_id_list = query_calls(start_date=start_date, end_date=end_date)
    call_data_list = map_task(download_call, concurrency=10)(call_id=call_id_list)
    transcriptions = map_task(torch_transcribe, concurrency=10)(audio=call_data_list)
    return transcriptions


