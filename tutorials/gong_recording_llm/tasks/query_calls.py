from flytekit import task, ImageSpec, Secret, current_context

from datetime import datetime, time
import pytz
import base64
import requests


query_calls_img = ImageSpec(
    packages=[
        "flytekit==1.13.0",
        "union==0.1.48"
    ],
)

# remember to define secrets using union secrets
# and show how to define secrets locally using _FSEC_<SECRET_KEY> env var

@task(
    container_image=query_calls_img,
    secret_requests=[Secret(key="gong_access_key"), Secret(key="gong_secret_key")],
    cache=True,
    cache_version="1.0",
)
def query_calls(start_date: datetime, end_date: datetime) -> list[str]:
    pacific_tz = pytz.timezone('US/Pacific')

    start_time_naive = datetime.combine(start_date, time(0, 0, 0))
    start_time_aware = pacific_tz.localize(start_time_naive)

    end_time_naive = datetime.combine(end_date, time(23, 59, 59, 999999))
    end_time_aware = pacific_tz.localize(end_time_naive)

    start_time_iso = start_time_aware.isoformat()
    end_time_iso = end_time_aware.isoformat()

    url = "https://api.gong.io/v2/calls/"

    access_key = current_context().secrets.get(key="gong_access_key")
    secret_key = current_context().secrets.get(key="gong_secret_key")
    basic_token = base64.b64encode(f"{access_key}:{secret_key}".encode()).decode()

    headers = {
        "Authorization": f"Basic {basic_token}",
        "Content-Type": "application/json"
    }

    params = {
        "fromDateTime": start_time_iso,
        "toDateTime": end_time_iso
    }

    cursor = None
    all_calls = []

    while True:
        data = run_request(url, headers, params, cursor)
        if not data:
            break
        calls = data.get('calls', [])
        all_calls.extend(calls)
        cursor = data.get('records', {}).get('cursor')
        print(f"got cursor {cursor}")
        if not cursor:
            break

    call_ids = [call["id"] for call in all_calls]
    print(f"Pulled {len(call_ids)} calls from Gong.")

    return call_ids


def run_request(url, headers, params, cursor=None):
    print(f"running requests with cursor {cursor}")

    if cursor:
        params['cursor'] = cursor

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        calls_data = response.json()
        return calls_data
    else:
        raise RuntimeError(f"Failed to retrieve calls. Status code: {response.status_code}\n{response.text}")