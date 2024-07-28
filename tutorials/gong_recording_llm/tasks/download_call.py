from flytekit import task, ImageSpec, Secret, current_context
from flytekit.types.file import FlyteFile
import base64
import requests
import os
import json


download_call_img = ImageSpec(
    packages=[
        "flytekit==1.13.0",
        "union==0.1.48"
    ],
    registry=os.getenv("DOCKER_REGISTRY")
)


from tasks.query_calls import CallData

@task(
    container_image=download_call_img,
    secret_requests=[Secret(key="gong_access_key"), Secret(key="gong_secret_key")],
    cache=True,
    cache_version="1.0",
)
def download_call(call_data: CallData) -> CallData:
    call_id = call_data.id
    extensive_url = "https://api.gong.io/v2/calls/extensive"

    access_key = current_context().secrets.get(key="gong_access_key")
    secret_key = current_context().secrets.get(key="gong_secret_key")
    basic_token = base64.b64encode(f"{access_key}:{secret_key}".encode()).decode()

    headers = {
        "Authorization": f"Basic {basic_token}",
        "Content-Type": "application/json"
    }

    body = {
        "filter": {
            "callIds": [call_id]
        },
        "contentSelector": {
            "context": "None",
            "exposedFields": {
                "parties": True,
                "content": {
                    "structure": False,
                    "topics": False,
                    "trackers": False,
                    "trackerOccurrences": False,
                    "pointsOfInterest": False,
                    "brief": False,
                    "outline": False,
                    "highlights": False,
                    "callOutcome": False,
                    "keyPoints": False
                },
                "interaction": {
                    "speakers": False,
                    "video": False,
                    "personInteractionStats": False,
                    "questions": False
                },
                "collaboration": {
                    "publicComments": False
                },
                "media": True
            }
        }
    }
    print(f"Downloading call {call_id}...")
    extensive_response = requests.post(extensive_url, headers=headers, json=body)

    if extensive_response.status_code == 200:
        extensive_data = extensive_response.json()
        call_metadata = extensive_data["calls"][0]

        metadata_path = os.path.join(os.getcwd(), call_id + ".json")
        with open(metadata_path, "w") as json_file:
            json.dump(call_metadata, json_file, indent=4)
        print(f"Metadata saved as {metadata_path}")

        audio_url = call_metadata["media"]["audioUrl"]
        audio_path = os.path.join(os.getcwd(), call_id + ".mp3")
        response = requests.get(audio_url, stream=True)

        if response.status_code == 200:
            with open(audio_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded file to {audio_path}")
            return CallData(
                id=call_id,
                call_metadata=FlyteFile(metadata_path),
                call_audio=FlyteFile(audio_path),
                transcription=None
            )
        else:
            raise RuntimeError(f"Failed to download file. Status code: {response.status_code}\n{response.text}")

    else:
        raise RuntimeError(f"Failed to retrieve extensive call details. Status code: {extensive_response.status_code}\n{extensive_response.text}")
