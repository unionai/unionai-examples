import base64
from typing import Iterator

from flytekit import task
from flytekit.types.file import FlyteFile
from flytekit.types.iterator import JSON


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


@task
def create_request(img_dir: list[FlyteFile]) -> Iterator[JSON]:
    for i, file in enumerate(img_dir):
        file_path = file.download()
        if file_path:
            base64_image = encode_image_to_base64(file_path)
            completion_request = {
                "model": "gpt-4-turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Answer the following yes or no question with either 'Yes.' or 'No.' followed by a description of why. Does this image have explicit content?",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 300,
            }
            batch_request = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": completion_request,
            }
            yield batch_request
