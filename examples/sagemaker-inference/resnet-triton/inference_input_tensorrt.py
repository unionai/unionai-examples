import json

from inference_input_pytorch import get_sample_image

payload = {
    "inputs": [
        {
            "name": "input",
            "shape": [1, 3, 224, 224],
            "datatype": "FP32",
            "data": get_sample_image(),
        }
    ]
}

if __name__ == "__main__":
    print(json.dumps(payload))  # copy the output to inference input file
