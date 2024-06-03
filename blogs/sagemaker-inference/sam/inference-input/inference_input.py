import base64
import json

with open("input_data.png", "rb") as image_file:
    image_data = base64.b64encode(image_file.read())

payload = {"image_data": image_data.decode("utf-8"), "prompt": [58, 23, 219, 107]}

if __name__ == "__main__":
    print(json.dumps(payload))  # copy the output to inference input file
