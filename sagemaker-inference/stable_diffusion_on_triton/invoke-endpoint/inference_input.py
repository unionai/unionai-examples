import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

prompt = "cute dragon creature"
inputs = []
outputs = []

text_obj = np.array([prompt], dtype="object").reshape((-1, 1))

inputs.append(
    httpclient.InferInput("prompt", text_obj.shape, np_to_triton_dtype(text_obj.dtype))
)
inputs[0].set_data_from_numpy(text_obj)

outputs.append(httpclient.InferRequestedOutput("generated_image"))

request_body, header_length = httpclient.InferenceServerClient.generate_request_body(
    inputs, outputs=outputs
)

with open("inference_input.bin", "wb") as f:
    f.write(request_body)

if __name__ == "__main__":
    print(header_length)  # 173
