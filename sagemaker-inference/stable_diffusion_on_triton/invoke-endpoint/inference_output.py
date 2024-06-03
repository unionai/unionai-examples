import numpy as np
import tritonclient.http as httpclient
from PIL import Image

header_length_prefix = "application/vnd.sagemaker-triton.binary+json;json-header-size="

# download the inference output file
with open("inference_output.out", "rb") as f:
    content = f.read()

# run this command `aws s3api head-object --bucket stable-diffusion-sagemaker --key inference-output/output/<output-file>` to get the response header length
result = httpclient.InferenceServerClient.parse_response_body(
    content, header_length=166
)

image_array = result.as_numpy("generated_image")
image = Image.fromarray(np.squeeze(image_array))
image.save("output.png")
