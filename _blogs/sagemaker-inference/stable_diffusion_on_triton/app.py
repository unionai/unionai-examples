import boto3
import numpy as np
import streamlit as st
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import np_to_triton_dtype


# Function to generate the request body for Triton Inference Server
def generate_request(prompt):
    inputs = []
    outputs = []

    text_obj = np.array([prompt], dtype="object").reshape((-1, 1))

    inputs.append(
        httpclient.InferInput(
            "prompt", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
        )
    )
    inputs[0].set_data_from_numpy(text_obj)

    outputs.append(httpclient.InferRequestedOutput("generated_image"))

    request_body, header_length = (
        httpclient.InferenceServerClient.generate_request_body(inputs, outputs=outputs)
    )

    return request_body, header_length


# Function to call the SageMaker endpoint and get the image
def get_image_from_sagemaker(endpoint_name, prompt, region):
    # Generate the request body for Triton Inference Server
    request_body, header_length = generate_request(prompt)

    runtime_sm_client = boto3.client("sagemaker-runtime", region_name=region)
    response = runtime_sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/vnd.sagemaker-triton.binary+json;json-header-size={}".format(
            header_length
        ),
        Body=request_body,
    )

    # Ensure the request was successful
    if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
        # Parse the response body
        result = httpclient.InferenceServerClient.parse_response_body(
            response["Body"].read(),
            header_length=int(
                response["ContentType"][
                    len(
                        "application/vnd.sagemaker-triton.binary+json;json-header-size="
                    ) :
                ]
            ),
        )

        # Convert the result to a NumPy array and then to an image
        image_array = result.as_numpy("generated_image")
        img = Image.fromarray(np.squeeze(image_array))
        return img
    else:
        st.error(
            "Failed to get a response from SageMaker. Status code: {}".format(
                response["ResponseMetadata"]["HTTPStatusCode"]
            )
        )
        return None


# Streamlit UI
st.title("SageMaker Image Generator")

# Inputs
endpoint_name = st.text_input("SageMaker endpoint name")
region = st.text_input("Region")
prompt = st.text_input("Prompt")

if st.button("Generate Image"):
    if endpoint_name and prompt:
        with st.spinner("Generating image..."):
            # Get the image from SageMaker
            img = get_image_from_sagemaker(endpoint_name, prompt, region)

            if img is not None:
                st.image(img, caption="Generated Image")
            else:
                st.error("Failed to generate image.")
    else:
        st.warning("Please provide the SageMaker endpoint name, region, and prompt.")
