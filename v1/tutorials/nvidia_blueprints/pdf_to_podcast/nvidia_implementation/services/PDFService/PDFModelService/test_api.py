import requests
import os
import sys

# Define the endpoint URL
URL = "http://127.0.0.1:8003/convert"

# Define the path to the PDF file
PDF_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../PNP_Proof.pdf")
)

if not os.path.exists(PDF_FILE_PATH):
    sys.exit(f"File not found: {PDF_FILE_PATH}")


def test_convert_pdf():
    # Open the PDF file
    with open(PDF_FILE_PATH, "rb") as file:
        # Create a dictionary with the file and job_id. If no job_id is provided, the server will generate one.
        files = {"file": file}
        data = {"job_id": None}

        # Make the request
        response = requests.post(URL, files=files, data=data)

        # Print the response
        print("Response Status Code:", response.status_code)
        print("Response Content:", response.json())

        # Handle the response based on status code
        if response.status_code == 202:
            # If the conversion is successfully started, retrieve the job ID
            job_id = response.json().get("job_id")
            print(f"Job ID: {job_id}")

            # To monitor the job status, you can connect to the WebSocket endpoint
            WS_URL = f"ws://127.0.0.1:8003/ws/{job_id}"
            print(f"WebSocket URL for job status: {WS_URL}")

            # Note: Actually using the WebSocket to get status updates might require additional code.
        else:
            print("Error:", response.json())


if __name__ == "__main__":
    test_convert_pdf()
