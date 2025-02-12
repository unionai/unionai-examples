import requests
import os
import time
from typing import Optional, List
from shared.api_types import StatusResponse
import sys
from pathlib import Path
from urllib.parse import urlparse

PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://localhost:8003")
POLL_INTERVAL = 2  # seconds
MAX_WAIT_TIME = 3600  # seconds
ALLOWED_HOSTS = {"localhost", "127.0.0.1"}
ALLOWED_PORTS = {8003}  # Add any other legitimate ports


def validate_service_url(url: str) -> bool:
    """Validate that the service URL is pointing to an allowed host and port"""
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        if not host or not port:
            return False

        return host in ALLOWED_HOSTS and port in ALLOWED_PORTS
    except Exception:
        return False


def poll_job_status(job_id: str) -> Optional[dict]:
    """Poll the job status until completion or failure"""
    start_time = time.time()

    while time.time() - start_time < MAX_WAIT_TIME:
        try:
            response = requests.get(f"{PDF_SERVICE_URL}/status/{job_id}")
            status_data = StatusResponse.model_validate(response.json())
            # print(f"Polling status... Response: {status_data}")

            # Check the job status from the response
            if status_data.status == "JobStatus.COMPLETED":
                return status_data
            elif status_data.status == "JobStatus.FAILED":
                print(f"Job failed: {status_data.message}")
                return None
            elif status_data.status == "JobStatus.PROCESSING":
                print(f"Job still processing... {status_data.message}")
                time.sleep(POLL_INTERVAL)
                continue
            else:
                print(f"Unknown status: {status_data.status}")
                time.sleep(POLL_INTERVAL)

        except requests.RequestException as e:
            print(f"Error polling status: {e}")
            return None

    print("Error: Job timed out")
    return None


def test_convert_pdf_endpoint(pdf_paths: List[str]) -> bool:
    """
    Test the PDF conversion endpoint by uploading PDF files and displaying the markdown results.

    Args:
        pdf_paths: List of paths to the PDF files to convert
    """
    # Check if files exist
    pdf_files = []
    for pdf_path in pdf_paths:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            print(f"Error: File {pdf_path} does not exist")
            return False
        pdf_files.append(pdf_file)

    # Submit the conversion job
    try:
        # Open all files at once and keep them open until the request is complete
        open_files = []
        files = []

        for pdf_file in pdf_files:
            f = open(pdf_file, "rb")
            open_files.append(f)  # Keep track of open files
            files.append(("files", (pdf_file.name, f, "application/pdf")))

        print(
            f"\nUploading {len(files)} files for conversion: {', '.join(f.name for f in pdf_files)}..."
        )
        response = requests.post(f"{PDF_SERVICE_URL}/convert", files=files)

        if response.status_code != 202:
            print(f"Error: Request failed with status code {response.status_code}")
            return False

        job_data = response.json()
        job_id = job_data.get("job_id")
        if not job_id:
            print("Error: No job_id in response")
            return False

        print(f"Job ID: {job_id}")
        print("Starting job polling...")

        # Poll for job completion
        status_data = poll_job_status(job_id)
        if not status_data:
            return False

        print(f"Job completed. Status data: {status_data}")

        # Get the output
        outputs = get_job_output(job_id)
        if not outputs:
            print("Failed to get output")
            return False

        print("Successfully retrieved output")

        # Validate output content for each PDF
        print("Validating output content...")
        for output in outputs:
            filename = output.get("filename")
            markdown = output.get("markdown", "")
            status = output.get("status")
            error = output.get("error")

            print(f"\nResults for {filename}:")
            if status == "success":
                print("Success: PDF conversion passed!")
                print("First 500 chars of content:")
                print(markdown[:500] + "..." if len(markdown) > 500 else markdown)
            else:
                print(f"Error: Conversion failed - {error}")
                return False

        return True

    except requests.RequestException as e:
        print(f"Error during request: {e}")
        return False
    finally:
        # Close all opened files
        for f in open_files:
            f.close()


def get_job_output(job_id: str) -> Optional[List[dict]]:
    """Get the markdown output for a completed job"""
    try:
        response = requests.get(f"{PDF_SERVICE_URL}/output/{job_id}")
        if response.status_code != 200:
            print(f"Error getting output: {response.status_code}")
            if response.status_code == 404:
                print(
                    "Job result not found. This might mean the job is still processing."
                )
            return None
        return response.json()
    except requests.RequestException as e:
        print(f"Error getting output: {e}")
        return None


def test_convert_pdf_endpoint_invalid_file():
    files = [("files", ("test.txt", b"This is not a PDF file", "text/plain"))]
    try:
        response = requests.post(f"{PDF_SERVICE_URL}/convert", files=files)

        if response.status_code != 400:
            print(f"Error: Expected 400 status code, got {response.status_code}")
            return False

        print("Success: Invalid file test passed!")
        return True

    except requests.RequestException as e:
        print(f"Error during request: {e}")
        return False


def test_health_endpoint():
    try:
        response = requests.get(f"{PDF_SERVICE_URL}/health")

        if response.status_code != 200:
            print(f"Error: Health check failed with status code {response.status_code}")
            return False

        health_data = response.json()
        if health_data.get("status") != "healthy":
            print(
                f"Error: Service unhealthy: {health_data.get('error', 'Unknown error')}"
            )
            return False

        print("Success: Health check passed!")
        return True

    except requests.RequestException as e:
        print(f"Error checking health: {e}")
        return False


def main():
    """Main entry point for the test script"""
    if not validate_service_url(PDF_SERVICE_URL):
        print(
            f"Error: Invalid service URL. Must be one of the allowed hosts: {ALLOWED_HOSTS}"
        )
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python test_api.py <path_to_pdf_file1> [path_to_pdf_file2 ...]")
        sys.exit(1)

    # Validate input length before passing to loop
    MAX_FILES = 1000
    pdf_paths = sys.argv[1:]
    if len(pdf_paths) > MAX_FILES:
        print(f"Error: Too many input files. Maximum allowed is {MAX_FILES}")
        sys.exit(1)

    print("Running PDF Service API tests...")

    print("\nTest 1: Health check")
    if not test_health_endpoint():
        sys.exit(1)

    print("\nTest 2: Valid PDF conversion")
    if not test_convert_pdf_endpoint(sys.argv[1:]):
        sys.exit(1)

    print("\nTest 3: Invalid file handling")
    if not test_convert_pdf_endpoint_invalid_file():
        sys.exit(1)

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()
