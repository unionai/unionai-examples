import requests
import sys
import ujson as json
import time
from pathlib import Path


def test_pdf_conversion(pdf_paths: list[str], api_url: str = "http://localhost:8003"):
    """
    Test the PDF conversion endpoint by uploading PDF files and displaying the markdown results.

    Args:
        pdf_paths: List of paths to the PDF files to convert
        api_url: Base URL of the API service
    """
    # First check if the service is healthy
    try:
        health_response = requests.get(f"{api_url}/health")
        health_response.raise_for_status()
        print("Health check response:", json.dumps(health_response.json(), indent=2))
    except requests.exceptions.RequestException as e:
        print(f"Error checking service health: {e}")
        sys.exit(1)

    # Check if files exist
    pdf_files = []
    for pdf_path in pdf_paths:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            print(f"Error: File {pdf_path} does not exist")
            sys.exit(1)
        pdf_files.append(pdf_file)

    # Prepare the files for upload
    files = [
        ("files", (pdf_file.name, open(pdf_file, "rb"), "application/pdf"))
        for pdf_file in pdf_files
    ]

    try:  # Make the conversion request
        print(
            f"\nUploading {len(files)} files for conversion: {', '.join(f.name for f in pdf_files)}..."
        )
        response = requests.post(f"{api_url}/convert", files=files)
        response.raise_for_status()

        # Get the task ID
        result = response.json()
        task_id = result["task_id"]
        print(f"Task ID: {task_id}")
        print("Waiting for conversion to complete...")

        # Poll the status endpoint until the task is complete
        while True:
            status_response = requests.get(f"{api_url}/status/{task_id}")

            try:
                status_data = status_response.json()
                print(
                    f"Status check response: Code={status_response.status_code}, Data={status_data}"
                )

                if status_response.status_code == 200:
                    # Task completed successfully
                    results = status_data.get("result", [])
                    if results:
                        # Print each result
                        for result in results:
                            if result["status"] == "success":
                                print(f"Successfully converted {result['filename']}")
                                print(
                                    f"Content: {result['content'][:200]}..."
                                )  # Show first 200 chars
                            else:
                                print(
                                    f"Failed to convert {result['filename']}: {result.get('error', 'Unknown error')}"
                                )
                        return True
                    print(f"No results found in response data: {status_data}")
                    return False
                elif status_response.status_code == 202:
                    # Task still processing
                    print("Task still processing, waiting 2 seconds...")
                    time.sleep(2)
                else:
                    error_msg = status_data.get("error", "Unknown error")
                    print(f"Error response received: {error_msg}")
                    return False
            except Exception as e:
                print(f"Error checking status: {str(e)}")
                return False

    except requests.exceptions.RequestException as e:
        print(f"Error during conversion: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response content: {e.response.text}")
    finally:
        # Ensure the file is closed
        for file_tuple in files:
            file_tuple[1][1].close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python test_pdf_api.py <path_to_pdf_file1> [path_to_pdf_file2 ...]"
        )
        sys.exit(1)

    test_pdf_conversion(sys.argv[1:], "http://localhost:8003")
