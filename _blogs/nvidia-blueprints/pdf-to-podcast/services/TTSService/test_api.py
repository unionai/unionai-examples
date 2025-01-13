import requests
import ujson as json
import os
import time
from fastapi import HTTPException
from datetime import datetime


def get_time():
    return datetime.now().strftime("%H:%M:%S")


def get_output_with_retry(base_url: str, job_id: str, max_retries=10, retry_delay=2):
    """Retry getting output with exponential backoff"""
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{base_url}/output/{job_id}")
            if response.status_code == 200:
                return response.content
            elif response.status_code == 202:
                # Result is being prepared, use shorter delay
                wait_time = min(retry_delay * (1.5**attempt), 10)  # Cap at 10 seconds
                print(
                    f"[{get_time()}] Result is being prepared, retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
                continue
            elif response.status_code == 404:
                # Result not found or job doesn't exist
                raise HTTPException(status_code=404, detail="Result not found")
            else:
                print(
                    f"[{get_time()}] Unexpected status code {response.status_code}: {response.text}"
                )
                response.raise_for_status()
        except requests.RequestException as e:
            print(f"[{get_time()}] Error getting output: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay * (2**attempt))

    raise TimeoutError("Failed to get output after maximum retries")


def test_tts_api():
    # API endpoint URLs
    base_url = os.getenv("TTS_SERVICE_URL", "http://localhost:8889")
    generate_url = f"{base_url}/generate_tts"

    print(f"[{get_time()}] Starting TTS test...")

    try:
        # Load sample JSON data
        with open("sample.json", "r") as f:
            data = json.load(f)

        # Add job_id if not present
        if "job_id" not in data:
            data["job_id"] = str(int(time.time()))

        # Make initial request to generate TTS
        print(f"[{get_time()}] Submitting TTS generation request...")
        response = requests.post(generate_url, json=data)

        if response.status_code != 202:
            print(
                f"[{get_time()}] Error: Unexpected status code {response.status_code}"
            )
            print(response.text)
            return

        # Get job ID from response
        job_data = response.json()
        job_id = job_data["job_id"]
        print(f"[{get_time()}] Job ID received: {job_id}")

        # Monitor job status
        last_status = None
        while True:
            status_response = requests.get(f"{base_url}/status/{job_id}")
            if status_response.status_code != 200:
                print(f"[{get_time()}] Error checking status: {status_response.text}")
                return

            status_data = status_response.json()
            current_status = status_data.get("status")
            message = status_data.get("message", "")

            # Only print if status has changed
            if current_status != last_status:
                print(f"[{get_time()}] Status: {current_status} - {message}")
                last_status = current_status

            if current_status in ["completed", "COMPLETED", "JobStatus.COMPLETED"]:
                break
            elif current_status in ["failed", "FAILED", "JobStatus.FAILED"]:
                print(f"[{get_time()}] Job failed: {message}")
                return

            time.sleep(2)

        # Get the final output with retry logic
        print(f"[{get_time()}] Retrieving audio file...")
        audio_content = get_output_with_retry(base_url, job_id)

        # Save the audio file
        output_filename = f"output_{job_id}.mp3"
        with open(output_filename, "wb") as f:
            f.write(audio_content)
        print(f"[{get_time()}] Audio file saved as '{output_filename}'")

    except Exception as e:
        print(f"[{get_time()}] Error: {str(e)}")


if __name__ == "__main__":
    test_tts_api()
