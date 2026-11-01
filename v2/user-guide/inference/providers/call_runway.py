# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "runwayml",
# ]
# main = "generate_video"
# params = "image_url=https://example.com/image.jpg prompt='A timelapse on a sunny day'"
# ///

"""Generate video with Runway from a task with your own API key."""

# {{docs-fragment task}}
import os

import flyte

env = flyte.TaskEnvironment(
    name="runway-inference",
    image=flyte.Image.from_debian_base().with_pip_packages("runwayml"),
    secrets=flyte.Secret(key="runway-api-key", as_env_var="RUNWAYML_API_SECRET"),
)


@env.task
def generate_video(image_url: str, prompt: str) -> list[str]:
    from runwayml import RunwayML

    client = RunwayML(api_key=os.environ["RUNWAYML_API_SECRET"])
    task = client.image_to_video.create(
        model="gen4.5",
        prompt_image=image_url,
        prompt_text=prompt,
        ratio="1280:720",
        duration=5,
    ).wait_for_task_output()
    return task.output


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(generate_video, image_url="https://example.com/image.jpg", prompt="A timelapse on a sunny day")
    print(run.url)
# {{/docs-fragment task}}
