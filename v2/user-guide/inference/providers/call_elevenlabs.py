# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "elevenlabs",
# ]
# main = "synthesize"
# params = "text='The first move is what sets everything in motion.'"
# ///

"""Generate speech with ElevenLabs from a task with your own API key."""

# {{docs-fragment task}}
import os

import flyte
from flyte.io import File

env = flyte.TaskEnvironment(
    name="elevenlabs-inference",
    image=flyte.Image.from_debian_base().with_pip_packages("elevenlabs"),
    secrets=flyte.Secret(key="elevenlabs-api-key", as_env_var="ELEVENLABS_API_KEY"),
)


@env.task
def synthesize(text: str) -> File:
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    audio = client.text_to_speech.convert(
        text=text,
        voice_id="<voice-id>",
        model_id="eleven_v4",
        output_format="mp3_44100_128",
    )
    with open("speech.mp3", "wb") as f:
        for chunk in audio:
            f.write(chunk)
    return File.from_local_sync("speech.mp3")


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(synthesize, text="The first move is what sets everything in motion.")
    print(run.url)
# {{/docs-fragment task}}
