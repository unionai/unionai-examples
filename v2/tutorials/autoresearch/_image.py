# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "flyte>=2.0.0b22",
#     "PyGithub>=2.5.0",
# ]
# ///
#
# Stable image definition — kept separate from run.py so edits to run.py
# don't invalidate the image cache. Only touch this file when the image itself needs to change.

import flyte

image = (
    flyte.Image.from_uv_script(__file__, name="autoresearch-agent", pre=True)
    .with_apt_packages("git")
)
