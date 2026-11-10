# Curate a video dataset for generative video

Turn long-form footage into short, filtered, captioned clips for training a video model, the
way open video generators such as Open-Sora 2.0 and Stable Video Diffusion build their data.
Each few-minute chunk of film is decoded and cut into shots on a pool of CPU workers, scored
for aesthetics, motion and on-screen text on a pool of GPU workers, filtered, and captioned by
a small vision-language model on a second GPU pool. The result is a parquet manifest and the
kept clips, published as a dataset artifact, and a report with the funnel, contact sheets
per filter, score histograms, captions and throughput per stage.

The tutorial that walks through this code is at
https://www.union.ai/docs/v2/union/tutorials/frontier-ai/video-data-curation/.

## Files

| File | What it does |
| --- | --- |
| `curate.py` | Images, task environments, the cut, score and caption tasks, the thresholds, and the driver |
| `sources.py` | The five source films, their licenses, and how they are split into chunks |
| `shots.py` | Decoding, shot detection, clip cutting, frame sampling and the CPU statistics |
| `scoring.py` | The aesthetic, optical-flow and text scorers, and the captioner |
| `reports.py` | The report tabs and the dataset card |
| `viz.py` | Shared HTML and SVG building blocks for reports and cards |

## Requirements

A Union.ai cluster with NVIDIA L4 GPUs (any CUDA GPU with 16 GB works; change `score_env` and
`caption_env` in `curate.py`). Tasks download the films and the models over the internet.

## Run it

```bash
uv run curate.py                          # all five films: 31 chunks, 56 minutes
uv run curate.py --smoke                  # 4 one-minute chunks from two films
uv run curate.py --set min_aesthetic=5.0  # change a threshold; everything upstream is cached
```

## Footage and license

The clips are cut from five Blender Foundation open movies, all under Creative Commons
Attribution licenses. Tears of Steel and Elephants Dream are read from download.blender.org;
Big Buck Bunny, Sintel and Spring from their 720p transcodes on Wikimedia Commons.

| Film | Credit | License |
| --- | --- | --- |
| [Big Buck Bunny](https://peach.blender.org/) | (c) copyright 2008, Blender Foundation, peach.blender.org | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |
| [Sintel](https://durian.blender.org/) | (c) copyright Blender Foundation, durian.blender.org | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |
| [Tears of Steel](https://mango.blender.org/) | (CC) Blender Foundation, mango.blender.org | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |
| [Elephants Dream](https://orange.blender.org/) | (c) copyright 2006, Blender Foundation, Netherlands Media Art Institute, orange.blender.org | [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/) |
| [Spring](https://studio.blender.org/films/spring/) | (CC) Blender Foundation, studio.blender.org | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

The pipeline changes the films: it cuts them into clips of 2 to 10 seconds, re-encodes them to
360p H.264 at 24 fps, and samples still frames and frame strips from them. The published
dataset carries the same attribution, with links and this note of the changes, in
`ATTRIBUTION.txt`.
