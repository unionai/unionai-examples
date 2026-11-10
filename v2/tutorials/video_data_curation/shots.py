"""CPU stage: decode a chunk, find the shot boundaries, cut clips, and sample their frames.

Each clip keeps four sample points. At each one we save the frame and the frame two steps
later: the first frames feed the aesthetic scorer, the text detector and the captioner, and
the pairs feed optical flow. Exposure, sharpness and a frame-difference motion estimate are
cheap enough to compute here, on the CPU, while the frames are in memory.
"""

import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Identify the client to the media servers, as Wikimedia's user-agent policy asks.
USER_AGENT = "unionai-examples video-data-curation tutorial (https://github.com/unionai/unionai-examples)"
FPS = 24  # every source is resampled to 24 fps
HEIGHT = 360  # proxy resolution; the scorers and the captioner work at or below this
MIN_S, MAX_S = 2.0, 10.0  # clip length bounds, as in Open-Sora's pipeline
TRIM = 2  # frames dropped at each end of a shot, to stay clear of the cut itself
GAP = 2  # frames between the two frames of a motion pair
POSITIONS = (0.125, 0.375, 0.625, 0.875)  # where in each clip the frames are sampled


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


# {{docs-fragment decode}}
def decode_window(url: str, start_s: float, duration_s: float, out: str) -> None:
    """Read ``duration_s`` seconds of a remote film into a 360p, 24 fps H.264 proxy.

    ``-ss`` before ``-i`` seeks with HTTP range requests, so only this window is downloaded.
    A keyframe every second keeps cutting clips out of the proxy fast and exact.
    """
    ffmpeg("-user_agent", USER_AGENT, "-ss", str(start_s), "-i", url, "-t", str(duration_s),
           "-vf", f"fps={FPS},scale=-2:{HEIGHT}", "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-g", str(FPS), "-pix_fmt", "yuv420p", out)
# {{/docs-fragment decode}}


# {{docs-fragment shots}}
def detect_shots(path: str) -> list[tuple[int, int]]:
    """Shot boundaries as (first frame, end frame) with PySceneDetect's adaptive content detector."""
    from scenedetect import AdaptiveDetector, SceneManager, open_video

    video = open_video(path)
    manager = SceneManager()
    manager.add_detector(AdaptiveDetector())
    manager.detect_scenes(video)
    return [(a.get_frames(), b.get_frames()) for a, b in manager.get_scene_list(start_in_scene=True)]


def shots_to_clips(shots: list[tuple[int, int]]) -> tuple[list[tuple[int, int, int]], int]:
    """Trim each shot, drop the ones under 2 s, and split the ones over 10 s into equal pieces.

    Returns (start frame, end frame, shot index) per clip, and the number of shots dropped.
    """
    clips, short = [], 0
    for i, (a, b) in enumerate(shots):
        a, b = a + TRIM, b - TRIM
        n = b - a
        if n < MIN_S * FPS:
            short += 1
            continue
        k = math.ceil(n / (MAX_S * FPS))
        clips += [(a + round(j * n / k), a + round((j + 1) * n / k), i) for j in range(k)]
    return clips, short
# {{/docs-fragment shots}}


# {{docs-fragment sample}}
def sample_frames(proxy: str, clips: list[tuple[int, int, int]], ids: list[str], out_dir: str) -> list[dict]:
    """One pass over the proxy: save each clip's frame pairs and compute its CPU statistics."""
    import cv2
    import numpy as np

    wanted: dict[int, list[tuple[int, int, str]]] = {}  # frame -> [(clip, sample point, "a"|"b")]
    for c, (a, b, _) in enumerate(clips):
        for k, p in enumerate(POSITIONS):
            f = a + int(p * (b - a - GAP))
            wanted.setdefault(f, []).append((c, k, "a"))
            wanted.setdefault(f + GAP, []).append((c, k, "b"))

    stats = [{"brightness": [], "sharpness": [], "frame_diff": []} for _ in clips]
    first: dict[tuple[int, int], np.ndarray] = {}
    cap = cv2.VideoCapture(proxy)
    f, last = 0, max(wanted)
    while f <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for c, k, side in wanted.get(f, []):
            cv2.imwrite(f"{out_dir}/frames/{ids[c]}_{k}{side}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if side == "a":
                first[(c, k)] = gray
                stats[c]["brightness"].append(float(gray.mean()))
                stats[c]["sharpness"].append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                if k == 1:  # a small thumbnail for the report and the card
                    h = 108
                    thumb = cv2.resize(frame, (round(frame.shape[1] * h / frame.shape[0]), h), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(f"{out_dir}/thumbs/{ids[c]}.jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            else:
                diff = np.abs(gray.astype(np.int16) - first.pop((c, k)).astype(np.int16)).mean()
                stats[c]["frame_diff"].append(float(diff) / GAP)
        f += 1
    cap.release()
    return [{k: round(sum(v) / max(len(v), 1), 3) for k, v in s.items()} for s in stats]
# {{/docs-fragment sample}}


def cut_clips(proxy: str, clips: list[tuple[int, int, int]], ids: list[str], out_dir: str, workers: int = 4) -> None:
    """Write each clip as its own small MP4, ready to train on or to look at."""
    def cut(c: int) -> None:
        a, b, _ = clips[c]
        ffmpeg("-ss", f"{a / FPS:.3f}", "-i", proxy, "-frames:v", str(b - a), "-an", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               f"{out_dir}/clips/{ids[c]}.mp4")

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(cut, range(len(clips))))


def write_index(out_dir: str, rows: list[dict]) -> None:
    with open(os.path.join(out_dir, "clips.json"), "w") as f:
        json.dump(rows, f)
