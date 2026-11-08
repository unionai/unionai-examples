"""GPU stage: aesthetic, motion and text scores for every clip, and captions for the kept ones.

Models are loaded on first use and kept in module globals. In a reusable environment the
Python process outlives the task call, so each replica loads each model once and every later
call on that replica reuses it.
"""

import socket
import threading
import time

import shots

AESTHETIC_HEAD = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)
_models: dict = {}
_lock = threading.Lock()


def replica() -> str:
    return socket.gethostname()


# {{docs-fragment load}}
def scorers() -> dict:
    """CLIP ViT-L/14 with the LAION aesthetic head, RAFT optical flow, and the CRAFT text detector."""
    with _lock:
        if "clip" in _models:
            return _models
        import easyocr
        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        from transformers import CLIPModel, CLIPProcessor

        t0 = time.time()
        clip_id = "openai/clip-vit-large-patch14"
        head = torch.nn.Sequential(  # the improved-aesthetic-predictor MLP, on L2-normalized CLIP embeddings
            torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2), torch.nn.Linear(1024, 128), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64), torch.nn.Dropout(0.1), torch.nn.Linear(64, 16), torch.nn.Linear(16, 1),
        )
        state = torch.hub.load_state_dict_from_url(AESTHETIC_HEAD, map_location="cpu", file_name="aesthetic-l14.pth")
        head.load_state_dict({k.removeprefix("layers."): v for k, v in state.items()})
        weights = Raft_Large_Weights.DEFAULT
        _models.update(
            clip=CLIPModel.from_pretrained(clip_id, torch_dtype=torch.float16).cuda().eval(),
            clip_processor=CLIPProcessor.from_pretrained(clip_id),
            aesthetic=head.cuda().eval(),
            raft=raft_large(weights=weights).cuda().eval(),
            raft_transforms=weights.transforms(),
            # Detection only: we need where the text is, not what it says.
            ocr=easyocr.Reader(["en"], gpu=True, recognizer=False, verbose=False),
            load_s=round(time.time() - t0, 1),
        )
        print(f"loaded scorers on {replica()} in {_models['load_s']}s")
        return _models
# {{/docs-fragment load}}


def _frames(chunk_dir: str, clip_id: str, side: str = "a"):
    from PIL import Image

    return [Image.open(f"{chunk_dir}/frames/{clip_id}_{k}{side}.jpg").convert("RGB") for k in range(len(shots.POSITIONS))]


# {{docs-fragment score}}
def aesthetic(m: dict, images: list) -> list[float]:
    """The LAION aesthetic predictor (a 1 to 10 scale) on each image."""
    import torch

    with torch.inference_mode():
        pixels = m["clip_processor"](images=images, return_tensors="pt")["pixel_values"].cuda().half()
        emb = m["clip"].get_image_features(pixel_values=pixels).float()
        return m["aesthetic"](emb / emb.norm(dim=-1, keepdim=True)).squeeze(-1).tolist()


def motion(m: dict, firsts: list, seconds: list, height: int = 256) -> list[float]:
    """Mean RAFT optical-flow magnitude per pair, in pixels per frame at ``height`` px tall."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    w = round(firsts[0].width * height / firsts[0].height / 8) * 8  # RAFT needs sides divisible by 8

    def batch(images):
        t = torch.from_numpy(np.stack([np.asarray(i) for i in images])).permute(0, 3, 1, 2).cuda()
        return F.interpolate(t.float(), size=(height, w), mode="bilinear", antialias=True).clamp(0, 255).to(torch.uint8)

    with torch.inference_mode():
        a, b = m["raft_transforms"](batch(firsts), batch(seconds))
        flow = m["raft"](a, b)[-1]  # the last of RAFT's iterative refinements
        return (flow.norm(dim=1).mean(dim=(1, 2)) / shots.GAP).tolist()


def text_area(m: dict, images: list) -> list[float]:
    """Share of each frame covered by detected text boxes."""
    import numpy as np

    out = []
    for img in images:
        arr = np.asarray(img)
        boxes, polys = m["ocr"].detect(arr)
        area = sum((x1 - x0) * (y1 - y0) for x0, x1, y0, y1 in boxes[0])
        for p in polys[0]:  # rotated boxes, by the shoelace formula
            xs, ys = [q[0] for q in p], [q[1] for q in p]
            area += abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(len(p)))) / 2
        out.append(min(area / (arr.shape[0] * arr.shape[1]), 1.0))
    return out


def score_clips(chunk_dir: str, clip_ids: list[str]) -> tuple[list[dict], dict]:
    """Score every clip of one chunk. Returns per-clip scores and GPU seconds per scorer."""
    import torch

    m = scorers()
    seconds = {"aesthetic": 0.0, "motion": 0.0, "text": 0.0}
    rows = []
    for cid in clip_ids:
        firsts, pairs = _frames(chunk_dir, cid, "a"), _frames(chunk_dir, cid, "b")
        row = {"id": cid}
        for name, fn, agg in (
            ("aesthetic", lambda: aesthetic(m, firsts), lambda v: sum(v) / len(v)),
            ("motion", lambda: motion(m, firsts, pairs), lambda v: sum(v) / len(v)),
            ("text", lambda: text_area(m, firsts), max),  # a title card in any sampled frame counts
        ):
            t0 = time.time()
            row[name] = round(float(agg(fn())), 4)
            torch.cuda.synchronize()
            seconds[name] += time.time() - t0
        rows.append(row)
    return rows, {k: round(v, 2) for k, v in seconds.items()}
# {{/docs-fragment score}}


CAPTION_PROMPT = (
    "These are four frames from one short video clip, in order. Write one sentence of at most 40 "
    "words that describes the clip for a text-to-video training set: the main subject, what "
    "happens, the setting, and how the camera moves. Describe only what you can see."
)


# {{docs-fragment caption}}
def captioner(model_id: str) -> tuple[dict, float]:
    """The captioning model and its processor; also how long this call spent loading them (0 if warm)."""
    with _lock:
        if _models.get("vlm_id") == model_id:
            return _models, 0.0
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        t0 = time.time()
        processor = AutoProcessor.from_pretrained(model_id)
        processor.image_processor.do_image_splitting = False  # one tile per frame is plenty at 360p
        model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda().eval()
        _models.update(vlm_id=model_id, vlm=model, vlm_processor=processor)
        load_s = round(time.time() - t0, 1)
        print(f"loaded {model_id} on {replica()} in {load_s}s")
        return _models, load_s


def caption(model_id: str, chunk_dir: str, clip_id: str, max_new_tokens: int = 72) -> tuple[str, float, float]:
    """Caption one clip from its four sampled frames. Returns the caption, GPU seconds and load seconds."""
    import torch

    m, load_s = captioner(model_id)
    images = _frames(chunk_dir, clip_id, "a")
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": CAPTION_PROMPT}]
    processor = m["vlm_processor"]
    prompt = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
    with _lock:  # one generation at a time on the GPU; downloads for other calls overlap with it
        t0 = time.time()
        inputs = processor(text=prompt, images=[images], return_tensors="pt").to("cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            out = m["vlm"].generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return " ".join(text.split()), time.time() - t0, load_s
# {{/docs-fragment caption}}
