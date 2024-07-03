import os
from typing import Dict

import torch
import torchvision.models as models
from PIL import Image
from flytekitplugins.ray import RayJobConfig, WorkerNodeConfig
import ray
from torchvision import transforms

from flytekit import task, workflow, Resources, ImageSpec
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

custom_image = ImageSpec(
    registry=os.environ.get("IMAGE_SPEC_REGISTRY"),
    requirements="requirements.txt",
    apt_packages=["wget"],
)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@ray.remote(num_gpus=1)
def process_batch(ray_batch_keys: list[FlyteFile], torch_batch_size: int) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True).to(device)
    model.eval()

    ray_batch_classes = []
    for j in range(0, len(ray_batch_keys), torch_batch_size):
        torch_batch_keys = ray_batch_keys[j:j + torch_batch_size]
        images = []
        for img_file in torch_batch_keys:
            img_file.download()
            img = Image.open(img_file.path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            images.append(img_tensor)

        batch_tensor = torch.cat(images, dim=0)
        out = model(batch_tensor)
        ray_batch_classes += torch.argmax(out, dim=1).cpu().tolist()

    class_dict = {im.remote_source.split("/")[-1]: class_val for im, class_val in
                  zip(ray_batch_keys, ray_batch_classes)}

    return class_dict


@task(
    container_image=custom_image,
    requests=Resources(mem="5Gi", cpu="2", gpu="1"),
    task_config=RayJobConfig(
        shutdown_after_job_finishes=True,
        worker_node_config=[
            WorkerNodeConfig(
                group_name="ray-job", replicas=4
            )
        ]),
)
def process_images_in_batches(input_bucket: FlyteDirectory, ray_batch_size: int, torch_batch_size: int) -> Dict[str, int]:
    image_files = FlyteDirectory.listdir(input_bucket)[1:]  # ignore dir

    futures = [process_batch.remote(image_files[i:i + ray_batch_size], torch_batch_size)
               for i in range(0, len(image_files), ray_batch_size)]
    pred_dits = ray.get(futures)
    combined_preds_dict = {}
    for d in pred_dits:
        combined_preds_dict.update(d)
    return combined_preds_dict


@workflow
def ray_wf(input_bucket: FlyteDirectory, ray_batch_size: int, torch_batch_size: int) -> Dict[str, int]:
    return process_images_in_batches(
        input_bucket=input_bucket,
        ray_batch_size=ray_batch_size,
        torch_batch_size=torch_batch_size,
    )
