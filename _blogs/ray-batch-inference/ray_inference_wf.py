import pickle

import torch
import torchvision.models as models
from PIL import Image
from flytekit import task, workflow, FlyteContextManager, Resources
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from flytekitplugins.ray import RayJobConfig, WorkerNodeConfig
from flytekitplugins.ray.task import ray
from torchvision import transforms

CUSTOM_IMAGE = "your-custom-image"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@ray.remote(num_gpus=1)
def process_batch(ray_batch_keys: list[FlyteFile], torch_batch_size: int, batch_number: int, output_bucket: str) -> str:
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

    ctx = FlyteContextManager.current_context()
    file_name = f"batch_{batch_number}.pkl"
    with open(file_name, 'wb') as f:
        pickle.dump(class_dict, f)
    ctx.file_access.put_data(file_name, f'{output_bucket}/{file_name}')

    return f"batch {batch_number} saved at {output_bucket}/{file_name}"


@task(
    container_image=CUSTOM_IMAGE,
    requests=Resources(mem="5Gi", cpu="2", gpu="1"),
    task_config=RayJobConfig(
        worker_node_config=[
            WorkerNodeConfig(
                group_name="ray-job", replicas=4
            )
        ]),
)
def process_images_in_batches(input_bucket: str, output_bucket: str, ray_batch_size: int, torch_batch_size: int):
    image_dir = FlyteDirectory.from_source(input_bucket)
    image_files = FlyteDirectory.listdir(image_dir)[1:]  # ignore dir

    futures = [process_batch.remote(image_files[i:i + ray_batch_size], torch_batch_size, batch_number, output_bucket)
               for batch_number, i in enumerate(range(0, len(image_files), ray_batch_size))]
    pred_filepaths = ray.get(futures)


@workflow
def ray_wf(input_bucket: str, output_bucket: str, ray_batch_size: int, torch_batch_size: int):
    process_images_in_batches(
        input_bucket=input_bucket,
        output_bucket=output_bucket,
        ray_batch_size=ray_batch_size,
        torch_batch_size=torch_batch_size,
    )
