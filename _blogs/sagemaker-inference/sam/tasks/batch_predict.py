import os
from pathlib import Path

import flytekit
import numpy as np
from flytekit import Resources, task
from flytekit.extras.accelerators import T4
from flytekit.types.directory import FlyteDirectory

from .fine_tune import get_bounding_box, model_image

if model_image.is_container():
    import matplotlib.pyplot as plt
    import torch
    from datasets import load_dataset
    from transformers import SamProcessor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


@task(
    cache=True,
    cache_version="2",
    container_image=model_image,
    requests=Resources(gpu="1", mem="20Gi"),
    accelerator=T4,
)
def batch_predict(model: torch.nn.Module) -> FlyteDirectory:
    dataset = load_dataset("nielsr/breast-cancer", split="train")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set the number of images to process in a batch
    batch_size = 4

    # Define a list to store input images
    images = []

    # Define a list to store input boxes
    input_boxes = []

    # Populate the lists with images and corresponding boxes
    for idx in range(batch_size):
        # Load image (replace this with your image loading logic)
        image = dataset[idx]["image"]
        images.append(image)

        # Get box prompt based on ground truth segmentation map
        ground_truth_mask = np.array(dataset[idx]["label"])
        prompt = get_bounding_box(ground_truth_mask)
        input_boxes.append([prompt])

    # Convert the list of images to a single numpy array
    images = np.array(images)

    # Prepare images + box prompts for the model
    inputs = processor(images, input_boxes=input_boxes, return_tensors="pt").to(device)

    model.eval()

    # Perform forward pass
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    # Apply sigmoid
    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))

    # Convert soft masks to hard masks
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

    # Plot the predicted masks and save as images
    working_dir = flytekit.current_context().working_directory
    local_dir = Path(os.path.join(working_dir, "sam_batch_predictions"))
    local_dir.mkdir(exist_ok=True)

    for idx in range(batch_size):
        img_file_path = os.path.join(local_dir, f"image_{idx + 1}.png")
        fig, axes = plt.subplots()

        axes.imshow(np.array(images[idx]))
        show_mask(medsam_seg[idx], axes)
        axes.title.set_text(f"Image {idx + 1}")

        # Save the plot as an image
        plt.savefig(img_file_path)

        # Close the plot to release memory
        plt.close(fig)

    return FlyteDirectory(str(local_dir))
