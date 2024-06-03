import os

import numpy as np
from flytekit import ImageSpec, Resources, task
from flytekit.extras.accelerators import T4

model_image = ImageSpec(
    name="sam-model",
    registry=os.getenv("REGISTRY"),
    packages=["transformers", "torch", "monai", "flytekit", "datasets", "matplotlib"],
    cuda="12.1.0",
    cudnn="8",
    python_version="3.11",
)


if model_image.is_container():
    from statistics import mean

    import monai
    import torch
    from datasets import load_dataset
    from torch.optim import Adam
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
    from transformers import SamModel, SamProcessor

    class SAMDataset(Dataset):
        def __init__(self, dataset, processor):
            self.dataset = dataset
            self.processor = processor

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            image = item["image"]
            ground_truth_mask = np.array(item["label"])

            # get bounding box prompt
            prompt = get_bounding_box(ground_truth_mask)

            # prepare image and prompt for the model
            inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

            # remove batch dimension which the processor adds by default
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}

            # add ground truth segmentation
            inputs["ground_truth_mask"] = ground_truth_mask

            return inputs


def get_bounding_box(ground_truth_map):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox


@task(
    cache=True,
    cache_version="2",
    container_image=model_image,
    requests=Resources(gpu="1", mem="20Gi"),
    accelerator=T4,
)
def fine_tune_sam(dataset_name: str) -> torch.nn.Module:
    dataset = load_dataset(dataset_name, split="train")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train_dataset = SAMDataset(dataset=dataset, processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    model = SamModel.from_pretrained("facebook/sam-vit-base")

    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    # Note: Hyperparameter tuning could improve performance here
    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

    seg_loss = monai.losses.DiceCELoss(
        sigmoid=True, squared_pred=True, reduction="mean"
    )

    num_epochs = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
            # forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                input_boxes=batch["input_boxes"].to(device),
                multimask_output=False,
            )

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f"EPOCH: {epoch}")
        print(f"Mean loss: {mean(epoch_losses)}")

    return model
