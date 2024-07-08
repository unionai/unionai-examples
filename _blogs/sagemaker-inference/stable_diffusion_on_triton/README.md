# Deploy Stable Diffusion Using Triton Inference Server on AWS SageMaker

## Overview

This pipeline enables you to fine-tune and deploy a Stable Diffusion model using Triton Inference Server on AWS SageMaker with minimal configuration.

## Execution Steps

1. **Fine-tuning**:

   The model undergoes distributed fine-tuning on a single node with 8 NVIDIA GPU instances using the Flyte PyTorch plugin.
   Fine-tuning uses the `svjack/pokemon-blip-captions-en-zh` dataset, consisting of 833 annotated images, and employs LoRA for optimization.
   Caching is enabled to avoid redundant fine-tuning with the same configuration. The HuggingFace token must be provided to push the model to the HuggingFace hub.

2. **Optimization**:

   The fine-tuned model is optimized using ONNX and TensorRT on an A10G instance.
   The optimization step generates a directory with Triton Inference Server configurations for efficient inference.
   The optimized model is compressed into a tar.gz file for deployment.

3. **Deployment**:

   The compressed, optimized model is deployed to a SageMaker endpoint, making it production-ready for real-time inference requests.
   It is possible to switch the inference mode to asynchronous or serverless.

## Fine-tuning

- Instance Type: T4
- Setup: Distributed training on 8 GPUs
- Dataset: svjack/pokemon-blip-captions-en-zh
- Duration: 2-3 hours
- Cache: Enabled
- HuggingFace Token: Required to push the model to the HF hub.

> There is also a workflow available that skips fine-tuning and only optimizes and deploys the model.

## Optimization

- Instance Type: A10G
- Tools: ONNX, TensorRT
- Prerequisite: Ensure `hf_env.tar.gz` is available in the `backend/pipeline` directory. It can be generated using the `conda_dependencies.sh` script in a SageMaker TritonServer container.

Ensure `hf_env.tar.gz` is available in the `backend/pipeline` directory. it can be generated using the `conda_dependencies.sh` script in a sagemaker-tritonserver container.

## SageMaker Deployment

- The model is deployed to a SageMaker endpoint for real-time inference.
- Can switch to asynchronous or serverless inference if needed.
- Instance: ml.g5.2xlarge (A10G instance)

> Artifacts can also be consumed by tasks when run as separate entities. In this case, since it's a single workflow, this isn't necessary.

## Installation and Execution

To install the required packages locally, run:

```bash
pip install -r requirements.txt
```

To register the workflows, run:

```bash
REGISTRY=ghcr.io/unionai-oss EXECUTION_ROLE_ARN=<YOUR_EXECUTION_ROLE_ARN> unionai register stable_diffusion_on_triton
```

Workflows to execute: `stable_diffusion_on_triton.workflow.stable_diffusion_on_triton_wf` and `stable_diffusion_on_triton.non_finetuned_workflow.stable_diffusion_on_triton_wf`

To run the streamlit app, export AWS credentials in your terminal and run:

```bash
streamlit run app.py
```

|                                                                           Non-fine-tuned                                                                           |                                                                             Fine-tuned                                                                             |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img width="785" alt="Screenshot 2024-07-06 at 5 00 58 PM" src="https://github.com/unionai/unionai-examples/assets/27777173/80bd5c0e-bf18-472a-aa8d-04ceaffaa571"> | <img width="776" alt="Screenshot 2024-07-06 at 5 00 50 PM" src="https://github.com/unionai/unionai-examples/assets/27777173/de3344c9-4750-40a7-be1d-93a9dc1e8025"> |
| <img width="832" alt="Screenshot 2024-07-06 at 4 59 05 PM" src="https://github.com/unionai/unionai-examples/assets/27777173/11c48991-b91e-4e8c-ad4d-d5a3206f0934"> | <img width="816" alt="Screenshot 2024-07-06 at 4 58 56 PM" src="https://github.com/unionai/unionai-examples/assets/27777173/5b4b97ce-fb03-4034-a1a4-2898f603069a"> |

### Manage script

To use the manage.py script, you need to install `pip install typer`. Once typer is installed, you can run `python manage.py --help` command.

```bash

 Usage: manage.py [OPTIONS] COMMAND [ARGS]...

╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                                                                                                                                           │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                                                                                                                    │
│ --help                        Show this message and exit.                                                                                                                                                                                         │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ delete                                                                                                                                                                                                                                            │
│ status                                                                                                                                                                                                                                            │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Now it is possible to `delete` or get `status` of any endpoint (as long as you have AWS connection) given the endpoint name.
