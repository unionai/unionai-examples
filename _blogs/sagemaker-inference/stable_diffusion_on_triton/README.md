# Deploy Stable Diffusion Using Triton Inference Server on AWS SageMaker

## Overview

This pipeline enables you to fine-tune and deploy a Stable Diffusion model using Triton Inference Server on AWS SageMaker with minimal configuration.

## Execution Steps

1. **Fine-tuning**:

   The model undergoes distributed fine-tuning on a single node with 8 NVIDIA GPU instances using the Flyte PyTorch plugin.
   Fine-tuning uses the `svjack/pokemon-blip-captions-en-zh` dataset, consisting of 833 annotated images, and employs LoRA for optimization.
   The process takes approximately 2.5 hours. Caching is enabled to avoid redundant fine-tuning with the same configuration. The HuggingFace token must be provided to push the model to the HuggingFace hub.

2. **Optimization**:

   The fine-tuned model is optimized using ONNX and TensorRT on an A10G instance.
   The optimization step generates a directory with Triton Inference Server configurations for efficient inference.
   The optimized model is compressed into a tar.gz file for deployment.

3. **Deployment**:

   The compressed, optimized model is deployed to a SageMaker endpoint, making it production-ready for real-time inference requests.
   It is possible to switch the inference mode to asynchronous or serverless.

## Fine-tuning

- Instance Type: T4
- Setup: Distributed training on 5 GPUs
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

## Installation and Execution

To install the required packages locally, run:

```bash
pip install -r requirements.txt
```

To register the workflows, run:

```bash
REGISTRY=ghcr.io/unionai-oss EXECUTION_ROLE_ARN=<YOUR_EXECUTION_ROLE_ARN> unionai register stable_diffusion_on_triton
```

> In the `fine_tune.py` file, replace `hub_model_id`, `SECRET_GROUP`, and `SECRET_KEY` with your HF model ID and secret.

> To run the non-finetuned workflow, ensure you replace `model` in the `backend/pipeline/1/model.py` file with the name of the non-finetuned model, e.g., `CompVis/stable-diffusion-v1-4`.

Workflows to execute: `stable_diffusion_on_triton.workflow.stable_diffusion_on_triton_wf` and `stable_diffusion_on_triton.non_finetuned_workflow.stable_diffusion_on_triton_wf`

To run the streamlit app, export AWS credentials in your terminal and run:

```bash
streamlit run app.py
```
