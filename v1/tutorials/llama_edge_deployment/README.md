# iServeAI

Fine-tunes a Llama 3 model on the Cohere Aya Telugu subset and generates a model artifact for deployment as an iOS app.

## Step 1: Store secrets

Start by securely storing your Weights & Biases and Hugging Face tokens as secrets:

```shell
$ union create secret <name-of-the-secret>
```

## Step 2: Replace placeholders with actual values

Ensure you replace the placeholders `WANDB_PROJECT`, `WANDB_ENTITY`, and `HF_REPO_ID` with the actual values for your Weights & Biases project and entity settings, as well as the Hugging Face repository ID, before running the workflow.

`mlc_llm_source_dir` must be an absolute path in `ios_app.py`.

- `WANDB_PROJECT`
- `WANDB_ENTITY`
- `HF_REPO_ID`
- `HF_REPO_URL`, e.g., `HF://<HF_REPO_ID>`

## Step 3: Register workflows

Register the fine-tuning and conversion workflows:

```shell
$ union register llama_edge_deployment.py
$ union launchplan finetuning_completion_trigger --activate
```

## Step 4: Run the fine-tuning workflow

Execute the fine-tuning workflow on the UI.
This initiates the fine-tuning process, converts the weights to MLC format, and uploads the model to Hugging Face.

## Step 5: Build the iOS app

Finally, run the `ios_app_installation.sh` script locally to build the iOS app using the `ios_app.py` file.

> Ensure Xcode is installed before building the iOS app.
