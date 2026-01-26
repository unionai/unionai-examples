# Model Training and Serving

This example demonstrates an end-to-end machine learning workflow using Flyte 2.0:

1. **Train**: Fine-tune a DistilGPT-2 model on the wikitext-2 dataset
2. **Serve**: Deploy the trained model as a FastAPI REST endpoint

## Prerequisites

- Python 3.12
- A Flyte/Union configuration file (`.flyte/config.yaml`)
- Access to a Union or Flyte instance

## Project Structure

```
model-training-serving/
├── train.py    # Training pipeline - fine-tunes DistilGPT-2
├── serve.py    # Serving app - FastAPI endpoint for text generation
└── README.md   # This file
```

## Running the Example

### Step 1: Train the Model

Run the training pipeline to fine-tune DistilGPT-2:

```bash
cd v2/tutorials/model-training-serving
uv run train.py
```

This will:
- Load and tokenize the wikitext-2 dataset
- Fine-tune DistilGPT-2 for 1 epoch (configurable)
- Save the model as a tar archive artifact

The training run URL will be printed. You can monitor progress in the Union/Flyte UI.

**Training parameters:**
- `max_samples`: Number of training samples (default: 1000)
- `epochs`: Number of training epochs (default: 1)

### Step 2: Deploy the Serving App

Once training is complete, deploy the model as an API:

```bash
uv run serve.py
```

This will:
- Load the trained model from the most recent `training_pipeline` run
- Start a FastAPI server with text generation endpoints

### Step 3: Test the API

Once deployed, test the API using curl:

```bash
# Health check
curl https://<your-app-url>/health

# Generate text
curl -X POST "https://<your-app-url>/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_length": 100,
    "temperature": 0.7
  }'
```

Or visit the Swagger docs at `https://<your-app-url>/docs`.

## API Endpoints

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /generate`

Generate text from a prompt.

**Request:**
```json
{
  "prompt": "The quick brown fox",
  "max_length": 100,
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.95
}
```

**Response:**
```json
{
  "prompt": "The quick brown fox",
  "generated_text": "The quick brown fox jumped over the lazy dog...",
  "model_name": "distilgpt2-finetuned"
}
```

## Key Concepts Demonstrated

- **TaskEnvironment**: Defines compute resources and container image for training
- **FastAPIAppEnvironment**: Configures the serving app with FastAPI
- **Parameter with RunOutput**: Links training outputs to serving inputs
- **@env.server decorator**: Handles model loading on app startup
- **flyte.io.File**: Manages model artifacts between training and serving

## Customization

### Using a Different Model

To fine-tune a different model, modify the `MODEL_NAME` in `train.py`:

```python
# In prepare_data and fine_tune_model functions
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # or any causal LM
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

### Using GPU for Training

The training automatically uses GPU if available. To request a specific GPU:

```python
training_env = flyte.TaskEnvironment(
    ...
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu="T4:1"),
)
```

### Adjusting Training Parameters

```bash
# Train with more data and epochs
uv run train.py --max_samples 5000 --epochs 3
```

## Related Documentation

- [FastAPI Apps](https://union.ai/docs/v2/byoc/user-guide/build-apps/fastapi-app)
- [Model Serving](https://union.ai/docs/v2/byoc/user-guide/serve-and-deploy-apps/)
- [Files and Directories](https://union.ai/docs/v2/byoc/user-guide/task-programming/files-and-directories)
