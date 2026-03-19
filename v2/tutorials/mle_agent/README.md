# mle-bot

An AI ML Engineer that uses [Flyte's](https://flyte.org) durable sandbox executions to train and evaluate models — in the cloud, not on your laptop.

## What it does

You describe a business problem in natural language. The agent:

1. **Profiles** your dataset (trusted tools touch the data — not the LLM)
2. **Designs** a set of experiments (algorithms, hyperparameters, feature strategies)
3. **Runs** each experiment by generating orchestration code that dispatches real compute as durable Flyte tasks
4. **Iterates** — analyzes results, designs follow-up experiments if needed
5. **Produces** a model card with the winning model's metrics, threshold analysis, and deployment recommendations

The Monty sandbox ensures LLM-generated orchestration code is safe — it can only call pre-approved tool functions and has no access to imports, network, or filesystem.

## Setup

```bash
uv sync
```

Requires `OPENAI_API_KEY` in your environment.

## Usage

### 1. Generate demo dataset (synthetic predictive maintenance data)

```bash
uv run main.py generate-data
```

Generates 175k rows of sensor readings from 20 simulated industrial pumps, with a ~3% failure rate and realistic pre-failure degradation patterns.

### 2. Run the MLE agent

```bash
uv run main.py run \
    --data data/predictive_maintenance.csv \
    --problem "Predict pump failures 24 hours before they happen based on sensor readings" \
    --target failure_24h \
    --time-column timestamp \
    --output results/model_card.md
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-iterations` | 3 | How many rounds of experiments to run |
| `--max-retries` | 3 | Retries per failed sandbox execution |
| `--model` | `gpt-4o` | OpenAI model to use |
| `--output` | (none) | Save model card to file |

## Example use cases

The agent accepts any tabular classification problem described in plain English. It adapts its feature strategy, algorithm choices, and pipeline structure based on the data it receives — not on hardcoded assumptions.

### Predictive maintenance

```bash
uv run main.py run \
    --data data/sensor_readings.csv \
    --problem "Predict equipment failures 24 hours before they happen based on vibration, temperature, and pressure sensor readings" \
    --target failure_24h \
    --time-column timestamp \
    --max-iterations 3
```

The agent will recognize the temporal structure and explore rolling window features to capture degradation trends. It may also try SMOTE or class weighting given the typical ~3% failure rate.

---

### Customer churn

```bash
uv run main.py run \
    --data data/customers.csv \
    --problem "Predict which customers will cancel their subscription in the next 30 days based on product usage, support tickets, and billing history" \
    --target churned \
    --max-iterations 3
```

No time column — the agent treats this as cross-sectional data and focuses on feature interactions, normalization for linear models, and class weighting for the imbalanced churn label.

---

### Credit risk

```bash
uv run main.py run \
    --data data/loan_applications.csv \
    --problem "Predict loan default risk based on applicant financial history and demographic features. High recall is critical — missing a default is more costly than a false alarm." \
    --target defaulted \
    --max-iterations 4 \
    --output results/credit_model_card.md
```

The problem description signals that recall matters more than precision. The agent incorporates this into its reasoning when comparing threshold options and selecting among models.

---

### Medical diagnosis

```bash
uv run main.py run \
    --data data/patient_records.csv \
    --problem "Identify patients at high risk of hospital readmission within 30 days of discharge based on diagnosis codes, lab results, and length of stay" \
    --target readmitted_30d \
    --max-iterations 3
```

---

### Fraud detection

```bash
uv run main.py run \
    --data data/transactions.csv \
    --problem "Detect fraudulent transactions in real-time payment data. The dataset is highly imbalanced — genuine fraud is less than 0.5% of transactions." \
    --target is_fraud \
    --max-iterations 4 \
    --output results/fraud_model_card.md
```

Severe imbalance (< 0.5% positive rate) will prompt the agent to explore SMOTE, aggressive scale_pos_weight tuning, and avg_precision as an additional metric alongside ROC-AUC.

---

### Employee attrition

```bash
uv run main.py run \
    --data data/hr_records.csv \
    --problem "Predict which employees are likely to leave the company in the next 6 months based on performance reviews, tenure, compensation, and engagement survey scores" \
    --target left_company \
    --max-iterations 3
```

---

The problem description is the primary lever. The more context you provide about what matters (e.g., "recall is critical", "false alarms are costly", "this data is ordered by time"), the better the agent can tailor its experimental strategy.

## Project structure

```
mle_bot/
├── environments.py      # Flyte TaskEnvironment definitions (tool_env, agent_env)
├── agent.py             # Main agent loop — LLM orchestration + Monty sandbox
├── synthetic_data.py    # Predictive maintenance dataset generator
└── tools/
    ├── data.py          # profile_dataset, split_dataset
    ├── exploration.py   # explore_dataset (class distributions, correlations, temporal trends, group stats)
    ├── features.py      # engineer_features (rolling stats, lags, normalization)
    ├── resampling.py    # resample_dataset (oversample, undersample, SMOTE)
    ├── selection.py     # select_features (mutual info, variance threshold, correlation filter)
    ├── training.py      # train_model (xgboost, random_forest, gradient_boosting, logistic_regression)
    ├── predictions.py   # get_predictions (predicted probabilities as CSV — enables stacking and error analysis)
    └── evaluation.py    # evaluate_model, rank_experiments
```

## How it's different from a retraining pipeline

| Automated retraining pipeline | mle-bot |
|-------------------------------|---------|
| Someone engineered the pipeline | Agent figures out the approach from your problem description |
| Fixed algorithm and feature strategy | Agent reasons about what to try and explains its choices |
| Runs on a schedule | Runs on demand, iterates until satisfied |
| Black box results | Produces a model card explaining what worked and why |
| Requires ML engineer to set up | Describe the problem, get a model |

## Why Flyte's durable sandbox

- **Safe**: LLM-generated orchestration runs in Monty — no imports, no network, only approved tool calls
- **Durable**: Each tool call is a Flyte task — retryable, observable, traceable
- **Cloud compute**: Tools run on managed infrastructure (GPU, distributed, large memory) — not your laptop
- **Parallelism**: Multiple experiments can run simultaneously on the cluster
