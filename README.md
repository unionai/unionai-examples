# Union Examples

This is a repository of runnable examples for [Union](https://docs.union.ai). Use it as a reference for learning how to build repeatable and scalable AI/ML workflows.

## Repository Structure

- **`v2/`** - Modern examples using Flyte 2.x (recommended)
- **`v1/`** - Legacy examples for Flyte 1.x compatibility
- **`_blogs/`** - Example code featured in Union blog posts (temprary)
- **`test/`** - Automated testing framework for all examples

## Quick Start

### Flyte 2 (Recommended)

Install the latest Flyte SDK:

```shell
pip install --pre flyte
```

Create a config file:

```shell
flyte create config \
--output ~/.flyte/config.yaml \
--endpoint demo.hosted.unionai.cloud \
--domain flytesnacks \
--project development \
--builder remote
```

Swap out `demo.hosted.unionai.cloud` for your endpoint of choice.
Then run an example from the `v2/` directory:

```shell
uv run v2/user-guide/getting-started/hello.py main
```

### Flyte 1 (Legacy)

Install the `union` SDK:

```shell
pip install union
```

Run an example in the `v1/` directory:

```shell
union run --remote v1/tutorials/sentiment_classifier/sentiment_classifier.py main
```

## Testing Framework (Flyte 2 only)

This repository includes a comprehensive testing framework that validates **Flyte 2.x example scripts in the `v2/` directory** using **uv** for dependency management. The testing framework is designed specifically for modern Flyte 2.x workflows and does not support legacy v1 examples.

### Quick Testing Commands

```bash
# Test all examples (cloud execution)
make test

# Preview what would run (dry-run)
make test-dry-run

# Test specific file
make test FILE=v2/user-guide/getting-started/hello.py

# Test examples matching pattern
make test FILTER=user-guide

# Development setup
make setup-venv
source ~/.venv/bin/activate
make update-flyte

# Clean test logs
make clean
```

### Available Make Targets

| Target | Description | Usage |
|--------|-------------|-------|
| `test` | Run tests with cloud execution | `make test [FILE=path] [FILTER=pattern]` |
| `test-dry-run` | Preview tests without execution | `make test-dry-run [FILE=path] [FILTER=pattern]` |
| `setup-venv` | Create virtual environment with uv | `make setup-venv` |
| `update-flyte` | Update to latest Flyte version | `make update-flyte` |
| `clean` | Clean test logs and reports | `make clean` |

### Testing Parameters

Both `test` and `test-dry-run` support:
- **`FILE`**: Test a specific file (takes precedence over FILTER)
- **`FILTER`**: Test files matching a pattern (e.g., `user-guide`, `trading`)

### Advanced Usage

```bash
# Manual test runner usage
python3 test/test_runner.py --production --dry-run
python3 test/test_runner.py --production --file "v2/user-guide/getting-started/hello.py"
python3 test/test_runner.py --production --filter "hello"
```

### Test Framework Features

- 🔍 **Auto-discovery**: Automatically finds runnable Flyte example scripts
- ⏱️ **Timeout support**: Prevents long-running scripts from hanging tests
- 🎯 **Smart filtering**: Discovers only scripts with `flyte.init` calls
- 📊 **Rich reporting**: Generates HTML and JSON reports
- 🔧 **Configurable**: Supports configuration files and command-line options
- 🚀 **Environment detection**: Skips tests that require missing secrets or config

### Test Reports

After running tests, find reports in `test/logs/`:
- `test_report.html`: Interactive HTML report
- `test_report.json`: Machine-readable JSON report
- Individual `.log` files for each script

## GitHub Actions Integration

This repository includes automated testing via GitHub Actions with **manual-only triggers** for controlled testing.

### Manual Workflow Execution

1. Go to the "Actions" tab in GitHub
2. Select "Test Examples" workflow
3. Click "Run workflow"
4. Specify:
   - **Test mode**: `test` (cloud execution) or `test-dry-run` (preview only)
   - **Filter**: Pattern to match specific scripts (optional)
   - **Python version**: Choose 3.12 or 3.13 (optional)

### Workflow Features

- 🔧 **Manual-only triggers** for controlled testing (automatic triggers disabled)
- 🐍 **Multi-version testing** against Python 3.12 and 3.13
- 📊 **Comprehensive reporting** with HTML and JSON reports
- 💾 **Artifact storage** for test results and logs (90-day retention)
- ⚙️ **Simplified interface** using standardized make targets

### Repository Secret Setup

**Required**: The GitHub Actions workflow requires `FLYTE_CLIENT_SECRET` to be configured as a repository secret for authentication with the Flyte backend.

To set up the secret:
1. Go to your repository's **Settings** tab
2. Navigate to **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `FLYTE_CLIENT_SECRET`
5. Value: Your Flyte client secret for the Union playground
6. Click **Add secret**

Without this secret, the GitHub Actions workflow will fail to authenticate and cannot execute cloud tests.

### Re-enabling Automatic Triggers

To re-enable automatic testing, uncomment the relevant sections in `.github/workflows/test-examples.yml`:

```yaml
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
```

## Contributing

Example code in this repo is maintained by the Union team.

### Testing Your Examples

Before submitting, test your examples using our testing framework:

```bash
# Test all examples
make test

# Preview what would run (dry-run)
make test-dry-run

# Test specific file
make test FILE=path/to/your/example.py

# Test examples in specific area
make test FILTER=tutorials
```

### Development Setup

Before contributing, run:
```bash
pip install -r requirements.txt
pre-commit install
```

### Modern Examples (v2)

The `v2` directory contains the latest examples using Flyte 2.x with current best practices:
- Use modern Flyte 2.x APIs and features
- Include PEP 723 script dependencies for `uv` compatibility
- Are automatically tested via our testing framework
- Support both local development and cloud execution

### Example Structure Guidelines

Tutorial examples should showcase what you can accomplish with Union:
- Training language models, time series models, classification and regression models
- Processing datasets with frameworks like Spark, Polars, etc.
- Scheduling retraining jobs for keeping models up-to-date
- Analyzing, modeling, and visualizing bioinformatics datasets
- Processing image data and other unstructured datasets
- Performing batch inference to generate, process, or analyze data

### Ideal Example Structure

- Example code in directories like `v2/tutorials/timeseries_forecasting`
- Single Python script like `timeseries_forecasting.py`
- Use `jupytext` [`light` format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-light-format)
- Follow [literate programming](https://en.wikipedia.org/wiki/Literate_programming) principles
- Include reasonable default inputs for workflows

> [!NOTE]
> The ideal structure above may not always be possible, but contributors should
> do their best to adhere to these guidelines.

### Blog Examples

> [!NOTE]
> The `_blogs` directory is a temporary space for example code to be used in Union blog posts. Once we've matured the testing and development process in this repo, contributors will ideally start developing example code in the `tutorials` or `guides` directories directly.

### Documentation Integration

Examples that we want to include in the Union documentation can be pulled into the docs build system explicitly in the `docs/sitemap.json` file (see the docs repo [README](https://github.com/unionai/docs/blob/main/README.md) for more details).

For example pages that require instructions on how to run them, the `run_commands.yaml` file needs to be updated like so:

```yaml
<path/to/example.py>:
  - git clone https://github.com/unionai/examples
  - cd examples
  - union run --remote <path/to/example.py> <workflow_name> <input_flags>
```

Adding an entry like the one above will add a dropdown element on the docs example page that tells the user how to run the code. This dropdown element will be inserted after the first Markdown element in the `.py` example file.

## Environment Setup

### Required Environment Variables

Set these for better test coverage:

```bash
# For scripts requiring API keys or secrets
export FLYTE_CLIENT_SECRET="your-secret"
export OPENAI_API_KEY="your-key"

# For custom endpoints
export FLYTE_ENDPOINT="https://your-flyte-instance.com"
```

### Dependencies with uv

The framework uses **uv** for dependency management:

#### Why uv?
- **Performance**: Faster package resolution and installation
- **Reliability**: Better dependency conflict resolution
- **Modern**: Rust-based package manager designed for Python
- **PEP 723 Support**: Native support for inline script metadata

#### Usage in Framework
```bash
# Update Flyte installation
uv pip install --upgrade 'flyte>=2.0.0b0'

# Script execution
uv run script.py
```

## Configuration

### Test Configuration

The test framework can be configured via `test/config.json`:

```json
{
  "timeout": 300,
  "excluded_patterns": [
    "__pycache__",
    ".git",
    "test_"
  ],
  "required_env_vars": {
    "PYTHONPATH": "."
  }
}
```

### Script Discovery

The framework automatically discovers Python scripts in the **`v2/` directory only** that:
- Have `if __name__ == "__main__":` guard
- Contain `flyte.init` calls (indicating they are Flyte 2.x workflows)

Scripts without `flyte.init` are skipped as they are considered utility files. Legacy v1 examples are not included in automated testing.

### Dynamic Configuration

For CI testing, the framework automatically creates temporary `config.yaml` files in each script's directory during execution. This ensures `flyte.init()` can find the configuration needed to connect to the Flyte backend.

The configuration is copied from a template file (`test/config-template.yaml`) that contains hard-coded settings for the Union playground environment. The only environment variable required is:
- `FLYTE_CLIENT_SECRET` - Client secret for authentication (set via GitHub Secrets)

These config files are created before script execution and automatically cleaned up afterward, ensuring no permanent files are left in the repository.## Directory Structure

```
unionai-examples/
├── v2/                     # Modern Flyte 2.x examples (recommended)
│   ├── user-guide/         # User guide examples
│   ├── tutorials/          # End-to-end tutorial examples
│   └── integrations/       # Integration examples
├── v1/                     # Legacy Flyte 1.x examples
├── _blogs/                 # Blog post example code
├── test/                   # Testing framework
│   ├── test_runner.py      # Main test framework
│   ├── config.json         # Test configuration
│   ├── config-template.yaml # Flyte config template for CI
│   ├── logs/              # Test results and reports
│   └── README.md          # Testing documentation
├── .github/
│   └── workflows/
│       └── test-examples.yml  # GitHub Actions workflow
├── Makefile               # Simplified test interface
├── README.md             # This file
└── requirements.txt      # Development dependencies
```

## Status Badge

Add this badge to show test status:

```markdown
![Test Examples](https://github.com/unionai/unionai-examples/workflows/Test%20Examples/badge.svg)
```

---

This simplified framework ensures all Flyte examples are validated and production-ready while maintaining a clean, intuitive interface using modern Python tooling.