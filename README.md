# Union Examples

This is a repository of runnable examples for [Union](https://docs.union.ai). Use it as a reference for learning how to build repeatable and scalable AI/ML workflows.

## Repository Structure

- **`v2/`** - Modern examples using Flyte 2.x (recommended)
- **`v1/`** - Legacy examples for Flyte 1.x compatibility
- **`_blogs/`** - Example code featured in Union blog posts (temprary)
- **`test/`** - Automated testing framework for V2 examples

## Testing Framework (Flyte 2 only)

This repository includes a testing framework that validates **Flyte 2.x example scripts in the `v2/` directory** using **uv** for dependency management. The testing framework is designed specifically for modern Flyte 2.x workflows and does not support legacy v1 examples.

### Test Modes

The testing framework supports three execution modes:

- **Cloud Mode** (`test`) - Executes examples on Union's cloud backend using `uv run`
- **Local Mode** (`test-local`) - Executes examples locally using `flyte run --local`
- **Preview Mode** (`test-preview`) - Shows what would be executed without running anything

### Quick Testing Commands

```bash
# Test all examples (cloud execution)
make test

# Test all examples (local execution)
make test-local

# Preview what would run (no execution)
make test-preview

# Test specific file in cloud
make test FILE=v2/user-guide/getting-started/hello.py

# Test specific file locally
make test-local FILE=v2/user-guide/getting-started/hello.py

# Test examples matching pattern
make test FILTER=user-guide
make test-local FILTER=user-guide

# Development setup
make setup-venv
source ~/.venv/bin/activate
make update-flyte

# Clean test logs, reports, and virtual environments
make clean
```

### Available Make Targets

| Target | Description | Usage |
|--------|-------------|-------|
| `test` | Run tests with cloud execution | `make test [FILE=path] [FILTER=pattern]` |
| `test-local` | Run tests with local execution | `make test-local [FILE=path] [FILTER=pattern]` |
| `test-preview` | Preview tests without execution | `make test-preview [FILE=path] [FILTER=pattern]` |
| `setup-venv` | Create virtual environment with uv | `make setup-venv` |
| `update-flyte` | Update to latest Flyte version | `make update-flyte` |
| `clean` | Clean test logs and reports | `make clean` |

### Testing Parameters

All testing commands support:
- **`FILE`**: Test a specific file (takes precedence over FILTER)
- **`FILTER`**: Test files matching a pattern (e.g., `user-guide`, `trading`)

### Cloud vs Local Testing

**Cloud Mode (`test`)**:
- Uses `uv run` to execute examples with dependencies
- Runs workflows on Union's cloud backend
- Requires valid Union credentials and config
- Best for comprehensive validation

**Local Mode (`test-local`)**:
- Creates **isolated virtual environments** for each test script
- Uses `uv pip install --requirement` to install dependencies from PEP 723 metadata
- Uses `flyte run --local` for local execution
- **Complete test isolation**: No dependency conflicts between scripts
- Runs entirely on your local machine
- Faster feedback, good for development
- Automatic cleanup of virtual environments
- May have limitations with certain features (reports, some integrations)

**Preview Mode (`test-preview`)**:
- Only discovers and lists scripts that would be tested
- No execution, just validation of test setup
- Useful for debugging test discovery issues

### Advanced Usage

**Verbosity Control:**
```bash
# Adjust Flyte verbosity levels
make test-local VERBOSE=v FILE=hello.py    # -v (minimal)
make test-local VERBOSE=vv FILE=hello.py   # -vv (medium)
make test-local VERBOSE=vvv FILE=hello.py  # -vvv (maximum)
make test-local VERBOSE=3 FILE=hello.py    # Same as -vvv
```

**Manual Test Runner Usage:**
```bash
# Direct test runner usage
python3 test/test_runner.py --preview
python3 test/test_runner.py --local --file "v2/user-guide/getting-started/hello.py"
python3 test/test_runner.py --filter "hello" --verbose "vv"
python3 test/test_runner.py --local --filter "user-guide"
```

### Test Reports

After running tests, find outputs in structured directories:
- **`test/reports/`**: Summary reports
  - `test_report.html`: Interactive HTML report
  - `test_report.json`: Machine-readable JSON report
- **`test/logs/`**: Individual execution logs
  - Individual `.log` files for each script
- **`test/venvs/`**: Isolated virtual environments (cleaned up automatically)

### Testing in the cloud

For running the cloud test you need to have a valid Flyte configuration that points to a Flyte backend.

The Flyte backend is configured in the file `test/config.flyte.yaml`.

Currently it is set to point to a Union-internal instance.

To run the cloud tests from your local machine you need a to set the environment variable `FLYTE_CLIENT_SECRET`.

(for the GitHub Actions workflow that runs these tests, this secret is already configured as a repository secret for authentication with the Flyte backend.)

### API Key Secrets for Examples

Some examples require additional API keys (e.g., OpenAI, Tavily, Together.ai, and Finnhub). These can be created as Flyte secrets referenced in the example scripts.

The script `test/create_secrets.sh` can be used to create these secrets in the Flyte backend that is used for testing environment. Make sure to set the corresponding environment variables before running the script.

For the Union-internal test backend these secrets have already been created.

### GitHub Actions Integration

This repository includes automated testing via GitHub Actions.

Currently it uses **manual-only triggers** for controlled testing.

### Manual Workflow Execution

1. Go to the "Actions" tab in GitHub
2. Select "Test Examples" workflow
3. Click "Run workflow"
4. Specify:
   - **Test mode**: `test` (cloud execution), `test-local` (local execution), or `test-preview` (preview only)
   - **Filter**: Pattern to match specific scripts (optional)
   - **Python version**: Choose 3.12 or 3.13 (optional)

### Enabling Automatic Triggers

To enable automatic testing, uncomment the relevant sections in `.github/workflows/test-examples.yml`:

```yaml
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
```
