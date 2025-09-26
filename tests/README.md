# Testing Framework for unionai-examples

This testing framework automatically discovers and runs all Python example scripts in the repository that have a `if __name__ == "__main__":` guard.

## Features

- 🔍 **Auto-discovery**: Automatically finds runnable Python scripts
- ⏱️ **Timeout support**: Prevents long-running scripts from hanging tests
- 🎯 **Smart filtering**: Detects Flyte vs regular Python scripts
- 📊 **Rich reporting**: Generates HTML and JSON reports
- 🔧 **Configurable**: Supports configuration files and command-line options
- 🚀 **Environment detection**: Skips tests that require missing secrets or config

## Quick Start

```bash
# Run all tests in v2 directory (default)
make test

# Run tests in a specific subdirectory
make test DIR=tutorials

# Run with shorter timeout for quick feedback
make test-quick DIR=integrations

# Run only Flyte examples in specific directory
make test-flyte DIR=v2

# Run specific pattern
make test-filter DIR=v2 FILTER=hello

# See what would be run without executing
make test-dry-run DIR=tutorials

# Clean test results
make clean
```

## Manual Usage

```bash
# Basic usage - scan v2 directory by default
python3 tests/test_runner.py

# Scan a specific subdirectory
python3 tests/test_runner.py tutorials

# Scan integrations directory with custom config
python3 tests/test_runner.py integrations --config tests/config.json

# Filter by pattern in specific directory
python3 tests/test_runner.py v2 --filter "hello"

# Custom timeout and directory
python3 tests/test_runner.py tutorials --timeout 120

# Dry run to see what would be executed
python3 tests/test_runner.py v2 --dry-run

# Custom root directory (if running from outside repo)
python3 tests/test_runner.py v2 --root /path/to/unionai-examples
```

## Configuration

The test framework can be configured via `tests/config.json`:

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

### Configuration Options

- `timeout`: Maximum time to wait for each script (seconds)
- `excluded_patterns`: Patterns to skip when discovering scripts
- `required_env_vars`: Environment variables to set for all scripts

## Test Categories

The framework automatically categorizes scripts:

- 🚀 **Flyte scripts**: Scripts that import flyte modules
- 🐍 **Regular Python scripts**: Standard Python scripts

## Smart Skipping

Scripts are automatically skipped if they:
- Require secrets but no secret environment variables are found
- Need `config.yaml` but no config file exists in expected locations
- Match excluded patterns

## Reports

After running tests, find reports in `tests/logs/`:

- `test_report.html`: Interactive HTML report
- `test_report.json`: Machine-readable JSON report
- Individual `.log` files for each script

## Environment Variables

Set these for better test coverage:

```bash
# For scripts requiring API keys or secrets
export FLYTE_CLIENT_SECRET="your-secret"
export OPENAI_API_KEY="your-key"

# For custom endpoints
export FLYTE_ENDPOINT="https://your-flyte-instance.com"
```

## Directory Structure

```
tests/
├── test_runner.py      # Main test framework
├── config.json         # Test configuration
├── logs/              # Test results and reports
│   ├── test_report.html
│   ├── test_report.json
│   └── *.log          # Individual script logs
└── README.md          # This file
```