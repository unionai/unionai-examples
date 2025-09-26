# GitHub Actions Integration for unionai-examples Testing

This repository includes automated testing via GitHub Actions that runs all example scripts to ensure they work correctly.

## Workflow Overview

The GitHub Actions workflow (`.github/workflows/test-examples.yml`) automatically:

- 🚀 Runs on every push and pull request to main/develop branches
- 📅 Runs daily at 2 AM UTC to catch issues early
- 🐍 Tests against Python 3.11 and 3.12
- 📂 Tests different subdirectories (v2, tutorials, integrations, etc.)
- 📊 Generates HTML and JSON reports
- 💾 Stores test results as artifacts for 30 days

## Manual Triggers

You can manually trigger the workflow with custom parameters:

1. Go to the "Actions" tab in GitHub
2. Select "Test Examples" workflow
3. Click "Run workflow"
4. Specify:
   - **Subdirectory**: Which folder to test (v2, tutorials, etc.)
   - **Timeout**: Maximum time per script in seconds
   - **Filter**: Pattern to match specific scripts

## Workflow Configuration

### Triggers
- **Push/PR**: Automatic testing on code changes
- **Schedule**: Daily runs to catch environmental issues
- **Manual**: On-demand testing with custom parameters

### Matrix Strategy
The workflow runs multiple jobs in parallel:
```yaml
strategy:
  matrix:
    python-version: ['3.11', '3.12']
    subdirectory: ['v2']
```

### Artifacts Generated
- `test-results-py{version}-{subdirectory}`: Individual test results
- `test-report-py{version}-{subdirectory}`: HTML/JSON reports
- `combined-test-results`: Aggregated results from all matrix runs

## Environment Variables

The workflow sets these environment variables for tests:
- `PYTHONPATH=.`: Ensures proper Python imports
- `GITHUB_ACTIONS=true`: Identifies CI environment

## Adding New Subdirectories

To test additional subdirectories, update the workflow matrix:

```yaml
strategy:
  matrix:
    python-version: ['3.11', '3.12']
    subdirectory: ['v2', 'tutorials', 'integrations']
```

## Local Development vs CI

| Feature | Local (Makefile) | GitHub Actions |
|---------|------------------|----------------|
| **Purpose** | Quick development testing | Comprehensive CI/CD |
| **Python Versions** | Current environment | 3.11, 3.12 |
| **Timeout** | Configurable (default 300s) | 300s (configurable via input) |
| **Reports** | Local files only | Artifacts + downloads |
| **Triggers** | Manual only | Automatic + manual |

## Status Badges

Add this badge to your README to show test status:

```markdown
![Test Examples](https://github.com/unionai/unionai-examples/workflows/Test%20Examples/badge.svg)
```

## Viewing Results

1. **GitHub UI**: Go to Actions → Test Examples → specific run
2. **Artifacts**: Download test reports and logs
3. **Logs**: View real-time console output during runs

## Failure Handling

- **Individual script failures**: Workflow continues, marked in reports
- **Setup failures**: Entire job fails, investigate dependencies
- **Timeout failures**: Scripts taking too long, adjust timeout or optimize

## Security Considerations

- No secrets are exposed by default
- Scripts requiring API keys will be skipped
- Add repository secrets if needed for authenticated tests:
  - Go to Settings → Secrets and variables → Actions
  - Add secrets like `FLYTE_API_KEY`, `OPENAI_API_KEY`, etc.
  - Reference in workflow: `${{ secrets.SECRET_NAME }}`