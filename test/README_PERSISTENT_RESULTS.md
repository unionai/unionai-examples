# Persistent Test Results System

This document explains how the persistent test results system works with GitHub Pages.

## Overview

The test runner now maintains historical test results across GitHub Actions runs by using GitHub Pages as persistent storage. This means:

- ✅ **Incremental Testing**: Only test the files you need, previous results are preserved
- 📊 **Complete History**: See all test results with timestamps showing when each script was last tested
- 🌐 **Public Access**: Results are available at `https://unionai.github.io/unionai-examples/`

## How It Works

### 1. **First Run**
```bash
# Creates initial baseline with all tests
uv run python test/test_runner.py v2
```

### 2. **Subsequent Runs**
```bash
# Only tests specific files, preserves other results
uv run python test/test_runner.py v2 --filter "async"
uv run python test/test_runner.py v2 --file "hello_world.py"
```

### 3. **GitHub Actions Integration**
- Downloads previous results from GitHub Pages before testing
- Merges new results with existing ones
- Deploys updated results back to GitHub Pages

## Generated Files

| File | Purpose |
|------|---------|
| `test_report.json` | Current run results only |
| `historical_results.json` | All historical results |
| `persistent_results.json` | Internal storage format |
| `index.html` | Interactive web report |

## GitHub Actions Workflow

The workflow supports several trigger modes:

### Manual Runs
```yaml
# Test all files
gh workflow run test-and-deploy.yml

# Test with filter
gh workflow run test-and-deploy.yml -f test_filter="async"

# Test specific file
gh workflow run test-and-deploy.yml -f test_file="hello_world.py"
```

### Automatic Runs
- **Push to main**: Full test suite
- **Pull Request**: Full test suite with PR comment
- **Daily**: Scheduled full test at 6 AM UTC

## Key Features

### 🔄 **Intelligent Merging**
- Only files tested in current run get updated
- Untested files keep their previous results and timestamps
- No data loss between runs

### 📅 **Timestamp Tracking**
- Each result includes when it was last tested
- Displayed in local time in the HTML report
- Sortable by recency

### 🌐 **GitHub Pages Integration**
- Automatic deployment on main branch
- Previous results downloaded before each run
- No manual setup required after first deployment

### 📊 **Rich Reporting**
- Interactive HTML with expandable details
- Links to execution logs and GitHub sources
- Status badges and visual indicators

## Local Development

### Run Tests Locally
```bash
# Preview what would be tested
uv run python test/test_runner.py v2 --preview

# Run specific tests
uv run python test/test_runner.py v2 --filter "basic"

# Run with local execution (faster for development)
uv run python test/test_runner.py v2 --local --filter "async"
```

### Generate Reports Only
```bash
# Test a few files and generate reports
uv run python test/test_runner.py v2 --file "async.py"
open test/reports/index.html
```

## CI/CD Integration

The system automatically detects GitHub Actions environment and:

1. **Downloads** previous results from GitHub Pages
2. **Runs** only the specified tests
3. **Merges** results with existing data
4. **Deploys** updated results to GitHub Pages
5. **Comments** on PRs with test summary

## Troubleshooting

### No Previous Results Found
```
📋 No previous results found on GitHub Pages (404) - starting fresh
```
This is normal on the first run or if GitHub Pages isn't set up yet.

### Network Errors
```
⚠️  Network error downloading results: [Errno -2] Name or service not known
```
Check internet connection and GitHub Pages URL.

### Merge Conflicts
The system is designed to avoid conflicts by using timestamps and script paths as unique keys.

## URLs

Once deployed, results are available at:

- **Main Report**: https://unionai.github.io/unionai-examples/
- **Current Results**: https://unionai.github.io/unionai-examples/test_report.json
- **All Historical**: https://unionai.github.io/unionai-examples/historical_results.json
- **Raw Data**: https://unionai.github.io/unionai-examples/persistent_results.json