# Testing for UnionAI Examples

This directory contains tests for the UnionAI examples repository.

## Test Organization

Tests are organized to mirror the repository structure:

- `tests/tutorials/` - Tests for tutorials
- `tests/flyte-integrations/` - Tests for Flyte integrations
- `tests/integrations/` - Tests for general integrations
- `tests/_blogs/` - Tests for blog examples

## Running Tests

To run all tests:

```bash
pytest
```

To run tests with coverage report:

```bash
pytest --cov=. --cov-report=term-missing
```

To run specific test categories:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run tests for a specific module
pytest tests/tutorials/
```

## Writing Tests

- Unit tests should be marked with the `@pytest.mark.unit` decorator
- Integration tests should be marked with the `@pytest.mark.integration` decorator
- Slow tests should be marked with the `@pytest.mark.slow` decorator

Example:

```python
import pytest

@pytest.mark.unit
def test_something():
    # Test code here
    assert True
```