# Testing Guide for unionai-examples

## Overview of Testing Infrastructure

  This repository now includes a comprehensive testing framework built with pytest. The testing infrastructure was designed to validate the functionality of Python modules across the repository while minimizing dependencies on external services.

  What's Included

  1. Basic Testing Framework
    - pytest.ini configuration file with test paths and markers
    - conftest.py with shared fixtures and test utilities
    - Test directory structure mirroring the main project organization
  2. Unit Tests for Key Modules
    - tutorials/agentic_rag/utils.py - Tests for environment secret management
    - _blogs/motherduck/openai_tools.py - Tests for DuckDB schema utilities
    - _blogs/ollama/utils.py - Tests for ML model configuration classes
    - tutorials/max_serve/cache_model.py - Tests for model caching functionality
    - tutorials/time_series_forecasting/forecasters/* - Tests for time series forecasting models
  3. Mocking Strategies
    - Mock implementations for external dependencies (flytekit, torch, prophet, etc.)
    - Test fixtures for common data patterns
    - Isolation strategies to test components without requiring their dependencies
  4. Test Coverage
    - 89% test coverage across tested modules
    - Focus on validating interfaces and critical functionality

  How to Run Tests

  Prerequisites

  Before running the tests, you'll need to set up the testing environment:

  # Clone the repository (if you haven't already)
  git clone https://github.com/unionai/unionai-examples.git
  cd unionai-examples

  # Create a virtual environment (using uv)
  uv venv

  # Activate the virtual environment
  source .venv/bin/activate  # On Linux/macOS
  # or
  .venv\Scripts\activate     # On Windows

  # Install test dependencies
  uv pip install -r tests/requirements-test.txt

  Running All Tests

  To run all tests in the repository:

  pytest

  Running Only Unit Tests

  To run only unit tests (excluding integration or slow tests):

  pytest -m unit

  Running Tests for a Specific Module

  To run tests for a specific module or directory:

  # Test a specific module
  pytest tests/tutorials/agentic_rag

  # Test a specific test file
  pytest tests/tutorials/time_series_forecasting/test_forecasters.py

  # Test a specific test function
  pytest tests/tutorials/time_series_forecasting/test_forecasters.py::test_prophet_forecaster

  Running Tests with Coverage Reports

  To run tests and generate a coverage report:

  pytest --cov=. --cov-report=term-missing

  Writing New Tests

  Test Organization

  Tests are organized to mirror the repository structure:

  - tests/tutorials/ - Tests for tutorial modules
  - tests/flyte-integrations/ - Tests for Flyte integrations
  - tests/integrations/ - Tests for other integrations
  - tests/_blogs/ - Tests for blog examples

  Test Markers

  Use these markers to categorize your tests:

  @pytest.mark.unit          # Fast, isolated unit tests
  @pytest.mark.integration   # Tests requiring multiple components
  @pytest.mark.slow          # Tests that take a long time to run

  Mocking External Dependencies

  When writing tests for modules with external dependencies:

  1. Use the existing mock implementations when possible
  2. For new dependencies, implement appropriate mocks in the test file
  3. Use patch or monkeypatch to temporarily replace dependencies during tests

  Example of mocking an external dependency:

  # Mock a class
  with patch('module.path.ClassName', MockImplementation):
      # Test code that uses the class

  # Mock a function
  with patch('module.path.function_name', mock_function):
      # Test code that calls the function

  Test Fixtures

  Use the fixtures in conftest.py for common test data and setup:

  def test_something(sample_data):
      # The sample_data fixture is defined in conftest.py
      assert sample_data["sample_number"] == 42

  Troubleshooting Tests

  Missing Dependencies

  If you encounter an import error, you may need to install additional dependencies:

  uv pip install <package-name>

  Mock Issues

  If your test fails because of a mocking issue, check:

  1. That you're mocking the correct module path
  2. That your mock implementation correctly mimics the behavior needed for the test
  3. That you're using the appropriate patching method for the context

  Test Directory Structure

  If pytest can't find your tests, ensure:

  1. Test files are named with a test_ prefix
  2. Test functions have a test_ prefix
  3. Test directories contain an __init__.py file

  Future Improvements

  The following improvements could be made to the testing infrastructure:

  1. Add more tests for remaining modules
  2. Create integration tests that test workflows end-to-end
  3. Add performance tests for computationally intensive operations
  4. Implement GitHub Actions CI to run tests automatically
  5. Add pre-commit hooks for running tests on commit

  Feel free to contribute to these improvements!