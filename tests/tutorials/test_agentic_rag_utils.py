"""Tests for the agentic_rag.utils module."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Mock the flytekit module
sys.modules['flytekit'] = MagicMock()

# Now import the module under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from tutorials.agentic_rag.utils import env_secret, openai_env_secret


@pytest.mark.unit
def test_env_secret_as_decorator():
    """Test env_secret when used as a decorator with explicit function."""
    # Setup mocked context and secrets
    mock_context = MagicMock()
    mock_context.return_value.secrets.get.return_value = "test-secret-value"
    
    # Clear any existing env var to ensure test isolation
    if "TEST_ENV_VAR" in os.environ:
        del os.environ["TEST_ENV_VAR"]
    
    # Apply decorator to test function
    @patch("tutorials.agentic_rag.utils.current_context", mock_context)
    @env_secret(secret_name="test-secret", env_var="TEST_ENV_VAR")
    def test_function():
        return os.environ.get("TEST_ENV_VAR")
    
    # Verify decorator functionality
    result = test_function()
    
    # Assertions
    assert result == "test-secret-value"
    assert os.environ.get("TEST_ENV_VAR") == "test-secret-value"
    mock_context.return_value.secrets.get.assert_called_once_with(key="test-secret")


@pytest.mark.unit
def test_env_secret_as_factory():
    """Test env_secret when used as a decorator factory."""
    # Setup mocked context and secrets
    mock_context = MagicMock()
    mock_context.return_value.secrets.get.return_value = "factory-secret-value"
    
    # Clear any existing env var to ensure test isolation
    if "FACTORY_ENV_VAR" in os.environ:
        del os.environ["FACTORY_ENV_VAR"]
    
    # Create and apply decorator via factory pattern
    factory_decorator = env_secret(secret_name="factory-secret", env_var="FACTORY_ENV_VAR")
    
    @patch("tutorials.agentic_rag.utils.current_context", mock_context)
    @factory_decorator
    def test_function():
        return os.environ.get("FACTORY_ENV_VAR")
    
    # Verify decorator functionality
    result = test_function()
    
    # Assertions
    assert result == "factory-secret-value"
    assert os.environ.get("FACTORY_ENV_VAR") == "factory-secret-value"
    mock_context.return_value.secrets.get.assert_called_once_with(key="factory-secret")


@pytest.mark.unit
def test_openai_env_secret():
    """Test the openai_env_secret partial function."""
    # Setup mocked context and secrets
    mock_context = MagicMock()
    mock_context.return_value.secrets.get.return_value = "openai-api-key-value"
    
    # Clear any existing env var to ensure test isolation
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    
    # Apply decorator to test function
    @patch("tutorials.agentic_rag.utils.current_context", mock_context)
    @openai_env_secret
    def test_function():
        return os.environ.get("OPENAI_API_KEY")
    
    # Verify decorator functionality
    result = test_function()
    
    # Assertions
    assert result == "openai-api-key-value"
    assert os.environ.get("OPENAI_API_KEY") == "openai-api-key-value"
    mock_context.return_value.secrets.get.assert_called_once_with(key="openai_api_key")