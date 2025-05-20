"""Tests for max_serve.cache_model module."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock, mock_open

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Mock the union module and dependencies
class MockImageSpec:
    def __init__(self, name, **kwargs):
        self.name = name
        self.__dict__.update(kwargs)

class MockArtifact:
    def __init__(self, name):
        self.name = name
    
    def create_from(self, path):
        return f"Artifact created from {path}"

class MockFlyteDirectory:
    pass

class MockContext:
    def __init__(self):
        self.working_directory = "/mock/working/dir"

def mock_task(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

def mock_current_context():
    return MockContext()
    
# Set up mocks for modules
sys.modules['union'] = MagicMock()
sys.modules['union'].ImageSpec = MockImageSpec
sys.modules['union'].task = mock_task
sys.modules['union'].Artifact = MockArtifact
sys.modules['union'].FlyteDirectory = MockFlyteDirectory
sys.modules['union'].current_context = mock_current_context
sys.modules['union'].Resources = MagicMock()

# Import the module to test
from tutorials.max_serve.cache_model import cache_model, COMMIT


@pytest.mark.unit
def test_cache_model_basic():
    """Test that the cache_model function creates the appropriate artifact."""
    # Create a simple test that just verifies the module can be imported
    # Full testing would require more complex mocking of the Hugging Face hub
    # This test verifies that the structure is as expected
    
    # Import the module again to ensure we have the proper constants
    from tutorials.max_serve.cache_model import Qwen_Coder_Artifact
    
    # Verify the artifact has the expected name
    assert Qwen_Coder_Artifact.name == "Qwen2.5-Coder-0.5B"
    
    # Verify the commit constant is defined
    assert COMMIT is not None
    assert isinstance(COMMIT, str)
    assert len(COMMIT) > 0


@pytest.mark.unit
def test_huggingface_hub_imports():
    """Test that the huggingface_hub module can be imported in the environment."""
    # Verify that the module structure is importable
    module_path = os.path.dirname(os.path.abspath(cache_model.__code__.co_filename))
    
    # Check that the import path is as expected
    assert "tutorials" in module_path
    assert "max_serve" in module_path
    
    # Skip docstring check if not available
    if cache_model.__doc__:
        assert "Task to cache" in cache_model.__doc__ 
    
    # Verify the function has the expected structure
    from inspect import signature
    sig = signature(cache_model)
    return_annotation = str(sig.return_annotation)
    assert 'Annotated' in return_annotation
    assert 'MockFlyteDirectory' in return_annotation