import os
import sys
import pytest

# Add parent directory to path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define shared fixtures for tests
@pytest.fixture
def sample_data():
    """Fixture providing sample data for tests"""
    return {
        "sample_text": "This is sample text for testing",
        "sample_number": 42,
        "sample_list": [1, 2, 3, 4, 5]
    }