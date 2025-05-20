import pytest

def test_pytest_setup():
    """Verify pytest is configured correctly"""
    assert True

def test_sample_fixture(sample_data):
    """Test that our fixture is working"""
    assert sample_data["sample_number"] == 42
    assert len(sample_data["sample_list"]) == 5