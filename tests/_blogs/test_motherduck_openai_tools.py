"""Tests for the motherduck.openai_tools module."""

import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Mock duckdb
sys.modules['duckdb'] = MagicMock()

# Import sales_trends_query
class MockQueries:
    sales_trends_query = "SELECT * FROM mock_table"

sys.modules['queries'] = MockQueries

# Now import the module under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from _blogs.motherduck.openai_tools import get_schema, get_tools, DUCKDB_FUNCTION_NAME, TABLE_DESCRIPTION


@pytest.fixture
def mock_duckdb_connection():
    """Create a mock DuckDB connection for testing."""
    mock_conn = MagicMock()
    
    # Mock the execute and fetchall methods
    mock_execute = MagicMock()
    mock_conn.execute = mock_execute
    
    # Setup return values for show tables
    tables_result = MagicMock()
    tables_result.fetchall.return_value = [("year_09_10",)]
    mock_execute.return_value = tables_result
    
    # Setup return values for describe table
    describe_result = MagicMock()
    describe_result.fetchall.return_value = [
        ("Invoice", "VARCHAR"),
        ("StockCode", "VARCHAR"),
        ("Description", "VARCHAR"),
        ("Quantity", "INTEGER"),
        ("InvoiceDate", "TIMESTAMP"),
        ("Price", "DECIMAL(10,2)"),
        ("Customer ID", "VARCHAR"),
        ("Country", "VARCHAR")
    ]
    
    # Make execute conditionally return different results
    def side_effect(query):
        if query == "SHOW TABLES":
            return tables_result
        elif query.startswith("DESCRIBE"):
            return describe_result
        return MagicMock()
    
    mock_execute.side_effect = side_effect
    
    return mock_conn


@pytest.mark.unit
def test_get_schema(mock_duckdb_connection):
    """Test the get_schema function returns properly formatted JSON."""
    # Call the function with our mock
    schema_json = get_schema(mock_duckdb_connection)
    
    # Parse the JSON to verify its structure
    schema = json.loads(schema_json)
    
    # Assertions
    assert isinstance(schema, dict)
    assert "year_09_10" in schema
    assert len(schema["year_09_10"]) == 8  # 8 columns
    
    # Check a few specific columns
    columns = schema["year_09_10"]
    assert any(col["column_name"] == "Invoice" and col["column_type"] == "VARCHAR" for col in columns)
    assert any(col["column_name"] == "Quantity" and col["column_type"] == "INTEGER" for col in columns)
    
    # Verify SQL calls
    mock_duckdb_connection.sql.assert_called_once_with("USE e_commerce")
    mock_duckdb_connection.execute.assert_any_call("SHOW TABLES")
    mock_duckdb_connection.execute.assert_any_call("DESCRIBE year_09_10")


@pytest.mark.unit
def test_get_tools(mock_duckdb_connection):
    """Test the get_tools function returns correctly structured OpenAI tools."""
    # Call the function with our mock
    tools = get_tools(mock_duckdb_connection)
    
    # Assertions
    assert isinstance(tools, list)
    assert len(tools) == 1
    
    tool = tools[0]
    assert tool["type"] == "function"
    assert "function" in tool
    
    function = tool["function"]
    assert function["name"] == DUCKDB_FUNCTION_NAME
    assert "description" in function
    assert "parameters" in function
    
    # Check parameters structure
    parameters = function["parameters"]
    assert parameters["type"] == "object"
    assert "query" in parameters["properties"]
    assert "description" in parameters["properties"]["query"]
    
    # Verify the query description contains our schema and table description
    query_desc = parameters["properties"]["query"]["description"]
    assert TABLE_DESCRIPTION in query_desc
    
    # Check schema is included in some form 
    # (exact formatting might differ due to mocking)
    assert "year_09_10" in query_desc
    assert "column_name" in query_desc
    assert "column_type" in query_desc
    assert "Invoice" in query_desc