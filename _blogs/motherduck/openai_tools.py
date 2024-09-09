import json

import duckdb

from queries import sales_trends_query

GPT_MODEL = "gpt-4o"
DUCKDB_FUNCTION_NAME = "ask_ecommerce_duckdb"

TABLE_DESCRIPTION = """
Abstract: A real online retail transaction data set of two years. \n

Data Set Information: \n
This Online Retail II data set contains all the transactions occurring for a UK-based and registered, non-store online retail between 01/12/2009 and 09/12/2011.The company mainly sells unique all-occasion gift-ware. Many customers of the company are wholesalers. \n

Attribute Information: \n
Invoice: Invoice number. Nominal. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation. \n
StockCode: Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product. \n
Description: Product (item) name. Nominal. \n
Quantity: The quantities of each product (item) per transaction. Numeric. \n
InvoiceDate: Invice date and time. Numeric. The day and time when a transaction was generated. \n
Price: Unit price. Numeric. Product price per unit in sterling (Â£). \n
Customer ID: Customer number. Nominal. A 5-digit integral number uniquely assigned to each customer. \n
Country: Country name. Nominal. The name of the country where a customer resides. \n
"""


def get_schema(con: duckdb.DuckDBPyConnection):
    con.sql("USE e_commerce")
    tables = con.execute("SHOW TABLES").fetchall()
    # Prepare a dictionary to hold the schema information
    schema_info = {}
    # Iterate over each table and get its schema
    for table in tables:
        table_name = table[0]
        schema = con.execute(f"DESCRIBE {table_name}").fetchall()
        # Format the schema for this table
        schema_info[table_name] = [{"column_name": col[0], "column_type": col[1]} for col in schema]
    # Convert the schema info to JSON format
    schema_json = json.dumps(schema_info, indent=4)
    return schema_json

def get_tools(con: duckdb.DuckDBPyConnection):
    database_schema_string = get_schema(con)

    tools = [
        {
            "type": "function",
            "function": {
                "name": f"{DUCKDB_FUNCTION_NAME}",
                "description": "Use this function to answer user questions about a local ecommerce dataframe, mydf, and a remote ecommerce DuckDB database. Input should be a fully formed DuckDB query. "
                               "You can only use this tool once, so if the user prompt requests information on both the recent mydf data and the historical e_commerce.year_09_10 data, format that in a single query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"""
                                    DuckDB query extracting info to answer the user's question.
                                    This query can use the user provided dataframe called mydf and the remote e_commerce.year_09_10 DuckDB database.
                                    Motherduck hosts the e_commerce.year_09_10 data, therefore the local mydf and the remote e_commerce.year_09_10 can be queried at the same time.
                                    DuckDB query should be written using this database schema:
                                    {database_schema_string}
                                    Use the following description of the table columns:
                                    {TABLE_DESCRIPTION}
                                    Here is an example query:
                                    {sales_trends_query}
                                    The query should be returned in plain text, not in JSON.
                                    """,
                        }
                    },
                    "required": ["query"],
                },
            }
        }
    ]

    return tools

