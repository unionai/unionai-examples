import json
from typing import Tuple, Optional, Union

import duckdb
import pandas as pd
from flytekit import kwtypes, task, workflow, Secret, Deck, ImageSpec, dynamic, current_context
from flytekit.deck import MarkdownRenderer
from flytekitplugins.duckdb import DuckDBQuery, DuckDBProvider
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from duckdb_artifacts import RecentEcommerceData
from openai_tools import get_tools, GPT_MODEL, DUCKDB_FUNCTION_NAME
from plots import make_elasticity_plot, make_sales_trend_plot, create_transactions_plot, \
    create_spend_plot
from queries import sales_trends_query, elasticity_query, customer_segmentation_query

image = ImageSpec(
    name="test-image",
    # registry=os.environ.get("DOCKER_REGISTRY", None),
    registry="ghcr.io/dansola",
    apt_packages=["git"],
    packages=["pandas==2.2.2", "plotly==5.23.0", "pyarrow==16.1.0", "openai==1.41.0", "flytekitplugins-openai"],#, "git+https://github.com/flyteorg/flytekit.git@6d5f81e81844c1a0bad7ba6614537c9afa8a4ff6#subdirectory=plugins/flytekit-duckdb"],
    # commands=["pip install git+https://github.com/flyteorg/flytekit.git@ff85ef933be0c16b37f85d1025749471041ba5b9#subdirectory=plugins/flytekit-duckdb"]
    commands=["pip install git+https://github.com/flyteorg/flytekit.git@f58107179be2171b55cae61b0df3d29633cb70f0#subdirectory=plugins/flytekit-duckdb"]
)

sales_trends_query_task = DuckDBQuery(
    name="sales_trends_query",
    query=sales_trends_query,
    inputs=kwtypes(mydf=pd.DataFrame),
    provider=DuckDBProvider.MOTHERDUCK,
    secret_requests=[Secret(group=None, key="md_token")],
    container_image=image,
)

elasticity_query_task = DuckDBQuery(
    name="elasticity_query",
    query=elasticity_query,
    inputs=kwtypes(mydf=pd.DataFrame),
    provider=DuckDBProvider.MOTHERDUCK,
    secret_requests=[Secret(group=None, key="md_token")],
    container_image=image,
)

customer_segmentation_query_task = DuckDBQuery(
    name="customer_segmentation_query",
    query=customer_segmentation_query,
    inputs=kwtypes(mydf=pd.DataFrame),
    provider=DuckDBProvider.MOTHERDUCK,
    secret_requests=[Secret(group=None, key="md_token")],
    container_image=image,
)

prompt_query_task = DuckDBQuery(
    name="prompt_query",
    inputs=kwtypes(query=str, mydf=pd.DataFrame),
    provider=DuckDBProvider.MOTHERDUCK,
    secret_requests=[Secret(group=None, key="md_token")],
    container_image=image,
)

@task(container_image=image, enable_deck=True)
def query_result_report(
        sales_trends_result: pd.DataFrame,
        elasticity_result: pd.DataFrame,
        customer_segmentation_result: pd.DataFrame,
):
    from plotly.subplots import make_subplots
    import plotly.io as pio
    import plotly.graph_objects as go

    main_deck = Deck("Ecommerce Report", MarkdownRenderer().to_html(""))

    # Add the Sales Trends plot
    sales_trends_trace = make_sales_trend_plot(sales_trends_result)
    fig_sales_trends = go.Figure(sales_trends_trace)
    fig_sales_trends.update_yaxes(title_text="Avg Change in Quantity")
    fig_sales_trends.update_xaxes(title_text="Product Code - Description", tickangle=-25)
    fig_sales_trends.update_layout(title_text="Highest and Lowest Avg Change in Order Quantity by Product")
    main_deck.append(pio.to_html(fig_sales_trends))

    # Add the Elasticity plot
    trace_recent, trace_historical = make_elasticity_plot(elasticity_result)
    fig_elasticity = go.Figure(data=[trace_recent, trace_historical])
    fig_elasticity.update_yaxes(title_text="Elasticity Coefficient")
    fig_elasticity.update_xaxes(title_text="Product Code - Description", tickangle=-15)
    fig_elasticity.update_layout(title_text="Top Price Elasticity By Product", barmode='group')
    main_deck.append(pio.to_html(fig_elasticity))

    # Add the Transactions plot
    trace_recent_transactions, trace_historical_transactions = create_transactions_plot(customer_segmentation_result)
    fig_transactions = go.Figure(data=[trace_recent_transactions, trace_historical_transactions])
    fig_transactions.update_yaxes(title_text="Number of Transactions")
    fig_transactions.update_xaxes(title_text="Customer ID", tickangle=-15)
    fig_transactions.update_layout(title_text="Customers with Largest Change in Transactions")
    main_deck.append(pio.to_html(fig_transactions))

    # Add the Spend plot
    trace_recent_spend, trace_historical_spend = create_spend_plot(customer_segmentation_result)
    fig_spend = go.Figure(data=[trace_recent_spend, trace_historical_spend])
    fig_spend.update_yaxes(title_text="Spend ($)")
    fig_spend.update_xaxes(title_text="Customer ID", tickangle=-15)
    fig_spend.update_layout(title_text="Customers with Largest Change in Spend")
    main_deck.append(pio.to_html(fig_spend))


@dynamic(container_image=image, secret_requests=[Secret(group=None, key="daniel_openai_key")])
def duckdb_to_openai(messages: list[Union[dict, ChatCompletionMessage]], results: pd.DataFrame, tool_call_id: str, tool_function_name: str) -> str:
    daniel_openai_key = current_context().secrets.get(key="daniel_openai_key")
    openai_client = OpenAI(api_key=daniel_openai_key)

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_function_name,
        "content": results.to_string()
    })

    # Step 4: Invoke the chat completions API with the function response appended to the messages list
    # Note that messages with role 'tool' must be a response to a preceding message with 'tool_calls'
    model_response_with_function_call = openai_client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
    )  # get a new response from the model where it can see the function response
    return model_response_with_function_call.choices[0].message.content

@dynamic(container_image=image, secret_requests=[Secret(group=None, key="md_token"), Secret(group=None, key="daniel_openai_key")])
def check_prompt(recent_data: pd.DataFrame, prompt: Optional[str]) -> Tuple[str, str]:
    if prompt is None:
        return "No prompt was provided.", "No query."
    # set up clients
    motherduck_token = current_context().secrets.get(key="md_token")
    daniel_openai_key = current_context().secrets.get(key="daniel_openai_key")
    con = duckdb.connect("md:", config={"motherduck_token": motherduck_token})
    openai_client = OpenAI(api_key=daniel_openai_key)

    tools = get_tools(con=con)

    messages = [{
        "role": "user",
        "content": f"{prompt}"
    }]

    response = openai_client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    # Append the message to messages list
    response_message = response.choices[0].message
    messages.append(response_message)

    tool_calls = response_message.tool_calls
    if tool_calls:
        # If true the model will return the name of the tool / function to call and the argument(s)
        tool_call_id = tool_calls[0].id
        tool_function_name = tool_calls[0].function.name
        tool_query_string = json.loads(tool_calls[0].function.arguments)['query']

        # Step 3: Call the function and retrieve results. Append the results to the messages list.
        if tool_function_name == DUCKDB_FUNCTION_NAME:
            results = prompt_query_task(query=tool_query_string, mydf=recent_data)
            content = duckdb_to_openai(messages=messages, results=results, tool_call_id=tool_call_id, tool_function_name=tool_function_name)
            return content, tool_query_string
        else:
            return f"Error: function {tool_function_name} does not exist", "No query."
    else:
        # Model did not identify a function to call, result can be returned to the user
        return response_message.content, "No query."


@workflow
def wf(recent_data: pd.DataFrame = RecentEcommerceData.query(), prompt: Optional[str] = None) -> str:
    # Answer prompt
    answer, query = check_prompt(recent_data=recent_data, prompt=prompt)

    # Make plot
    sales_trends_result = sales_trends_query_task(mydf=recent_data)
    elasticity_result = elasticity_query_task(mydf=recent_data)
    customer_segmentation_result = customer_segmentation_query_task(mydf=recent_data)
    query_result_report(
        sales_trends_result=sales_trends_result,
        elasticity_result=elasticity_result,
        customer_segmentation_result=customer_segmentation_result,
    )
    return answer
