from typing import Tuple

import pandas as pd
from flytekit import kwtypes, task, workflow, Secret, Deck
from flytekit.deck import MarkdownRenderer
from flytekitplugins.duckdb import DuckDBQuery, DuckDBProvider
from plotly.subplots import make_subplots

from _blogs.motherduck.plots import make_elasticity_plot, make_sales_trend_plot, create_transactions_plot, \
    create_spend_plot
from _blogs.motherduck.queries import sales_trends_query, elasticity_query, customer_segmentation_query

sales_trends_query_task = DuckDBQuery(
    name="sales_trends_query",
    query=sales_trends_query,
    inputs=kwtypes(mydf=pd.DataFrame),
    provider=DuckDBProvider.MOTHERDUCK,
    hosted_secret=Secret(key="md_token", group="1"),
)

elasticity_query_task = DuckDBQuery(
    name="elasticity_query",
    query=elasticity_query,
    inputs=kwtypes(mydf=pd.DataFrame),
    provider=DuckDBProvider.MOTHERDUCK,
    hosted_secret=Secret(key="md_token", group="1"),
)

customer_segmentation_query_task = DuckDBQuery(
    name="customer_segmentation_query",
    query=customer_segmentation_query,
    inputs=kwtypes(mydf=pd.DataFrame),
    provider=DuckDBProvider.MOTHERDUCK,
    hosted_secret=Secret(key="md_token", group="1"),
)

@task
def get_pandas_df() -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv('/Users/danielsola/repos/unionai-examples/Year 2010-2011-Table 1.csv')
    df['dt'] = pd.to_datetime(df['InvoiceDate'])

    # Find the oldest date in the dataset
    oldest_date = df['dt'].min()

    # Filter for data from the oldest month
    start_of_oldest_month = oldest_date.replace(day=1)
    end_of_oldest_month = (start_of_oldest_month + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
    oldest_month_data = df[(df['dt'] >= start_of_oldest_month) & (df['dt'] <= end_of_oldest_month)]

    equal_time_before_start = (oldest_month_data['dt'].min() - (oldest_month_data['dt'].max() - oldest_month_data['dt'].min())).strftime('%Y-%m-%d %H:%M:%S')

    return oldest_month_data.drop(columns=['dt']), equal_time_before_start

@task(enable_deck=True)
def query_result_report(
        sales_trends_result: pd.DataFrame,
        elasticity_result: pd.DataFrame,
        customer_segmentation_result: pd.DataFrame,
):
    from plotly.subplots import make_subplots
    import plotly.io as pio

    # Create the figure with 4 rows and 1 column
    fig = make_subplots(rows=4, cols=1, shared_xaxes=False, vertical_spacing=0.22,
                        subplot_titles=("Higheset and Lowest Avg Change in Order Quantity by Product",
                                        "Top Price Elasticity By Product",
                                        "Customers with Largest Change in Transactions",
                                        "Customers with Largest Change in Spend"))

    # Add the Sales Trends subplot (row 1)
    sales_trends_trace = make_sales_trend_plot(sales_trends_result)
    fig.add_trace(sales_trends_trace, row=1, col=1)
    fig.update_yaxes(title_text="Avg Change in Quantity", row=1, col=1)
    fig.update_xaxes(title_text="Product Code - Description", row=1, col=1)

    # Add the Elasticity subplot (row 2)
    trace_recent, trace_historical = make_elasticity_plot(elasticity_result)
    fig.add_trace(trace_recent, row=2, col=1)
    fig.add_trace(trace_historical, row=2, col=1)
    fig.update_yaxes(title_text="Elasticity Coefficient", row=2, col=1)
    fig.update_xaxes(title_text="Product Code - Description", row=2, col=1)

    # Add the Transactions subplot (row 3)
    trace_recent_transactions, trace_historical_transactions = create_transactions_plot(customer_segmentation_result)
    fig.add_trace(trace_recent_transactions, row=3, col=1)
    fig.add_trace(trace_historical_transactions, row=3, col=1)
    fig.update_yaxes(title_text="Number of Transactions", row=3, col=1)
    fig.update_xaxes(title_text="Customer ID", row=3, col=1)

    # Add the Spend subplot (row 4)
    trace_recent_spend, trace_historical_spend = create_spend_plot(customer_segmentation_result)
    fig.add_trace(trace_recent_spend, row=4, col=1)
    fig.add_trace(trace_historical_spend, row=4, col=1)
    fig.update_yaxes(title_text="Spend ($)", row=4, col=1)
    fig.update_xaxes(title_text="Customer ID", row=4, col=1)

    # Update layout to group bars for the elasticity subplot
    fig.update_layout(barmode='group', showlegend=False,
                      xaxis_showticklabels=True,  # Show x-axis labels for all plots
                      xaxis2_showticklabels=True,  # X-axis labels for the second plot
                      xaxis3_showticklabels=True,  # X-axis labels for the third plot
                      xaxis4_showticklabels=True  # X-axis labels for the fourth plot
                      )

    # Rotate the x-axis labels for better readability if needed
    fig.update_xaxes(tickangle=-15)
    main_deck = Deck("Ecommerce Report", MarkdownRenderer().to_html(""))
    main_deck.append(pio.to_html(fig))


@workflow
def wf() -> list[pd.DataFrame]:
    recent_data, equal_time_before_start = get_pandas_df()
    sales_trends_result = sales_trends_query_task(mydf=recent_data)
    elasticity_result = elasticity_query_task(mydf=recent_data)
    customer_segmentation_result = customer_segmentation_query_task(mydf=recent_data)
    query_result_report(
        sales_trends_result=sales_trends_result,
        elasticity_result=elasticity_result,
        customer_segmentation_result=customer_segmentation_result,
    )



if __name__ == "__main__":
    out = wf()

    sales_trends_result = out[0]
    elasticity_result = out[1]
    outlier_result = out[2]
    customer_segmentation_result = out[3]

    # Create the figure with 4 rows and 1 column
    fig = make_subplots(rows=4, cols=1, shared_xaxes=False, vertical_spacing=0.22,
                        subplot_titles=("Higheset and Lowest Avg Change in Order Quantity by Product",
                                        "Top Price Elasticity By Product",
                                        "Customers with Largest Change in Transactions",
                                        "Customers with Largest Change in Spend"))

    # Add the Sales Trends subplot (row 1)
    sales_trends_trace = make_sales_trend_plot(sales_trends_result)
    fig.add_trace(sales_trends_trace, row=1, col=1)
    fig.update_yaxes(title_text="Avg Change in Quantity", row=1, col=1)
    fig.update_xaxes(title_text="Product Code - Description", row=1, col=1)

    # Add the Elasticity subplot (row 2)
    trace_recent, trace_historical = make_elasticity_plot(elasticity_result)
    fig.add_trace(trace_recent, row=2, col=1)
    fig.add_trace(trace_historical, row=2, col=1)
    fig.update_yaxes(title_text="Elasticity Coefficient", row=2, col=1)
    fig.update_xaxes(title_text="Product Code - Description", row=2, col=1)

    # Add the Transactions subplot (row 3)
    trace_recent_transactions, trace_historical_transactions = create_transactions_plot(customer_segmentation_result)
    fig.add_trace(trace_recent_transactions, row=3, col=1)
    fig.add_trace(trace_historical_transactions, row=3, col=1)
    fig.update_yaxes(title_text="Number of Transactions", row=3, col=1)
    fig.update_xaxes(title_text="Customer ID", row=3, col=1)

    # Add the Spend subplot (row 4)
    trace_recent_spend, trace_historical_spend = create_spend_plot(customer_segmentation_result)
    fig.add_trace(trace_recent_spend, row=4, col=1)
    fig.add_trace(trace_historical_spend, row=4, col=1)
    fig.update_yaxes(title_text="Spend ($)", row=4, col=1)
    fig.update_xaxes(title_text="Customer ID", row=4, col=1)

    # Update layout to group bars for the elasticity subplot
    fig.update_layout(barmode='group', title_text="Query Summary", showlegend=False,
                      xaxis_showticklabels=True,  # Show x-axis labels for all plots
                      xaxis2_showticklabels=True,  # X-axis labels for the second plot
                      xaxis3_showticklabels=True,  # X-axis labels for the third plot
                      xaxis4_showticklabels=True  # X-axis labels for the fourth plot
                      )

    # Rotate the x-axis labels for better readability if needed
    fig.update_xaxes(tickangle=-15)

    # Show the figure
    fig.show()

    print(out.head())