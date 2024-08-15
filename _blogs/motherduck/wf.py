from typing import Tuple

import pandas as pd
import plotly
from flytekit import kwtypes, task, workflow, Secret, Deck
from flytekit.deck import MarkdownRenderer
from flytekitplugins.duckdb import DuckDBQuery, DuckDBProvider

from _blogs.motherduck.queries import sales_trends_query, elasticity_query, outlier_query, customer_segmentation_query

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

outlier_query_task = DuckDBQuery(
    name="outlier_query",
    query=outlier_query,
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
        outlier_result: pd.DataFrame,
        customer_segmentation_result: pd.DataFrame,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    # Create subplots with 2 rows and 2 columns
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Sales Trends", "Price Elasticity", "Outliers", "Customer Segmentation"),
                        vertical_spacing=0.15)

    # Sales Trends
    fig.add_trace(go.Scatter(x=sales_trends_result['StockCode'],
                             y=sales_trends_result['Avg_Quantity_Historical'],
                             mode='lines+markers',
                             name='Historical Avg Quantity',
                             marker=dict(color='blue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=sales_trends_result['StockCode'],
                             y=sales_trends_result['Avg_Quantity_Recent'],
                             mode='lines+markers',
                             name='Recent Avg Quantity',
                             marker=dict(color='red')),
                  row=1, col=1)

    # Elasticity
    fig.add_trace(go.Bar(x=elasticity_result['StockCode'],
                         y=elasticity_result['Elasticity_Change'],
                         name='Elasticity Change',
                         marker=dict(color='orange')),
                  row=1, col=2)
    fig.add_trace(go.Bar(x=elasticity_result['StockCode'],
                         y=elasticity_result['Recent_Price_Elasticity'],
                         name='Recent Price Elasticity',
                         marker=dict(color='cyan')),
                  row=1, col=2)
    fig.add_trace(go.Bar(x=elasticity_result['StockCode'],
                         y=elasticity_result['Historical_Price_Elasticity'],
                         name='Historical Price Elasticity',
                         marker=dict(color='magenta')),
                  row=1, col=2)

    # Outliers
    fig.add_trace(go.Bar(x=outlier_result['StockCode'],
                         y=outlier_result['Deviation'],
                         name='Deviation',
                         marker=dict(color='purple')),
                  row=2, col=1)

    # Customer Segmentation
    fig.add_trace(go.Bar(x=customer_segmentation_result['Customer ID'],
                         y=customer_segmentation_result['Spend_Percentage'],
                         name='Spend Percentage',
                         marker=dict(color='green')),
                  row=2, col=2)
    fig.add_trace(go.Bar(x=customer_segmentation_result['Customer ID'],
                         y=customer_segmentation_result['Recent_Spend'],
                         name='Recent Spend',
                         marker=dict(color='blue')),
                  row=2, col=2)
    fig.add_trace(go.Bar(x=customer_segmentation_result['Customer ID'],
                         y=customer_segmentation_result['Historical_Spend'],
                         name='Historical Spend',
                         marker=dict(color='red')),
                  row=2, col=2)

    # Update layout
    fig.update_layout(title_text='Ecommerce Data Report',
                      height=800,
                      showlegend=True,
                      barmode='group',
                      template='plotly_white')

    main_deck = Deck("Ecommerce Report", MarkdownRenderer().to_html(""))
    main_deck.append(pio.to_html(fig))


@workflow
def wf(): # -> list[pd.DataFrame]:
    recent_data, equal_time_before_start = get_pandas_df()
    sales_trends_result = sales_trends_query_task(mydf=recent_data)
    elasticity_result = elasticity_query_task(mydf=recent_data)
    outlier_result = outlier_query_task(mydf=recent_data)
    customer_segmentation_result = customer_segmentation_query_task(mydf=recent_data)
    # return [sales_trends_result, elasticity_result, outlier_result, customer_segmentation_result]
    query_result_report(
        sales_trends_result=sales_trends_result,
        elasticity_result=elasticity_result,
        outlier_result=outlier_result,
        customer_segmentation_result=customer_segmentation_result,
    )



if __name__ == "__main__":
    out = wf()
    print(out.head())