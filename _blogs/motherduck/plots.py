import pandas as pd
import plotly.graph_objs as go

def make_sales_trend_plot(sales_trends_result, top_n=5):
    # Calculate the difference in average quantity
    sales_trends_result['Avg_Quantity_Diff'] = (
            sales_trends_result['Avg_Quantity_Recent'] - sales_trends_result['Avg_Quantity_Historical']
    )

    # Sort the DataFrame to get top 3 and bottom 3 items
    top = sales_trends_result.nlargest(top_n, 'Avg_Quantity_Diff')
    bottom = sales_trends_result.nsmallest(top_n, 'Avg_Quantity_Diff')

    # Combine top 3 and bottom 3
    combined = pd.concat([top, bottom])

    # Create the bar chart
    trace1 = go.Bar(
        x=combined['StockCode'] + ' - ' + combined['Description'],
        y=combined['Avg_Quantity_Diff'],
        text=combined['Description'],
        # name='Difference in Average Orders',
        marker_color=combined['Avg_Quantity_Diff'].apply(lambda x: 'green' if x > 0 else 'red')
    )

    return trace1

def make_elasticity_plot(elasticity_result, top_n=10):
    top = elasticity_result.nlargest(top_n, 'Elasticity_Change')

    # Create traces for recent and historical price elasticity
    trace_recent = go.Bar(
        x=top['StockCode'] + ' - ' + top['Description'],
        y=top['Recent_Price_Elasticity'],
        # name='Recent Price Elasticity',
        marker_color='blue'
    )

    trace_historical = go.Bar(
        x=top['StockCode'] + ' - ' + top['Description'],
        y=top['Historical_Price_Elasticity'],
        # name='Historical Price Elasticity',
        marker_color='orange'
    )

    return trace_recent, trace_historical


def create_transactions_plot(customer_segmentation_result, top_n=5):
    # Sort the DataFrame to get the top N customers based on transaction percentage
    top_n_transactions = customer_segmentation_result.nlargest(top_n, 'Transaction_Percentage')

    # Combine Customer ID with a string format for better labeling
    x_labels = top_n_transactions['Customer ID'].astype(str)

    # Create traces for recent and historical transactions
    trace_recent_transactions = go.Bar(
        x=x_labels,
        y=top_n_transactions['Recent_Transactions'],
        # name='Recent Transactions',
        marker_color='blue'
    )

    trace_historical_transactions = go.Bar(
        x=x_labels,
        y=top_n_transactions['Historical_Transactions'],
        # name='Historical Transactions',
        marker_color='orange'
    )

    return trace_recent_transactions, trace_historical_transactions


def create_spend_plot(customer_segmentation_result, top_n=5):
    # Sort the DataFrame to get the top N customers based on spend percentage
    top_n_spend = customer_segmentation_result.nlargest(top_n, 'Spend_Percentage')

    # Combine Customer ID with a string format for better labeling
    x_labels = top_n_spend['Customer ID'].astype(str)

    # Create traces for recent and historical spend
    trace_recent_spend = go.Bar(
        x=x_labels,
        y=top_n_spend['Recent_Spend'],
        # name='Recent Spend',
        marker_color='blue'
    )

    trace_historical_spend = go.Bar(
        x=x_labels,
        y=top_n_spend['Historical_Spend'],
        # name='Historical Spend',
        marker_color='orange'
    )

    return trace_recent_spend, trace_historical_spend