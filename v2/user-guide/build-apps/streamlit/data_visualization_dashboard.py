# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b45",
#    "streamlit",
#    "pandas",
#    "numpy",
# ]
# ///

"""A data visualization dashboard example using Streamlit."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import flyte
import flyte.app


# {{docs-fragment streamlit-app}}
def main():
    st.set_page_config(page_title="Sales Dashboard", page_icon="📊")

    st.title("Sales Dashboard")

    # Load data
    @st.cache_data
    def load_data():
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=100, freq="D"),
            "sales": np.random.randint(1000, 5000, 100),
        })

    data = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start date", value=data["date"].min())
    end_date = st.sidebar.date_input("End date", value=data["date"].max())

    # Filter data
    filtered_data = data[
        (data["date"] >= pd.Timestamp(start_date)) &
        (data["date"] <= pd.Timestamp(end_date))
    ]

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sales", f"${filtered_data['sales'].sum():,.0f}")
    with col2:
        st.metric("Average Sales", f"${filtered_data['sales'].mean():,.0f}")
    with col3:
        st.metric("Days", len(filtered_data))

    # Chart
    st.line_chart(filtered_data.set_index("date")["sales"])

# {{/docs-fragment streamlit-app}}

# {{docs-fragment app-env}}
file_name = Path(__file__).name
app_env = flyte.app.AppEnvironment(
    name="sales-dashboard",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "streamlit==1.41.1",
        "pandas==2.2.3",
        "numpy==2.2.3",
    ),
    args=["streamlit run", file_name, "--server.port", "8080", "--", "--server"],
    port=8080,
    resources=flyte.Resources(cpu="2", memory="2Gi"),
    requires_auth=False,
)
# {{/docs-fragment app-env}}

# {{docs-fragment serve}}
if __name__ == "__main__":
    import logging
    import sys

    if "--server" in sys.argv:
        main()
    else:
        flyte.init_from_config(
            root_dir=Path(__file__).parent,
            log_level=logging.DEBUG,
        )
        app = flyte.serve(app_env)
        print(f"Dashboard URL: {app.url}")
# {{/docs-fragment serve}}
