"""Streamlit app for the report generator."""

# {{docs-fragment imports}}
import os

import streamlit as st
# {{end-fragment}}

# Page configuration
st.set_page_config(
    page_title="Report Generator",
    page_icon="📝",
    layout="wide",
)

st.title("Report Generator")
st.write("Generate AI-powered reports with iterative refinement.")


# {{docs-fragment load-report}}
def load_latest_report():
    """Load the latest generated report from the pipeline output."""
    report_path = os.environ.get("LATEST_REPORT_PATH")
    if not report_path:
        return None, None, None

    report_md = None
    report_html = None
    summary = None

    md_path = os.path.join(report_path, "report.md")
    if os.path.exists(md_path):
        with open(md_path) as f:
            report_md = f.read()

    html_path = os.path.join(report_path, "report.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            report_html = f.read()

    summary_path = os.path.join(report_path, "summary.txt")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = f.read()

    return report_md, report_html, summary
# {{end-fragment}}


# {{docs-fragment display-report}}
# Display latest report if available
report_md, report_html, summary = load_latest_report()

if report_md:
    st.header("Latest Generated Report")

    # Summary section
    if summary:
        with st.expander("Executive Summary", expanded=True):
            st.write(summary)

    # Tabbed view for different formats
    tab_md, tab_html = st.tabs(["Markdown", "HTML Preview"])

    with tab_md:
        st.markdown(report_md)

    with tab_html:
        st.components.v1.html(report_html, height=600, scrolling=True)

    # Download options
    st.subheader("Download")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="Download Markdown",
            data=report_md,
            file_name="report.md",
            mime="text/markdown",
        )

    with col2:
        if report_html:
            st.download_button(
                label="Download HTML",
                data=report_html,
                file_name="report.html",
                mime="text/html",
            )

    with col3:
        if summary:
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain",
            )

else:
    st.info("No reports generated yet. Run the report pipeline to create your first report.")
# {{end-fragment}}


# {{docs-fragment generation-ui}}
st.divider()
st.header("Generate New Report")
st.write("""
To generate a new report, run the pipeline with your topic:

```bash
uv run generate.py
```

Or modify the topic in `generate.py` and run the pipeline to create a custom report.
The app will automatically display the latest generated report.
""")

# Show pipeline parameters info
with st.expander("Pipeline Parameters"):
    st.markdown("""
    **Available parameters:**

    | Parameter | Default | Description |
    |-----------|---------|-------------|
    | `topic` | (required) | The topic to write about |
    | `max_iterations` | 3 | Maximum refinement cycles |
    | `quality_threshold` | 8 | Minimum score (1-10) to accept |

    **Example:**
    ```python
    run = flyte.run(
        report_pipeline,
        topic="The Future of Renewable Energy",
        max_iterations=5,
        quality_threshold=9,
    )
    ```
    """)
# {{end-fragment}}
