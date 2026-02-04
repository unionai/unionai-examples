"""Streamlit app for the report generator."""

# {{docs-fragment imports}}
import os

import streamlit as st
# {{/docs-fragment imports}}

# Page configuration
st.set_page_config(
    page_title="Report Generator",
    page_icon="📝",
    layout="wide",
)

st.title("Report Generator")
st.write("Generate AI-powered reports with iterative refinement.")


# {{docs-fragment load-reports}}
def load_report_from_dir(report_dir: str) -> dict | None:
    """Load a single report from a directory."""
    if not os.path.isdir(report_dir):
        return None

    report = {"path": report_dir, "name": os.path.basename(report_dir)}

    md_path = os.path.join(report_dir, "report.md")
    if os.path.exists(md_path):
        with open(md_path) as f:
            report["markdown"] = f.read()

    html_path = os.path.join(report_dir, "report.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            report["html"] = f.read()

    summary_path = os.path.join(report_dir, "summary.txt")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            report["summary"] = f.read()

    # Only return if we found at least markdown content
    return report if "markdown" in report else None


def load_all_reports() -> list[dict]:
    """Load all reports from the batch pipeline output."""
    reports_path = os.environ.get("REPORTS_PATH")
    if not reports_path or not os.path.exists(reports_path):
        return []

    reports = []

    # Check if this is a single report directory (has report.md directly)
    if os.path.exists(os.path.join(reports_path, "report.md")):
        report = load_report_from_dir(reports_path)
        if report:
            report["name"] = "Report"
            reports.append(report)
    else:
        # Batch output: scan subdirectories for reports
        for entry in sorted(os.listdir(reports_path)):
            entry_path = os.path.join(reports_path, entry)
            report = load_report_from_dir(entry_path)
            if report:
                reports.append(report)

    return reports
# {{/docs-fragment load-reports}}


# {{docs-fragment display-reports}}
reports = load_all_reports()

if reports:
    # Sidebar for report selection if multiple reports
    if len(reports) > 1:
        st.sidebar.header("Select Report")
        report_names = [f"Report {i+1}: {r['name']}" for i, r in enumerate(reports)]
        selected_idx = st.sidebar.selectbox(
            "Choose a report to view:",
            range(len(reports)),
            format_func=lambda i: report_names[i],
        )
        selected_report = reports[selected_idx]
        st.sidebar.markdown(f"**Viewing {len(reports)} reports**")
    else:
        selected_report = reports[0]

    st.header(f"Generated Report: {selected_report['name']}")

    # Summary section
    if "summary" in selected_report:
        with st.expander("Executive Summary", expanded=True):
            st.write(selected_report["summary"])

    # Tabbed view for different formats
    tab_md, tab_html = st.tabs(["Markdown", "HTML Preview"])

    with tab_md:
        st.markdown(selected_report.get("markdown", ""))

    with tab_html:
        if "html" in selected_report:
            st.components.v1.html(selected_report["html"], height=600, scrolling=True)

    # Download options
    st.subheader("Download")
    col1, col2, col3 = st.columns(3)

    with col1:
        if "markdown" in selected_report:
            st.download_button(
                label="Download Markdown",
                data=selected_report["markdown"],
                file_name="report.md",
                mime="text/markdown",
            )

    with col2:
        if "html" in selected_report:
            st.download_button(
                label="Download HTML",
                data=selected_report["html"],
                file_name="report.html",
                mime="text/html",
            )

    with col3:
        if "summary" in selected_report:
            st.download_button(
                label="Download Summary",
                data=selected_report["summary"],
                file_name="summary.txt",
                mime="text/plain",
            )

else:
    st.info("No reports generated yet. Run the batch pipeline to create reports.")
# {{/docs-fragment display-reports}}


# {{docs-fragment generation-ui}}
st.divider()
st.header("Generate New Reports")
st.write("""
To generate reports, run the batch pipeline:

```bash
uv run generate.py
```

This generates reports for multiple topics in parallel, demonstrating
how ReusePolicy efficiently handles many concurrent LLM calls.
""")

# Show pipeline parameters info
with st.expander("Pipeline Parameters"):
    st.markdown("""
    **Available parameters:**

    | Parameter | Default | Description |
    |-----------|---------|-------------|
    | `topics` | (required) | List of topics to write about |
    | `max_iterations` | 3 | Maximum refinement cycles per topic |
    | `quality_threshold` | 8 | Minimum score (1-10) to accept |

    **Example:**
    ```python
    run = flyte.run(
        report_batch_pipeline,
        topics=[
            "The Future of Renewable Energy",
            "Advances in Quantum Computing",
            "The Rise of Edge AI",
        ],
        max_iterations=3,
        quality_threshold=8,
    )
    ```
    """)
# {{/docs-fragment generation-ui}}
