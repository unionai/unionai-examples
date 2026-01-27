# /// script
# requires-python = "==3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "streamlit>=1.41.0",
#    "openai>=1.0.0",
#    "pydantic>=2.0.0",
# ]
# ///

"""
Report Generator App Deployment

Deploys a Streamlit application that:
- Allows users to generate reports on custom topics
- Connects to the report generation pipeline
- Displays generated reports with download options
"""

# {{docs-fragment imports}}
import flyte
from flyte.app import AppEnvironment, Parameter, RunOutput
# {{end-fragment}}

# {{docs-fragment app-env}}
# Define the app environment
env = AppEnvironment(
    name="report-generator-app",
    description="Interactive report generator with AI-powered refinement",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "streamlit>=1.41.0",
        "openai>=1.0.0",
        "pydantic>=2.0.0",
    ),
    secrets=[flyte.Secret(key="openai-api-key", as_env_var="OPENAI_API_KEY")],
    args=["streamlit", "run", "app.py", "--server.port", "8080"],
    port=8080,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    parameters=[
        # Connect to the latest report pipeline output
        Parameter(
            name="latest_report",
            value=RunOutput(
                task_name="report-generator.report_pipeline",
                type="directory",
            ),
            download=True,
            env_var="LATEST_REPORT_PATH",
        ),
    ],
    include=["app.py", "prompts.py"],
    requires_auth=False,
)
# {{end-fragment}}

# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config()

    # Deploy the report generator app
    print("Deploying report generator app...")
    deployment = flyte.serve(env)
    print(f"App deployed at: {deployment.url}")
# {{end-fragment}}
