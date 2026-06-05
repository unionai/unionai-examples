"""Flyte TaskEnvironment definitions for mle-bot.

Two environments:
- tool_env: Runs the ML tools (data loading, feature engineering, training, evaluation).
            Has sklearn, xgboost, pandas, numpy, joblib.
- agent_env: Runs the orchestrating agent (OpenAI calls, Monty sandbox orchestration).
             Has openai, pydantic-monty. Depends on tool_env.
"""

import flyte

tool_env = flyte.TaskEnvironment(
    "mle-tools",
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    image=(
        flyte.Image.from_debian_base(name="mle-tools-image").with_pip_packages(
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "xgboost>=2.0.0",
            "numpy>=1.24.0",
            "joblib>=1.3.0",
        )
    ),
)

agent_env = flyte.TaskEnvironment(
    "mle-agent",
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    secrets=[flyte.Secret(key="OPENAI_API_KEY", as_env_var="OPENAI_API_KEY")],
    env_vars={"PYTHONUNBUFFERED": "1"},
    image=(
        flyte.Image.from_debian_base(name="mle-agent-image")
        .with_apt_packages("git")
        .with_pip_packages(
            "openai>=1.0.0",
            "flyte[sandbox] @ git+https://github.com/flyteorg/flyte-sdk.git@36097a64",
        )
    ),
    depends_on=[tool_env],
)
