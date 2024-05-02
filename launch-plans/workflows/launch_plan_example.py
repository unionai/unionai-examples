from flytekit import LaunchPlan, FixedRate
from datetime import timedelta
from .workflow_example import my_workflow

LaunchPlan.get_or_create(
    workflow=my_workflow,
    name="my_workflow_custom_lp",
    fixed_inputs={"a": 6},
    default_inputs={"b": 4, "c": 5}
)
