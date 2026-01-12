# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==2.0.0b45",
# ]
# ///

import flyte
import flyte.app

# {{docs-fragment basic-scaling}}
# Basic example: scale from 0 to 1 replica
app_env = flyte.app.AppEnvironment(
    name="autoscaling-app",
    scaling=flyte.app.Scaling(
        replicas=(0, 1),  # Scale from 0 to 1 replica
        scaledown_after=300,  # Scale down after 5 minutes of inactivity
    ),
    # ...
)
# {{/docs-fragment basic-scaling}}

# {{docs-fragment always-on}}
# Always-on app
app_env2 = flyte.app.AppEnvironment(
    name="always-on-api",
    scaling=flyte.app.Scaling(
        replicas=(1, 1),  # Always keep 1 replica running
        # scaledown_after is ignored when min_replicas > 0
    ),
    # ...
)
# {{/docs-fragment always-on}}

# {{docs-fragment scale-to-zero}}
# Scale-to-zero app
app_env3 = flyte.app.AppEnvironment(
    name="scale-to-zero-app",
    scaling=flyte.app.Scaling(
        replicas=(0, 1),  # Can scale down to 0
        scaledown_after=600,  # Scale down after 10 minutes of inactivity
    ),
    # ...
)
# {{/docs-fragment scale-to-zero}}

# {{docs-fragment high-availability}}
# High-availability app
app_env4 = flyte.app.AppEnvironment(
    name="ha-api",
    scaling=flyte.app.Scaling(
        replicas=(2, 5),  # Keep at least 2, scale up to 5
        scaledown_after=300,  # Scale down after 5 minutes
    ),
    # ...
)
# {{/docs-fragment high-availability}}

# {{docs-fragment burstable}}
# Burstable app
app_env5 = flyte.app.AppEnvironment(
    name="bursty-app",
    scaling=flyte.app.Scaling(
        replicas=(1, 10),  # Start with 1, scale up to 10 under load
        scaledown_after=180,  # Scale down quickly after 3 minutes
    ),
    # ...
)
# {{/docs-fragment burstable}}

