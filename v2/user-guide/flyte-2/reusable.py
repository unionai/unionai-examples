env = flyte.TaskEnvironment(
    name="reusable",
    resources=flyte.Resources(memory="500Mi", cpu=1),
    reusable=flyte.ReusePolicy(
        replicas=4,  # Min of 2 replacas are needed to ensure no-starvation of tasks.
        idle_ttl=300,
    ),
    image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse==0.1.3"),
)
