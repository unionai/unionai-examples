# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "typesafe-sdk",
# ]
# main = "route_ticket"
# params = "ticket='Our API integration started returning 500 errors.'"
# ///

"""Get a typed answer from TypeSafe in a task with your own API key."""

# {{docs-fragment task}}
import flyte

env = flyte.TaskEnvironment(
    name="typesafe-inference",
    image=flyte.Image.from_debian_base().with_pip_packages("typesafe-sdk"),
    secrets=flyte.Secret(key="typesafe-api-key", as_env_var="TYPESAFE_API_KEY"),
)


@env.task
def route_ticket(ticket: str) -> str:
    from typesafe_sdk import Choice, TypeSafeClient

    with TypeSafeClient(model="jev-latest") as client:  # Reads TYPESAFE_API_KEY.
        response = client.system_one(
            state=ticket,
            questions={
                "department": Choice(
                    instructions="Which team should handle this",
                    criteria={
                        "billing": "Payment or subscription issues",
                        "technical": "Bugs or integration problems",
                    },
                ),
            },
        )
    return response.choices["department"].choice


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(route_ticket, ticket="Our API integration started returning 500 errors.")
    print(run.url)
# {{/docs-fragment task}}
