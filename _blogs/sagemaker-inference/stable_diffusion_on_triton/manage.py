#!/usr/bin/env python


import time
import typing

import boto3
import typer

app = typer.Typer()


@app.command()
def status(model_name: str):
    client = boto3.client('sagemaker', 'us-east-2')
    response = client.describe_endpoint(
        EndpointName=model_name,
    )
    print(response['EndpointStatus'])


def recursive_update(d: typing.Any, u: dict) -> typing.Any:
    if d is None:
        return None
    if isinstance(d, str):
        return d.format(**u)
    if isinstance(d, list):
        return [recursive_update(i, u) for i in d]
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = recursive_update(d.get(k), u)
    return d


def apply(method: str, j: dict, model_name: str, model_path: typing.Optional[str],
          image: str = "356633062068.dkr.ecr.us-east-2.amazonaws.com/flytecookbook:ketan-test-1",
          service: str = "sagemaker", region: str = "us-east-2") -> dict:
    client = boto3.client(service, region)
    j = recursive_update(j, {"model_name": model_name, "model_path": model_path, "image": image})
    print(j)
    return getattr(client, method)(**j)


@app.command()
def delete(model_name: str):
    client = boto3.client('sagemaker', 'us-east-2')
    try:
        apply("delete_endpoint", {
            "EndpointName": "{model_name}",
        }, model_name, None)
    except:
        pass
    try:
        apply("delete_endpoint_config", {
            "EndpointConfigName": "{model_name}",
        }, model_name, None)
    except:
        pass
    try:
        apply("delete_model", {
            "ModelName": "{model_name}",
        }, model_name, None)
    except:
        pass


if __name__ == "__main__":
    app()
