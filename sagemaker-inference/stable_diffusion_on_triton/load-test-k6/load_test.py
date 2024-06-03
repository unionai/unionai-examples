import argparse
import os
import time

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("--tp_degree", type=int)
    parser.add_argument("--instance_type", type=str)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--vu", type=int, default=1)
    parser.add_argument("--config-file", type=str, default=None)
    parser.add_argument("--endpoint_name", type=str)
    parser.add_argument("--endpoint_region", type=str)
    parser.add_argument("--inference_component", type=str, default=None)

    return parser.parse_args()


def run_benchmark(
    token,
    instance_type,
    tp_degree,
    vu,
    endpoint_name,
    endpoint_region,
    inference_component=None,
):
    print(f"token: {token[:10] if token else None}")
    print(f"instance type: {instance_type}")
    print(f"tp_degree: {tp_degree}")
    print(f"vu: {vu}")
    print(f"endpoint_name: {endpoint_name}")
    print(f"endpoint_region: {endpoint_region}")
    print(f"inference_component: {inference_component}")

    # run benchmark
    try:
        benchmark_start_time = time.time()
        command = f"k6 run sagemaker_load.js -e ENDPOINT_NAME={endpoint_name} -e AWS_REGION={endpoint_region} -e DO_SAMPLE=0 -e VU={vu} -e AWS_ACCESS_KEY_ID={os.getenv("AWS_ACCESS_KEY_ID")} -e AWS_SECRET_ACCESS_KEY={os.getenv("AWS_SECRET_ACCESS_KEY")} -e AWS_SESSION_TOKEN={os.getenv("AWS_SESSION_TOKEN")}"

        if inference_component:
            command += f" -e INFERENCE_COMPONENT={inference_component}"

        os.system(command)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    args = parse_args()

    if args.config_file:
        with open(args.config_file, "r") as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)["configs"]

        # run each config for each vu group
        vu_group = [1, 5, 10, 20]

        # run each config for each vu group
        print(f"Running benchmark for {len(configs) * 4} configs")
        print(
            f"Expected time between: {len(configs) * 4 * 18}-{len(configs) * 4 * 27} minutes"
        )

        for config in configs:
            for vu in vu_group:
                run_benchmark(
                    token=args.token,
                    instance_type=config["instance_type"],
                    tp_degree=config["tp_degree"],
                    vu=vu,
                )

    else:
        del args.config_file
        run_benchmark(**vars(args))
