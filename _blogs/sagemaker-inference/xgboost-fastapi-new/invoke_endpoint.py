import boto3
import argparse

def main(region_name, endpoint_name, input_data):
    # Initialize the SageMaker Runtime client
    sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=region_name)
    
    # Gets inference from the model hosted at the specified endpoint:
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=bytes(input_data, 'utf-8')
    )
    
    # Decodes and prints the response body:
    print(response['Body'].read().decode('utf-8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invoke a SageMaker endpoint and print the response.")
    parser.add_argument("--region_name", required=True, help="AWS region where the SageMaker endpoint is hosted.")
    parser.add_argument("--endpoint_name", required=True, help="Name of the SageMaker endpoint.")
    parser.add_argument(
        "--input_data", 
        default='[6, 148, 72, 35, 0, 33.6, 0.627, 50]', 
        help="Input data to send to the endpoint, formatted as a string. Default is '[6, 148, 72, 35, 0, 33.6, 0.627, 50]'."
    )
    
    args = parser.parse_args()
    main(region_name=args.region_name, endpoint_name=args.endpoint_name, input_data=args.input_data)
