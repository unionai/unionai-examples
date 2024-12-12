# SageMaker Inference: XGBoost & FastAPI

1. Ensure you are authenticated to Amazon Elastic Container Registry (ECR) before proceeding, as the SageMaker inference image will be pushed to ECR.
2. Use the following command to run the workflow:

   ```
   AWS_REGISTRY=<AWS_REGISTRY> EXECUTION_ROLE_ARN=<YOUR_EXECUTION_ROLE_ARN> union run --remote deploy.py xgboost_fastapi_wf
   ```

3. Use the `invoke_endpoint.py` script to invoke the endpoint:

   ```
   python invoke_endpoint.py --region_name "us-east-2" --endpoint_name "<YOUR_ENDPOINT_NAME>"
   ```
