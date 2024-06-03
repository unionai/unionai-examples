# Load Testing

```bash
python load_test.py \
  --instance_type ml.g4dn.4xlarge \
  --tp_degree 1 \
  --vu 5 \
  --token $(cat ~/.cache/huggingface/token) \
  --endpoint_name stable-diffusion-endpoint \
  --endpoint_region us-east-2
```
| TensorRT | ONNX |
| :------: | :--: |
| ![stable-diffusion-endpoint](https://github.com/unionai-oss/sagemaker-agent-examples/assets/27777173/7fc47ce6-db4a-463e-b846-22119d9b54bd) | ![stable-diffusion-endpoint-onnx](https://github.com/unionai-oss/sagemaker-agent-examples/assets/27777173/6c018d21-4c0e-4eaa-81ff-aca4943c5cc7) |

