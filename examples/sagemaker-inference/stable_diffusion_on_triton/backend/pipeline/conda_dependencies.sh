# should be built in sagemaker-tritonserver container
#!/bin/bash

conda create -y -n hf_env python=3.10.12
conda activate hf_env
export PYTHONNOUSERSITE=True
conda install -c conda-forge conda-pack
conda install -c conda-forge numpy==1.26.2
pip install torch==2.2.2
pip install transformers==4.39.2
pip install ftfy==6.2.0 
pip install scipy==1.10.1
pip install accelerate==0.28.0
pip install diffusers==0.27.2
pip install peft==0.10.0
pip install transformers[onnxruntime]

conda-pack
