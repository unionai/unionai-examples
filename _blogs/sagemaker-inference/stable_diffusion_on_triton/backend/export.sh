#!/bin/bash

set -e

cd /root
python export.py $3 $2

# Accelerating VAE with TensorRT
/usr/src/tensorrt/bin/trtexec --onnx=vae.onnx --saveEngine="$1" --minShapes=latent_sample:1x4x64x64 --optShapes=latent_sample:4x4x64x64 --maxShapes=latent_sample:8x4x64x64 --fp16 --verbose
