ARG CUDA=11.8.0
FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu22.04

WORKDIR /root
ENV VENV /opt/venv
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common gcc && add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y python3.10 python3.10-distutils python3.10-venv build-essential git ffmpeg

ENV VENV /opt/venv
RUN python3.10 -m venv ${VENV}
ENV PATH="${VENV}/bin:$PATH"

# Install Python dependencies
COPY requirements.txt /tmp
RUN pip3 install --upgrade pip --no-cache-dir \
    && pip3 install -r /tmp/requirements.txt --no-cache-dir \
    && pip3 install --upgrade --no-cache-dir \
      jax[cuda11_pip] \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Copy the actual code
COPY whisper /root/whisper
COPY whisper-models /root/whisper-models