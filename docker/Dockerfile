FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt update
RUN apt install git build-essential ninja-build zsh wget -y

# Install Python3 and python-is-python3
RUN apt install python3 python3-pip python-is-python3 -y

# Torch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Requirements
RUN pip install \
    networkx \
    sqlglot==18.15.0 \
    loguru \
    matplotlib \
    peewee \
    psycopg2-binary \
    pydot \
    hypothesis \
    localstack \
    boto3 \
    psycopg[binary] \
    seaborn \
    bokeh \
    awscli-local \
    celery \
    # Lolbo reqs
    botorch \
    gpytorch \
    wandb \
    numpy \
    tqdm \
    lightning \
    scipy \
    datasets \
    fire

# User stuff
WORKDIR /workspace
ENV USER=anon

RUN git config --global --add safe.directory /workspace