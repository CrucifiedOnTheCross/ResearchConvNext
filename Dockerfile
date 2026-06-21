FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements-docker.txt .
RUN python -m pip install -r requirements-docker.txt
COPY . .
RUN python -m compileall -q src scripts && mkdir -p /workspace/data /workspace/runs
ENTRYPOINT ["python"]
CMD ["scripts/train.py", "--config", "configs/default.yaml"]
