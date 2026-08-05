FROM nvidia/cuda:13.3.1-cudnn-runtime-ubuntu26.04

# This ensures Python output is immediately visible in logs
ENV PYTHONUNBUFFERED=1

WORKDIR /microstructures

# software-properties-common required for PPAs
# 'ppa:deadsnakes/ppa' required for older python versions
# DEBIAN_FRONTEND=noninteractive TZ='Europe/Berlin' required for noninteractive install of python
# /var/lib/apt/lists/* contains cached files that would infalte image size
RUN apt update -y \
    && apt install -y software-properties-common \
    && add-apt-repository -y 'ppa:deadsnakes/ppa' \
    && DEBIAN_FRONTEND=noninteractive TZ='Europe/Berlin' \
        apt install -y --no-install-recommends tzdata

RUN apt update -y \
    && apt install -y python3.12 python3-pip git g++ libgl1 libglib2.0-0 \
    && apt autoclean \
    && apt autoremove \
    && rm -rf /var/lib/apt/lists/*

# It's already an isolated container so breaking system packages is fine
COPY ./requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Bake models into the image to avoid downloading them each time
COPY ./infra/preload_models.py .
RUN python3 -m preload_models && rm -r __pycache__ preload_models.py

COPY ./transfer_learning ./transfer_learning
COPY ./configs ./configs
COPY ./scripts/**/* .

RUN find . -type f  \( -name '*.sh' -o -name '*.py' \) -exec chmod +x '{}' \;

ENV DATA_ROOT=/microstructures/results

ENTRYPOINT ["/bin/bash"]
