FROM runpod/pytorch:1.1.0-rc.154-cu1290-torch290-ubuntu2404

WORKDIR /microstructures

ENV DEBIAN_FRONTEND=noninteractive
ENV MLDB_DATA_ROOT=/workspace/results
ENV DATA_ROOT=/workspace
ENV PLOTS_ROOT=/workspace/plots

COPY packages.txt .
RUN apt update -y \
    && xargs apt -y install < packages.txt \
    && rm packages.txt \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

# Bake models into the image to avoid downloading them each time
COPY ./scripts/preload_models.py .
RUN python3 -m preload_models && rm -r __pycache__ preload_models.py

COPY ./transfer_learning ./transfer_learning
COPY ./configs ./configs
COPY ./scripts/**/* .
RUN find . -type f  \( -name '*.sh' -o -name '*.py' \) -exec chmod +x '{}' \;

ENTRYPOINT ["/bin/bash"]
