ARG BASE_IMAGE=python:3.12
FROM ${BASE_IMAGE} AS dev

WORKDIR /workspaces/microstructures

ENV DEBIAN_FRONTEND=noninteractive
ENV MLDB_DATA_ROOT="/workspaces/microstructures/results"
ENV DATA_ROOT="/workspaces/microstructures"
ENV PLOTS_ROOT="/workspaces/microstructures/plots"

COPY packages.txt .
RUN apt update -y \
    && xargs apt -y install < packages.txt \
    && rm packages.txt \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

RUN wget -qO- cli.runpod.net | bash

# Add vscode user for the agent path through this dockerfile, devcontainer already has it
# Create .cache folder with correct permissions, otherwise compose creates it as root
ARG UID=1000
RUN id -u ${UID} >>/dev/null 2>&1 || useradd -u ${UID} -m vscode \
    && mkdir -p /home/vscode/.cache && chown ${UID}:${UID} /home/vscode/.cache

FROM dev AS agent

USER vscode

RUN curl -fsSL https://opencode.ai/install | bash
ENV PATH="$PATH:/home/vscode/.opencode/bin"