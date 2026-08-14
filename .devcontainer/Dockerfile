# syntax=docker/dockerfile:1.4
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
    && rm packages.txt

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirements.txt \
    && rm requirements.txt

RUN wget -qO- cli.runpod.net | bash

# Add vscode user for the agent path through this dockerfile, devcontainer already has it
# Create .cache folder with correct permissions, otherwise compose creates it as root
ARG UID=1000
RUN id -u ${UID} >>/dev/null 2>&1 || useradd -u ${UID} -m vscode \
    && mkdir -p /home/vscode/.cache && chown ${UID}:${UID} /home/vscode/.cache \
    && mkdir -p /workspaces/worktrees && chown ${UID}:${UID} /workspaces/worktrees

FROM dev AS agent

RUN --mount=type=cache,target=/tmp/install-cache \
    curl -fsSL https://opencode.ai/install -o /tmp/install-cache/install.sh \
    && HOME=/home/vscode bash /tmp/install-cache/install.sh

USER vscode