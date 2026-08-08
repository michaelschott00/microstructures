FROM mcr.microsoft.com/devcontainers/python:3.12

WORKDIR /workspaces/microstructures

ENV DEBIAN_FRONTEND=noninteractive
ENV MLDB_DATA_ROOT="/workspaces/microstructures/results"
ENV DATA_ROOT="/workspaces/microstructures"

# Install opencv dependencies
RUN apt update -y \
    && apt install -y g++ libgl1 libglib2.0-0 \
    && apt autoclean \
    && apt autoremove \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --root-user-action ignore --no-cache-dir --upgrade pip \
    && pip install --root-user-action ignore --no-cache-dir -r requirements.txt \
    && rm requirements.txt

# Restrict files and directories the agent can access
# -u 1001 makes agent distinct from my host system user, similarly for -g 2000
# -L prevents the agent from using its password at all, disabling sudo as a side-effect
RUN groupadd -g 2000 -U vscode contributors \
    && useradd -g 2000 -u 1001 -m --no-user-group -s /usr/sbin/nologin agent \
    && usermod -L agent

# Install opencode as agent user so it has access
USER agent
RUN curl -fsSL https://opencode.ai/install | bash
