FROM runpod/base:1.1.0-rc.154-ubuntu2404

ENV MLDB_DATA_ROOT=/workspace/results
ENV PLOTS_ROOT=/workspace/plots

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir git+https://github.com/michaelschott00/mldb