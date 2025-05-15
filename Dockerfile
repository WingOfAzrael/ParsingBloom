###############################################################################
#  ParsingBloom • GPU image  (CUDA 12.6   +   Python 3.10)                    #
#  Host requirements:                                                         #
#    • NVIDIA driver ≥ 550.xx                                                #
#    • nvidia-container-toolkit installed                                    #
###############################################################################
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS base

# ---------- system packages --------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3-venv \
        build-essential gcc g++ git \
        tesseract-ocr poppler-utils \
        libglib2.0-0 libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------- Poetry + project deps (cached) -----------------------------------
RUN pip3 install --no-cache-dir --upgrade pip poetry==1.8.2

WORKDIR /app
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction --no-ansi --no-cache --no-parallel

# ---------- Copy source last (fast rebuilds) ---------------------------------
COPY . .

# ---------- Runtime env ------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/models/hf_cache \
    PIP_DISABLE_PIP_VERSION_CHECK=1

HEALTHCHECK CMD python3 -c "import importlib; importlib.import_module('pipeline.scheduler')" || exit 1
ENTRYPOINT ["python3", "-m", "pipeline.scheduler"]
