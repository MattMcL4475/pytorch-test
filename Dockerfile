# Use NVIDIA PyTorch base image with CUDA support
FROM nvcr.io/nvidia/pytorch:25.09-py3

# Set working directory
WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv (this will use the exact versions from uv.lock)
RUN uv sync --frozen

# Copy application code
COPY test.py ./
