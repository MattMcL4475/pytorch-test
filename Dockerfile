# Use NVIDIA PyTorch base image with CUDA support
# cached in my ACR for faster pulls
# az acr import -n msftmattmcl --source nvcr.io/nvidia/pytorch:25.08-py3 --image nvidia/pytorch:25.08-py3
#FROM msftmattmcl.azurecr.io/nvidia/pytorch:25.08-py3
#FROM nvcr.io/nvidia/pytorch:25.09-py3
FROM msftmattmcl.azurecr.io/nvidia/pytorch:25.08-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml uv.lock ./
COPY test.py ./
COPY gpu-audio.py ./

# Install Python dependencies using uv (if available) or pip
RUN pip install --no-cache-dir torch>=2.2 numpy>=1.24
