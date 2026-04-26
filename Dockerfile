# Use NVIDIA CUDA-enabled Python base image for GPU support (optional, but available for training)
# Fallback: nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 (requires NVIDIA Container Toolkit)
# Simpler: use standard Python image for CPU-bound data processing
FROM python:3.11-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies (needed for some packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project into container
COPY . .

# Default command: bash (can override with python command)
CMD ["/bin/bash"]
