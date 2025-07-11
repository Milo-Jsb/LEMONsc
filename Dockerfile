# Use multi-stage build for a smaller final image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime as builder

# Set environment variables in one layer. Set CUDA environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set the working directory
WORKDIR /workspace

# Install system dependencies and cleanup in a single layer. Group all apt-get commands to reduce layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1-mesa-glx \
    python3-pip \
    nvidia-cuda-toolkit \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file first, separately
COPY requirements.txt .

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8883 available to the world outside this container
EXPOSE 8883