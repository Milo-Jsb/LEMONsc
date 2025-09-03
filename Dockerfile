# Use multi-stage build for a smaller final image
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel AS builder

# Environment variables for smoother builds & CUDA support
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# Set the working directory
WORKDIR /workspace

# Fix time synchronization issue and install system dependencies
RUN echo "Acquire::Check-Valid-Until \"false\";" > /etc/apt/apt.conf.d/99no-check-valid && \
    apt-get update --allow-releaseinfo-change && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1-mesa-glx \
    python3-pip \
    nvidia-cuda-toolkit \
    texlive \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file first, separately
COPY requirements.txt .

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user with same UID/GID as host
ARG HOST_UID=1000
ARG HOST_GID=1000

RUN groupadd -g ${HOST_GID} appuser && \
    useradd -m -u ${HOST_UID} -g ${HOST_GID} appuser && \
    echo "appuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Make port 8883 available to the world outside this container
EXPOSE 8889

# Final command
CMD ["/bin/bash"]